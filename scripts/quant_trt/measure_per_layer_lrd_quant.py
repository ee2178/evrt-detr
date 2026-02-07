#!/usr/bin/env python3
"""
Option A: Per-Conv2d breakdown (best-effort) using:
- ONNX post-processing: rename Conv nodes to module paths based on conv weight initializer names
- trtexec exportProfile JSON for per-TRT-layer timing
- TRT engine inspector JSON for per-TRT-layer precision
- Bucket TRT layers back to Conv2d modules via substring + longest-match

Outputs:
- summary.csv
- layer_benefit_fp16_vs_int8.csv                 (TRT-layer level)
- conv2d_benefit_fp16_vs_int8.csv                (Conv2d-bucket level, best-effort)
- renamed ONNX file used for trtexec

Assumptions:
- PyTorch ONNX exporter names conv weights like: "backbone.layerX.Y.convZ.weight"
  (this is typical; if yours differs, adjust extract_module_from_weight_name()).

"""

import argparse
import importlib
import json
import logging
import os
import re
import subprocess
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch

from evlearn.eval.eval import load_model

LOGGER = logging.getLogger("trt_conv2d_breakdown")


# ----------------------------
# trtexec harness (reference-compatible)
# ----------------------------
def run_trtexec(
    onnx_path: Optional[str] = None,
    load_engine: Optional[str] = None,
    precision: str = "fp32",
    calib_cache: Optional[str] = None,
    warmup_ms: int = 200,
    duration_sec: int = 3,
    iterations: Optional[int] = None,
    use_cuda_graph: bool = False,
    enable_profiling: bool = False,
    profile_json_path: Optional[str] = None,
    save_engine: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[Optional[float], str]:
    cmd = ["/usr/src/tensorrt/bin/trtexec"]

    if onnx_path:
        cmd.append(f"--onnx={onnx_path}")
    elif load_engine:
        cmd.append(f"--loadEngine={load_engine}")
    else:
        raise ValueError("Must specify either onnx_path or load_engine")

    prec = precision.lower()
    if prec == "fp16":
        cmd.append("--fp16")
    elif prec == "int8":
        cmd.append("--int8")
        if calib_cache and os.path.exists(calib_cache):
            cmd.append(f"--calib={calib_cache}")
        else:
            LOGGER.warning("INT8 calib cache not found: %s", calib_cache)

    cmd.append(f"--warmUp={warmup_ms}")
    cmd.append(f"--duration={duration_sec}")
    if iterations is not None:
        cmd.append(f"--iterations={iterations}")

    if use_cuda_graph:
        cmd.append("--useCudaGraph")

    if enable_profiling:
        cmd.append("--dumpProfile")
        cmd.append("--separateProfileRun")
        cmd.append("--profilingVerbosity=detailed")
        if profile_json_path:
            cmd.append(f"--exportProfile={profile_json_path}")

    if save_engine:
        cmd.append(f"--saveEngine={save_engine}")

    if verbose:
        cmd.append("--verbose")

    LOGGER.info("Running: %s", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        raise RuntimeError(f"trtexec failed (code {r.returncode})\n{out}")

    latency_ms = parse_latency_from_trtexec_output(out)
    return latency_ms, out


def parse_latency_from_trtexec_output(output: str) -> Optional[float]:
    patterns = [
        r"mean\s*=\s*([\d.]+)\s*ms",
        r"Average\s*=\s*([\d.]+)\s*ms",
    ]
    for pat in patterns:
        m = re.search(pat, output)
        if m:
            return float(m.group(1))

    lines = output.split("\n")
    for line in reversed(lines):
        if "mean" in line.lower() and "ms" in line.lower():
            nums = re.findall(r"([\d.]+)\s*ms", line)
            if nums:
                return float(nums[0])

    return None


# ----------------------------
# Parse trtexec profile json -> per TRT layer avg time
# ----------------------------
def parse_trtexec_profile_json(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    layers = None
    if isinstance(obj, dict):
        if isinstance(obj.get("layers"), list):
            layers = obj["layers"]
        elif isinstance(obj.get("profiles"), list):
            layers = obj["profiles"]

    if layers is None:
        raise RuntimeError(f"Unrecognized trtexec profile JSON schema: {path}")

    rows = []
    for L in layers:
        if not isinstance(L, dict):
            continue
        name = L.get("name") or L.get("layerName") or L.get("layer_name")
        avg = L.get("averageMs") or L.get("avgMs") or L.get("average") or L.get("average_ms")
        if avg is None:
            avg_us = L.get("averageUs") or L.get("avgUs")
            if avg_us is not None:
                avg = float(avg_us) / 1000.0
        if name is None or avg is None:
            continue
        rows.append({"layer_name": str(name), "avg_ms": float(avg)})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No layer timing records parsed from: {path}")
    return df.groupby("layer_name", as_index=False)["avg_ms"].mean().sort_values("avg_ms", ascending=False)


# ----------------------------
# TRT Engine Inspector -> per TRT layer precision/type
# ----------------------------
def dump_engine_inspector(engine_path: str, out_json_path: str):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    insp = engine.create_engine_inspector()
    s = insp.get_engine_information(trt.LayerInformationFormat.JSON)
    with open(out_json_path, "w", encoding="utf-8") as f:
        f.write(s)


def parse_engine_inspector_precisions(inspector_json_path: str) -> pd.DataFrame:
    with open(inspector_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    layers = obj.get("layers", [])
    rows = []
    for L in layers:
        rows.append(
            {
                "layer_name": L.get("name"),
                "layer_type": L.get("type"),
                "precision": L.get("precision"),
                "output_type": L.get("outputType") or L.get("output_type"),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------
# ONNX post-processing: rename Conv nodes to module path from weight initializer name
# ----------------------------
def extract_module_from_weight_name(weight_name: str) -> Optional[str]:
    """
    Typical names:
      'backbone.layer2.0.conv2.weight'
      'model.backbone.layer1.0.conv1.weight'
      'backbone.layer3.5.downsample.0.weight'

    Return the module path without trailing ".weight" (or similar).
    """
    if not weight_name:
        return None

    # Normalize to dotted paths; some exporters use '/'.
    w = weight_name.replace("/", ".")

    # Strip common suffixes
    for suf in [".weight", ".bias", ":weight", ":bias"]:
        if w.endswith(suf):
            w = w[: -len(suf)]
            break

    # Heuristic: keep from "backbone." onwards if present (helps avoid prefixes)
    idx = w.find("backbone.")
    if idx >= 0:
        w = w[idx:]

    # Require it to look like a module path
    if "." not in w:
        return None
    return w


def rename_onnx_conv_nodes(onnx_in: str, onnx_out: str) -> int:
    """
    Rename Conv nodes using their weight initializer name (node.input[1]).
    Also renames the node outputs (and updates downstream inputs) to keep graph consistent.

    Returns number of Conv nodes renamed.
    """
    import onnx

    m = onnx.load(onnx_in)
    g = m.graph

    out_rename_map: Dict[str, str] = {}
    renamed = 0

    for node in g.node:
        if node.op_type != "Conv":
            continue
        if len(node.input) < 2:
            continue

        weight_in = node.input[1]
        mod = extract_module_from_weight_name(weight_in)
        if not mod:
            continue

        old_name = node.name if node.name else "(unnamed)"
        node.name = mod

        # Rename its first output tensor (common case: single output)
        if len(node.output) >= 1:
            old_out = node.output[0]
            new_out = mod + ":0"
            if old_out != new_out:
                out_rename_map[old_out] = new_out
                node.output[0] = new_out

        renamed += 1
        if renamed <= 5:
            LOGGER.info("Renamed Conv node '%s' -> '%s' (weight=%s)", old_name, mod, weight_in)

    # Update downstream inputs
    if out_rename_map:
        for node in g.node:
            for i, inp in enumerate(node.input):
                if inp in out_rename_map:
                    node.input[i] = out_rename_map[inp]

        # Update graph outputs too (rare, but safe)
        for out in g.output:
            if out.name in out_rename_map:
                out.name = out_rename_map[out.name]

    onnx.save(m, onnx_out)
    return renamed


# ----------------------------
# Conv2d inventory from model (so we know valid module names)
# ----------------------------
def get_conv_module_names_from_backbone(model) -> List[str]:
    import torch.nn as nn
    from torch.nn import Conv2d

    names = []

    def traverse(mod: nn.Module, prefix: str):
        for n, ch in mod.named_children():
            full = f"{prefix}.{n}" if prefix else n
            if isinstance(ch, Conv2d):
                names.append(full)
            else:
                traverse(ch, full)

    # same backbone discovery logic you used
    backbone = None
    if hasattr(model, "_nets") and hasattr(model._nets, "backbone"):
        backbone = model._nets.backbone
        traverse(backbone, "backbone")
    elif hasattr(model, "backbone") and isinstance(model.backbone, nn.Module):
        traverse(model.backbone, "backbone")
    else:
        raise RuntimeError("Could not locate backbone module for conv name extraction")

    # sort longest-first helps matching
    names = sorted(set(names), key=len, reverse=True)
    return names


# ----------------------------
# Bucketing: map TRT layer name -> best matching conv module name
# ----------------------------
def best_match_conv(trt_layer_name: str, conv_names: List[str]) -> Optional[str]:
    """
    Best-effort:
      - exact substring match on the longest conv name
      - else None
    """
    if not trt_layer_name:
        return None
    s = trt_layer_name.replace("/", ".")
    for cn in conv_names:
        if cn in s:
            return cn
    return None


def conv_bucket_report(
    df_layer: pd.DataFrame,
    conv_names: List[str],
    fp16_col: str,
    int8_col: str,
    precision_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    df_layer: contains TRT-layer rows with timing columns
    conv_names: backbone conv module names (e.g., backbone.layer2.0.conv2)

    Produces one row per conv bucket with summed timing and a crude INT8 coverage indicator.
    """
    df = df_layer.copy()
    df["conv_bucket"] = df["layer_name"].apply(lambda x: best_match_conv(str(x), conv_names))

    # Keep only matched rows (you can also keep unmatched in a separate table if useful)
    df = df[~df["conv_bucket"].isna()].copy()

    agg = {
        fp16_col: "sum",
        int8_col: "sum",
        "layer_name": "count",
    }
    if precision_col and precision_col in df.columns:
        # fraction of layer rows that are INT8 within this conv bucket
        df["_is_int8"] = df[precision_col].astype(str).str.upper().eq("INT8").astype(float)
        agg["_is_int8"] = "mean"

    out = df.groupby("conv_bucket", as_index=False).agg(agg).rename(columns={"layer_name": "num_trt_layers"})
    out["speedup_fp16_over_int8"] = out[fp16_col] / out[int8_col]
    if "_is_int8" in out.columns:
        out = out.rename(columns={"_is_int8": "fraction_trt_layers_int8"})
    out = out.sort_values("speedup_fp16_over_int8", ascending=False)
    return out


# ----------------------------
# INT8 calibration cache (keep your behavior: random)
# ----------------------------
def generate_int8_calib_cache_random(
    onnx_path: str,
    calib_cache_path: str,
    batch_size: int,
    input_shape_chw: Tuple[int, int, int],
    num_batches: int,
    workspace_gb: float = 4.0,
):
    import numpy as np
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401

    class RandomInt8Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.current = 0
            self.d_input = cuda.mem_alloc(int(batch_size * np.prod(input_shape_chw) * np.float32().nbytes))

        def get_batch_size(self):
            return batch_size

        def get_batch(self, names):
            if self.current >= num_batches:
                return None
            data = np.random.randn(batch_size, *input_shape_chw).astype(np.float32)
            cuda.memcpy_htod(self.d_input, data)
            self.current += 1
            return [int(self.d_input)]

        def read_calibration_cache(self):
            if os.path.exists(calib_cache_path):
                with open(calib_cache_path, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            with open(calib_cache_path, "wb") as f:
                f.write(cache)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            msgs = "\n".join([str(parser.get_error(i)) for i in range(parser.num_errors)])
            raise RuntimeError(f"ONNX parse failed:\n{msgs}")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1024**3)))

    config.int8_calibrator = RandomInt8Calibrator()

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Failed to build INT8 engine during calibration cache generation")

    LOGGER.info("Wrote INT8 calib cache: %s", calib_cache_path)


# ----------------------------
# CLI / main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser()

    # model + lrd
    p.add_argument("model_dir", type=str)
    p.add_argument("--epoch", type=int, default=None)
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--lrd-module", type=str, default="real_lrd_resnet",
                   help="Module name for your LRD script (apply_lrd_to_model etc.)")
    p.add_argument("--ratio", type=float, default=0.5)
    p.add_argument("--ratio-mode", type=str, default="param", choices=["param", "rank"])
    p.add_argument("--scheme", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--whitelist-file", type=str, default=None)
    p.add_argument("--skip-1x1", action="store_true")
    p.add_argument("--skip-if-no-mac-reduction", action="store_true")
    p.add_argument("--skip-if-rank-gt-out", action="store_true")

    # export
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--input-shape", type=str, default="20,256,320", help="C,H,W")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--onnx-prefix", type=str, default="model")

    # trtexec
    p.add_argument("--warmup-ms", type=int, default=200)
    p.add_argument("--duration-sec", type=int, default=3)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--use-cuda-graph", action="store_true")
    p.add_argument("--enable-profiling", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--workspace-gb", type=float, default=4.0)

    # int8
    p.add_argument("--int8-calibration-batches", type=int, default=10)

    # output
    p.add_argument("--output-dir", type=str, default="trt_conv2d_breakdown_out")
    p.add_argument("--keep-engines", action="store_true")

    return p.parse_args()


def load_whitelist(path: str) -> Dict[str, Tuple[str, float]]:
    wl: Dict[str, Tuple[str, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                raise ValueError(f"Invalid whitelist line: {line}")
            name, method, ratio_text = parts[:3]
            method = method.lower()
            if method not in ("svd1", "svd2", "tucker", "orig"):
                raise ValueError(f"Invalid method in whitelist: {method}")
            ratio_val = float(ratio_text) if method != "orig" else 1.0
            wl[name] = (method, ratio_val)
            if not name.startswith("backbone."):
                wl[f"backbone.{name}"] = (method, ratio_val)
    return wl


def parse_input_shape(shape_str: str) -> Tuple[int, int, int]:
    parts = [int(x.strip()) for x in shape_str.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected C,H,W, got: {shape_str}")
    return (parts[0], parts[1], parts[2])


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)

    input_shape = parse_input_shape(args.input_shape)

    # import LRD module
    import real_lrd_resnet as lrd
    # load model
    model_args, model = load_model(args.model_dir, args.epoch, args.device)
    model.eval()

    # collect conv module names from backbone for bucketing
    conv_names = get_conv_module_names_from_backbone(model)
    LOGGER.info("Collected %d backbone Conv2d module names for bucketing.", len(conv_names))

    # apply LRD (backbone convs)
    whitelist_map = load_whitelist(args.whitelist_file) if args.whitelist_file else None
    lrd.apply_lrd_to_model(
        model,
        args.ratio,
        args.ratio_mode,
        args.device,
        args.scheme,
        skip_1x1=args.skip_1x1,
        skip_if_no_mac_reduction=args.skip_if_no_mac_reduction,
        skip_if_rank_gt_out=args.skip_if_rank_gt_out,
        whitelist_map=whitelist_map,
    )

    # export ONNX
    onnx_prefix = f"{args.onnx_prefix}_lrd" if args.ratio < 1.0 else args.onnx_prefix
    onnx_raw = os.path.join(args.output_dir, f"{onnx_prefix}_fp32_raw.onnx")
    onnx_named = os.path.join(args.output_dir, f"{onnx_prefix}_fp32_named.onnx")
    
    inference_engine = model.construct_inference_engine(fuse_postproc=False)
    model_lrd = inference_engine.construct_torch_model().to(args.device)
    model_lrd.eval()

    # 准备输入
    frame = torch.randn((args.batch_size, *input_shape), 
                       device=args.device, dtype=torch.float32)
    is_new_frame = torch.zeros(args.batch_size, dtype=torch.bool, device=args.device)
    memory = inference_engine.init_mem(args.batch_size)
    
    # 获取输入输出名称
    input_names = [name for name, _ in inference_engine.input_specs]
    output_names = list(inference_engine.output_names)
    
    onnx_fp32_path = os.path.join(args.output_dir, f"{args.onnx_prefix}_fp32.onnx")
    
    print(f"导出FP32 ONNX到 {onnx_fp32_path} ...")
    print(f"  输入: {input_names}")
    print(f"  输出: {output_names}")
    
    inputs = (frame, is_new_frame, *memory)
    input_names = [name for name, _ in inference_engine.input_specs]
    output_names = list(inference_engine.output_names)
    
    onnx_fp32_path = os.path.join(args.output_dir, f"{args.onnx_prefix}_fp32.onnx")
    
    print(f"导出FP32 ONNX到 {onnx_fp32_path} ...")
    print(f"  输入: {input_names}")
    print(f"  输出: {output_names}")

    dummy = torch.randn(args.batch_size, *input_shape, device=args.device)
    LOGGER.info("Exporting raw ONNX: %s", onnx_raw)
    torch.onnx.export(
        model_lrd,
        inputs,
        onnx_raw,
        input_names=input_names,
        output_names=output_names,
        opset_version=args.opset,
    )

    # rename Conv nodes in ONNX to module paths
    renamed = rename_onnx_conv_nodes(onnx_raw, onnx_named)
    LOGGER.info("Renamed %d ONNX Conv nodes -> module-path-based names.", renamed)

    # int8 calib cache (random)
    calib_cache = os.path.join(args.output_dir, f"{onnx_prefix}_int8_calib.cache")
    if not os.path.exists(calib_cache):
        generate_int8_calib_cache_random(
            onnx_path=onnx_named,
            calib_cache_path=calib_cache,
            batch_size=args.batch_size,
            input_shape_chw=input_shape,
            num_batches=args.int8_calibration_batches,
            workspace_gb=args.workspace_gb,
        )

    # run trtexec fp16/int8 with profiling
    fp16_engine = os.path.join(args.output_dir, f"{onnx_prefix}_fp16.engine")
    int8_engine = os.path.join(args.output_dir, f"{onnx_prefix}_int8.engine")

    fp16_profile = os.path.join(args.output_dir, f"{onnx_prefix}_fp16_profile.json")
    int8_profile = os.path.join(args.output_dir, f"{onnx_prefix}_int8_profile.json")

    fp16_lat, _ = run_trtexec(
        onnx_path=onnx_named,
        precision="fp16",
        warmup_ms=args.warmup_ms,
        duration_sec=args.duration_sec,
        iterations=args.iterations,
        use_cuda_graph=args.use_cuda_graph,
        enable_profiling=args.enable_profiling,
        profile_json_path=fp16_profile if args.enable_profiling else None,
        save_engine=fp16_engine,
        verbose=args.verbose,
    )
    int8_lat, _ = run_trtexec(
        onnx_path=onnx_named,
        precision="int8",
        calib_cache=calib_cache,
        warmup_ms=args.warmup_ms,
        duration_sec=args.duration_sec,
        iterations=args.iterations,
        use_cuda_graph=args.use_cuda_graph,
        enable_profiling=args.enable_profiling,
        profile_json_path=int8_profile if args.enable_profiling else None,
        save_engine=int8_engine,
        verbose=args.verbose,
    )

    summary = pd.DataFrame(
        [
            {"precision": "FP16", "mean_ms": fp16_lat},
            {"precision": "INT8", "mean_ms": int8_lat},
        ]
    )
    summary.to_csv(os.path.join(args.output_dir, "summary.csv"), index=False)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    if not args.enable_profiling:
        print("\nEnable --enable-profiling to generate per-layer and per-Conv2d breakdown CSVs.")
        return

    # TRT-layer tables
    df_fp16 = parse_trtexec_profile_json(fp16_profile).rename(columns={"avg_ms": "fp16_avg_ms"})
    df_int8 = parse_trtexec_profile_json(int8_profile).rename(columns={"avg_ms": "int8_avg_ms"})
    df_layers = df_fp16.merge(df_int8, on="layer_name", how="outer")
    df_layers["speedup_fp16_over_int8"] = df_layers["fp16_avg_ms"] / df_layers["int8_avg_ms"]

    # inspector precision table (INT8 engine)
    insp_json = os.path.join(args.output_dir, "int8_engine_inspector.json")
    dump_engine_inspector(int8_engine, insp_json)
    df_prec = parse_engine_inspector_precisions(insp_json)

    df_layers = df_layers.merge(df_prec[["layer_name", "layer_type", "precision", "output_type"]],
                                on="layer_name", how="left")
    df_layers = df_layers.sort_values("speedup_fp16_over_int8", ascending=False)
    df_layers.to_csv(os.path.join(args.output_dir, "layer_benefit_fp16_vs_int8.csv"), index=False)

    # Conv2d-bucket report (best-effort)
    df_conv = conv_bucket_report(
        df_layer=df_layers,
        conv_names=conv_names,
        fp16_col="fp16_avg_ms",
        int8_col="int8_avg_ms",
        precision_col="precision",
    )
    df_conv.to_csv(os.path.join(args.output_dir, "conv2d_benefit_fp16_vs_int8.csv"), index=False)

    print("\nWrote:")
    print(" - layer_benefit_fp16_vs_int8.csv (TRT layer level)")
    print(" - conv2d_benefit_fp16_vs_int8.csv (Conv2d buckets; best-effort)")

    # cleanup
    if not args.keep_engines:
        for p in [fp16_engine, int8_engine]:
            if os.path.exists(p):
                os.remove(p)

    print(f"\nAll outputs in: {args.output_dir}")
    print(f"Named ONNX used for TRT: {onnx_named}")


if __name__ == "__main__":
    main()

