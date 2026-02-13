#!/usr/bin/env python3
"""
EvRT-DETR TensorRT PTQ (INT8) + Metrics Evaluation (Route A, NO trtexec)

End-to-end:
1) Load EvRT-DETR model via evlearn.eval.eval.load_model
2) (Optional) Apply your LRD transform (best-effort if real_lrd_resnet exists)
3) Construct EvRT inference engine + underlying TorchModel (same as bench_model_video.py)
4) Export TorchModel to ONNX (multi-input: frame, is_new_frame, *memory)
5) Build TensorRT engine via TensorRT Python Builder (NO trtexec)
   - INT8 calibration uses a REAL dataloader (not random data)
   - Optionally rolls memory forward with the PyTorch model during calibration
   - Saves calibration cache for reuse
6) Load TRT engine and run the SAME evaluator metrics path as bench_model_video.py
7) Save metrics + timing CSVs

Notes:
- For TRT stability, fuse_postproc=False is usually safer (postproc stays in evaluator path).
- If you build and load the engine using the same Python environment (same TensorRT version),
  you avoid the "Version tag does not match" serialization failure.


For no distillation

PYTHONPATH=$PWD python scripts/quant_trt/bench_lrd_quant_video_v2.py \
  models/gen1_backup/video_evrtdetr_presnet50 \
  --device cuda \
  --data-name video \
  --split test \
  --calib-split test \
  --batch-size 1 \
  --precision int8 \
  --calib-steps 128 \
  --calib-memory-rollout \
  --apply-lrd \
  --ratio 0.5 \
  --ratio-mode param \
  --scheme 2 \
  --skip_1x1 \
  --skip_if_no_mac_reduction

For with distillation 
PYTHONPATH=$PWD python scripts/quant_trt/bench_lrd_quant_video_v2.py \
  models/gen1_backup/video_evrtdetr_presnet50 \
  --device cuda \
  --data-name video \
  --split test \
  --calib-split test \
  --batch-size 1 \
  --precision int8 \
  --calib-steps 128 \
  --calib-memory-rollout \
  --apply-lrd \
  --ratio 0.5 \
  --ratio-mode param \
  --scheme 2 \
  --skip-1x1 \
  --skip-if-no-mac-reduction \
  --backbone-ckpt models/gen1_backup/video_evrtdetr_presnet50/distilled/backbone_rr50_distilled.pt


"""

import argparse
import os
from itertools import islice
from typing import Optional

import torch
import tqdm
import pandas as pd

from evlearn.bundled.leanbase.base.metrics import Metrics
from evlearn.train.train import infer_steps
from evlearn.eval.eval import load_model, load_eval_dset


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="EvRT-DETR TensorRT PTQ (INT8) + Metrics Evaluation (no trtexec)"
    )

    # Model + data
    p.add_argument("model_dir", type=str, help="Model directory (config + checkpoints)")
    p.add_argument("-e", "--epoch", type=int, default=None, help="Epoch to load")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument(
        "--data-name",
        choices=["clip", "frame", "video"],
        default="video",
        help="Which datastream to use (video recommended)",
    )
    p.add_argument("--data-path", type=str, default=None, help="Override dataset path")
    p.add_argument("--split", type=str, default="test", help="Eval split")
    p.add_argument("--calib-split", type=str, default="train", help="Calibration split")
    p.add_argument("--steps", type=int, default=None, help="Eval temporal steps to run")
    p.add_argument(
        "--calib-steps",
        type=int,
        default=128,
        help="Calibration inner steps (frames/clips) to consume",
    )
    p.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    p.add_argument("--workers", type=int, default=None, help="DataLoader workers override")

    # Dtype + precision
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument(
        "--precision",
        choices=["fp16", "int8"],
        default="int8",
        help="TensorRT precision for engine build/eval",
    )
    p.add_argument(
        "--fuse-postproc",
        action="store_true",
        help="Fuse postprocessor into TorchModel forward (may reduce TRT stability).",
    )

    # TRT build knobs
    p.add_argument("--workspace-gb", type=int, default=4, help="TRT workspace GB")
    p.add_argument(
        "--calib-memory-rollout",
        action="store_true",
        help="During calibration, update memory by running torch_model forward (recommended).",
    )

    # Output
    p.add_argument("--outdir", type=str, default=None, help="Output directory (default: model_dir)")
    p.add_argument("--prefix", type=str, default="model", help="Filename prefix")
    p.add_argument("--keep-artifacts", action="store_true", help="Keep ONNX/engine/cache files")

    # Optional LRD flags (best-effort; script still runs if module not found)
    p.add_argument("--apply-lrd", action="store_true", help="Apply LRD before export/build")
    p.add_argument("--ratio", type=float, default=0.5)
    p.add_argument("--ratio-mode", type=str, default="param", choices=["param", "rank"])
    p.add_argument("--scheme", type=int, default=1)
    p.add_argument("--whitelist-file", type=str, default=None)
    p.add_argument("--skip-1x1", action="store_true")
    p.add_argument("--skip-if-no-mac-reduction", action="store_true")
    p.add_argument("--skip-if-rank-gt-out", action="store_true")

    # Some new flags for distilled checkpoints
    p.add_argument("--backbone-ckpt", type=str, default=None,
                    help="Path to distilled backbone checkpoint (contains backbone_state_dict).")

    return p.parse_args()


# ----------------------------
# TensorRT runner (engine -> torch tensors)
# ----------------------------
class TrtEngineRunner:
    """
    Runs a TensorRT *.engine built from your ONNX-exported TorchModel.

    Uses the newer IO-tensors API (execute_async_v3) if available; falls back to legacy bindings API.
    """

    def __init__(self, engine_path: str):
        import tensorrt as trt

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.ERROR)

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        self.stream = torch.cuda.current_stream().cuda_stream

        self._use_v3 = hasattr(self.context, "execute_async_v3") and hasattr(self.engine, "num_io_tensors")

        if self._use_v3:
            self.io_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            self.input_names = [
                n for n in self.io_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
            ]
            self.output_names = [
                n for n in self.io_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT
            ]
        else:
            self.binding_names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)]
            self.input_names = [n for n in self.binding_names if self.engine.binding_is_input(n)]
            self.output_names = [n for n in self.binding_names if not self.engine.binding_is_input(n)]

    def _trt_dtype_to_torch(self, dt):
        trt = self.trt
        if dt == trt.DataType.FLOAT:
            return torch.float32
        if dt == trt.DataType.HALF:
            return torch.float16
        if dt == trt.DataType.INT8:
            return torch.int8
        if dt == trt.DataType.INT32:
            return torch.int32
        if dt == trt.DataType.BOOL:
            return torch.bool
        raise TypeError(f"Unsupported TRT dtype: {dt}")

    def __call__(self, *inputs: torch.Tensor):
        for t in inputs:
            if not (isinstance(t, torch.Tensor) and t.is_cuda):
                raise ValueError("All inputs must be CUDA torch tensors")
        if self._use_v3:
            return self._run_v3(*inputs)
        return self._run_legacy(*inputs)

    def _run_v3(self, *inputs: torch.Tensor):
        if len(inputs) != len(self.input_names):
            raise RuntimeError(f"Engine expects {len(self.input_names)} inputs, got {len(inputs)}")

        # inputs
        for name, t in zip(self.input_names, inputs):
            if hasattr(self.context, "set_input_shape"):
                self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, int(t.data_ptr()))

        # outputs
        outputs = []
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self._trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            out = torch.empty(shape, device="cuda", dtype=dtype)
            self.context.set_tensor_address(name, int(out.data_ptr()))
            outputs.append(out)

        ok = self.context.execute_async_v3(self.stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")

        return tuple(outputs)

    def _run_legacy(self, *inputs: torch.Tensor):
        if len(inputs) != len(self.input_names):
            raise RuntimeError(f"Engine expects {len(self.input_names)} inputs, got {len(inputs)}")

        bindings = [0] * self.engine.num_bindings

        for name, t in zip(self.input_names, inputs):
            idx = self.engine.get_binding_index(name)
            if hasattr(self.context, "set_binding_shape"):
                self.context.set_binding_shape(idx, tuple(t.shape))
            bindings[idx] = int(t.data_ptr())

        outputs = []
        for name in self.output_names:
            idx = self.engine.get_binding_index(name)
            shape = tuple(self.context.get_binding_shape(idx))
            dtype = self._trt_dtype_to_torch(self.engine.get_binding_dtype(idx))
            out = torch.empty(shape, device="cuda", dtype=dtype)
            bindings[idx] = int(out.data_ptr())
            outputs.append(out)

        ok = self.context.execute_async_v2(bindings, self.stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v2 failed")

        return tuple(outputs)


# ----------------------------
# TensorRT build (no trtexec)
# ----------------------------
def build_engine_with_trt_python(
    *,
    onnx_path: str,
    engine_path: str,
    precision: str,  # "fp16" or "int8"
    calib_cache_path: Optional[str],
    dl_calib,
    inference_engine,
    torch_model,
    data_name: str,
    batch_size: int,
    calib_steps: int,
    workspace_gb: int,
    use_memory_rollout: bool,
):
    import tensorrt as trt

    if precision not in ("fp16", "int8"):
        raise ValueError(f"Unsupported precision: {precision}")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb) << 30)

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)

    if precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # allow fallback

        # Dataloader-driven calibrator (multi-input + memory)
        # Since our activations are sparse, use a minmax calibrator
        class EvrtCalibrator(trt.IInt8MinMaxCalibrator):
            def __init__(self):
                super().__init__()
                self.cache_file = calib_cache_path
                self.temporal_it = iter(dl_calib)
                self.inner_it = None
                self.memory = inference_engine.init_mem(batch_size)
                self.step = 0
                self.input_names = [name for (name, _spec) in inference_engine.input_specs]

            def get_batch_size(self):
                return batch_size

            def read_calibration_cache(self):
                if self.cache_file and os.path.exists(self.cache_file):
                    print(f"[CALIB] Using existing cache: {self.cache_file}", flush=True)
                    with open(self.cache_file, "rb") as f:
                        return f.read()
                print("[CALIB] No cache; calibrating from data", flush=True)
                return None    
            def write_calibration_cache(self, cache):
                if self.cache_file:
                    with open(self.cache_file, "wb") as f:
                        f.write(cache)

            def _next_real_step(self):
                if self.step >= calib_steps:
                    return None

                # Ensure we have an active "inner" iterator (frames/clips inside a temporal batch)
                while self.inner_it is None:
                    temporal_batch = next(self.temporal_it, None)
                    if temporal_batch is None:
                        return None
                    inference_engine.set_inputs(temporal_batch)
                    self.inner_it = iter(inference_engine.data_it(data_name))

                item = next(self.inner_it, None)
                if item is None:
                    self.inner_it = None
                    return self._next_real_step()

                frame, is_new_frame, _labels = item

                frame = frame.contiguous()
                is_new_frame = is_new_frame.to(dtype=torch.bool).contiguous()

                mem_in = [m.contiguous() for m in self.memory]

                if use_memory_rollout:
                    with torch.inference_mode():
                        outs = torch_model(frame, is_new_frame, *mem_in)
                    # outs = (..., *mem_out)
                    # unfused: (logits, boxes, *mem_out)
                    # fused: (labels, boxes, scores, *mem_out)
                    n_base = 3 if inference_engine._fuse_postproc else 2
                    self.memory = list(outs[n_base:])

                self.step += 1
                tensors = [frame, is_new_frame, *mem_in]
                return dict(zip(self.input_names, tensors))

            def get_batch(self, names):
                print(f"[CALIB] get_batch step={self.step}", flush=True)
                sample = self._next_real_step()
                if sample is None:
                    return None
                return [int(sample[n].data_ptr()) for n in names]

        config.int8_calibrator = EvrtCalibrator()

    print(f"[TRT-PY] Building engine ({precision}) -> {engine_path}")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("Failed to build TensorRT engine (builder returned None)")

    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    print(f"[TRT-PY] Saved engine: {engine_path}")


# ----------------------------
# Evaluation loop (same semantics as bench_model_video.py)
# ----------------------------
@torch.no_grad()
def benchmark_eval_epoch(
    dl_test,
    inference_engine,
    runner,  # either torch_model or TrtEngineRunner
    steps_per_epoch,
    batch_size,
    data_name,
    time_cuda: bool = True,
):
    inference_engine.eval_epoch_start()

    steps = infer_steps(dl_test, steps_per_epoch)
    progbar = tqdm.tqdm(desc="Bench", total=steps, dynamic_ncols=True)
    metrics = Metrics(prefix="bench_")
    times = []
    memory = inference_engine.init_mem(batch_size)

    for temporal_batch in islice(dl_test, steps):
        inference_engine.set_inputs(temporal_batch)

        for batch in inference_engine.data_it(data_name):
            inputs = list(batch[:-1]) + list(memory)

            if time_cuda and torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                start.record()
                outputs = runner(*inputs)
                end.record()
                torch.cuda.synchronize()
                elapsed = start.elapsed_time(end)
            else:
                outputs = runner(*inputs)
                elapsed = float("nan")

            curr_metrics, memory = inference_engine.eval_step_standanlone(outputs, batch[-1])
            metrics.update(curr_metrics)

            times.append((len(batch[0]), elapsed))

        progbar.set_postfix(metrics.get(), refresh=False)
        progbar.update()

    metrics.update(inference_engine.eval_epoch_end())
    progbar.set_postfix(metrics.get(), refresh=True)
    progbar.close()

    return metrics, times


def make_eval_directory(model, savedir, mkdir=True):
    result = os.path.join(savedir, "evals")
    if model.current_epoch is None:
        result = os.path.join(result, "final")
    else:
        result = os.path.join(result, f"epoch_{model.current_epoch}")
    if mkdir:
        os.makedirs(result, exist_ok=True)
    return result


def save_metrics(metrics_dict, evaldir, fname_suffix):
    path = os.path.join(evaldir, f"metrics_{fname_suffix}.csv")
    pd.Series(metrics_dict).to_frame().T.to_csv(path, index=False)
    print(f"[Saved] {path}")


def save_times(times, evaldir, fname_suffix):
    path = os.path.join(evaldir, f"times_{fname_suffix}.csv")
    pd.DataFrame(times, columns=["batch_size", "time_ms"]).to_csv(path, index=False)
    print(f"[Saved] {path}")


def print_times(times, n_warmup=5):
    df = pd.DataFrame(times, columns=["batch_size", "time_ms"])
    if df.empty:
        print("No timing data.")
        return
    df_warm = df.iloc[n_warmup:] if len(df) > n_warmup else df

    def stats(d):
        total_events = int(d["batch_size"].sum())
        total_time = float(d["time_ms"].sum())
        tpe = total_time / max(total_events, 1)
        return (
            total_time,
            total_events,
            tpe,
            float(d["time_ms"].mean()),
            float(d["time_ms"].median()),
            float(d["time_ms"].std()),
        )

    print("All iterations:")
    tot, ne, tpe, meanb, medb, stdb = stats(df)
    print(f"  - total time [ms]: {tot:.3f}")
    print(f"  - n events: {ne}")
    print(f"  - avg time per event [ms]: {tpe:.6f}")
    print(f"  - avg time per batch [ms]: {meanb:.3f}")
    print(f"  - median time per batch [ms]: {medb:.3f}")
    print(f"  - stdev time per batch [ms]: {stdb:.3f}")

    print(f"Warm iterations (>{n_warmup}):")
    tot, ne, tpe, meanb, medb, stdb = stats(df_warm)
    print(f"  - total time [ms]: {tot:.3f}")
    print(f"  - n events: {ne}")
    print(f"  - avg time per event [ms]: {tpe:.6f}")
    print(f"  - avg time per batch [ms]: {meanb:.3f}")
    print(f"  - median time per batch [ms]: {medb:.3f}")
    print(f"  - stdev time per batch [ms]: {stdb:.3f}")


# ----------------------------
# Optional LRD application (best-effort)
# ----------------------------
def maybe_apply_lrd(cmdargs, model):
    if not cmdargs.apply_lrd:
        return

    whitelist_map = None
    if cmdargs.whitelist_file:
        whitelist_map = {}
        with open(cmdargs.whitelist_file, "r", encoding="utf-8") as f:
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
                whitelist_map[name] = (method, ratio_val)
                if not name.startswith("backbone."):
                    whitelist_map[f"backbone.{name}"] = (method, ratio_val)

    try:
        import real_lrd_resnet as lrd
    except Exception as e:
        print(f"[LRD] WARNING: Could not import real_lrd_resnet; skipping LRD. Error: {repr(e)}")
        return

    print("[LRD] Applying low-rank decomposition...")
    lrd.apply_lrd_to_model(
        model,
        cmdargs.ratio,
        cmdargs.ratio_mode,
        cmdargs.device,
        cmdargs.scheme,
        skip_1x1=cmdargs.skip_1x1,
        skip_if_no_mac_reduction=cmdargs.skip_if_no_mac_reduction,
        skip_if_rank_gt_out=cmdargs.skip_if_rank_gt_out,
        whitelist_map=whitelist_map,
    )
    print("[LRD] Done.")


# ----------------------------
# Main
# ----------------------------
def main():
    cmdargs = parse_args()

    if cmdargs.device != "cuda" and cmdargs.precision in ("fp16", "int8"):
        raise ValueError("TensorRT path requires --device cuda")

    if cmdargs.device != "cuda":
        raise ValueError("This script is intended for CUDA (TensorRT + CUDA timing).")

    outdir = cmdargs.outdir or cmdargs.model_dir
    os.makedirs(outdir, exist_ok=True)

    # 1) Load model wrapper
    args, model = load_model(cmdargs.model_dir, epoch=cmdargs.epoch, device=cmdargs.device)
    model.eval()

    # 2) Apply LRD only if needed to create low-rank modules
    maybe_apply_lrd(cmdargs, model)

    # 2b) If provided, load distilled backbone weights
    if cmdargs.backbone_ckpt is not None:
        ckpt = torch.load(cmdargs.backbone_ckpt, map_location="cpu")
        bb_sd = ckpt["backbone_state_dict"] if "backbone_state_dict" in ckpt else ckpt
        model._nets.backbone.load_state_dict(bb_sd, strict=False)
        print(f"[CKPT] Loaded distilled backbone from: {cmdargs.backbone_ckpt}")


    # 3) Construct inference engine + underlying torch model (this is what we export)
    inference_engine = model.construct_inference_engine(fuse_postproc=cmdargs.fuse_postproc)
    torch_model = inference_engine.construct_torch_model().to(cmdargs.device).eval()

    # 4) Configure dataset (eval + calib)
    data_config = args.config.data.eval[cmdargs.data_name]
    if cmdargs.batch_size is not None:
        data_config.batch_size = cmdargs.batch_size
    if cmdargs.workers is not None:
        data_config.workers = cmdargs.workers
    if cmdargs.data_path is not None:
        data_config.dataset["path"] = cmdargs.data_path

    args.config.data.eval = {cmdargs.data_name: data_config}
    dl_eval = load_eval_dset(args, split=cmdargs.split)
    dl_calib = load_eval_dset(args, split=cmdargs.calib_split)

    bs = data_config.batch_size
    if bs is None:
        raise RuntimeError("batch_size ended up None; set --batch-size or ensure config has it.")

    # --------------------------------------
    # Build optional LRD tag
    # --------------------------------------
    if getattr(cmdargs, "apply_lrd", False):
        rr_tag = f"_rr{int(round(cmdargs.ratio * 100))}"
    else:
        rr_tag = ""

    # --------------------------------------
    # 5) Export ONNX / TensorRT artifacts
    # --------------------------------------
    onnx_path = os.path.join(
        outdir, f"{cmdargs.prefix}{rr_tag}_fp32.onnx"
    )

    engine_path = os.path.join(
        outdir, f"{cmdargs.prefix}{rr_tag}_{cmdargs.precision}.engine"
    )

    calib_cache_path = os.path.join(
        outdir, f"{cmdargs.prefix}{rr_tag}_int8_calib.cache"
    )


    input_names = [name for (name, _spec) in inference_engine.input_specs]
    output_names = list(inference_engine.output_names)

    # Get frame shape from input_specs (avoid reaching into private fields)
    spec_map = {name: spec for (name, spec) in inference_engine.input_specs}
    if "frame" not in spec_map:
        raise RuntimeError(f"Could not find 'frame' in input_specs. Found: {list(spec_map.keys())}")
    frame_shape = spec_map["frame"].shape  # (C,H,W)
    if len(frame_shape) != 3:
        raise RuntimeError(f"Expected frame spec shape (C,H,W), got: {frame_shape}")

    c, h, w = frame_shape
    frame = torch.randn((bs, c, h, w), device=cmdargs.device, dtype=torch.float32)
    is_new_frame = torch.zeros(bs, device=cmdargs.device, dtype=torch.bool)
    memory = inference_engine.init_mem(bs)

    print(f"[ONNX] Exporting to: {onnx_path}")
    print(f"[ONNX] Inputs : {input_names}")
    print(f"[ONNX] Outputs: {output_names}")

    with torch.no_grad():
        torch.onnx.export(
            torch_model,
            (frame, is_new_frame, *memory),
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            do_constant_folding=True,
        )

    # 6) Build engine with TensorRT Python Builder (no trtexec)
    build_engine_with_trt_python(
        onnx_path=onnx_path,
        engine_path=engine_path,
        precision=cmdargs.precision,
        calib_cache_path=calib_cache_path if cmdargs.precision == "int8" else None,
        dl_calib=dl_calib,
        inference_engine=inference_engine,
        torch_model=torch_model,
        data_name=cmdargs.data_name,
        batch_size=bs,
        calib_steps=cmdargs.calib_steps,
        workspace_gb=cmdargs.workspace_gb,
        use_memory_rollout=cmdargs.calib_memory_rollout,
    )

    # 7) Metrics evaluation using TRT engine
    runner = TrtEngineRunner(engine_path)
    metrics, times = benchmark_eval_epoch(
        dl_eval,
        inference_engine,
        runner,
        cmdargs.steps,
        bs,
        cmdargs.data_name,
        time_cuda=True,
    )

    evaldir = make_eval_directory(model, cmdargs.model_dir)
    suffix = (
        f"{cmdargs.prefix}_trtpy_{cmdargs.precision}"
        f"_data({cmdargs.data_name})_split({cmdargs.split})"
        f"_bs({bs})"
        f"_calibsplit({cmdargs.calib_split})_calibsteps({cmdargs.calib_steps})"
        f"_fuse({int(cmdargs.fuse_postproc)})"
    )
    save_metrics(metrics.get(), evaldir, suffix)
    save_times(times, evaldir, suffix)
    print_times(times)

    # 8) Cleanup
    if not cmdargs.keep_artifacts:
        for pth in [engine_path, onnx_path]:
            if os.path.exists(pth):
                os.remove(pth)
                print(f"[Cleanup] Removed {pth}")
        if cmdargs.precision == "int8" and os.path.exists(calib_cache_path):
            os.remove(calib_cache_path)
            print(f"[Cleanup] Removed {calib_cache_path}")

    print("[Done]")


if __name__ == "__main__":
    main()

