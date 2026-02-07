#!/usr/bin/env python3
"""
Latency eval script for EVRT-DETR that follows the repo's eval workflow:

- Uses evlearn.eval.eval.load_model + load_eval_dset
- Calls eval_epoch(...) (patched Option A) with return_times=True
- Optionally hot-swaps TensorRT INT8 engines into model._nets before timing
- By default: latency-only (skip_metrics=True), but can be toggled

Assumptions:
- You have patched evlearn.train.train.eval_epoch to accept:
    return_times, time_set_inputs, skip_metrics
  and to return (metrics, times_dict) when return_times=True.

- Your TRT engines exist under --engine-dir:
    backbone_int8.engine, encoder_int8.engine, decoder_int8.engine

- Backbone outputs list[3] multi-scale feature maps; encoder outputs list[3] too.
"""

import argparse
import os
from pathlib import Path
from typing import Any, List, Tuple, Union

import numpy as np
import torch

# repo utilities
from evlearn.train.train import eval_epoch
from evlearn.eval.eval import load_model, load_eval_dset

# Optional: only needed if you enable --hotswap-int8
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401


# ----------------------------
# CLI
# ----------------------------
def parse_cmdargs():
    p = argparse.ArgumentParser(description="Evaluate latency (optionally with TRT INT8 hot-swap)")

    p.add_argument("model", metavar="MODEL", type=str, help="model directory")
    p.add_argument("-e", "--epoch", default=None, dest="epoch", type=int, help="epoch")

    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda", dest="device", type=str)

    p.add_argument("--data-name", default=None, dest="data_name", type=str,
                   help="name of the dataset to use")
    p.add_argument("--data-path", default=None, dest="data_path", type=str,
                   help="path to the new dataset to evaluate")
    p.add_argument("--split", default="test", dest="split", type=str, help="dataset split")

    p.add_argument("--steps", default=None, dest="steps", type=int,
                   help="steps for evaluation (timed steps)")
    p.add_argument("--batch-size", default=None, dest="batch_size", type=int,
                   help="batch size for evaluation")
    p.add_argument("--workers", default=None, dest="workers", type=int,
                   help="number of workers to use for evaluation")

    # latency behavior
    p.add_argument("--warmup-drop", default=10, type=int,
                   help="drop first N timed steps from summary stats")
    p.add_argument("--time-set-inputs", action="store_true",
                   help="also time model.set_inputs(batch) per step")
    p.add_argument("--compute-metrics", action="store_true",
                   help="also compute metrics (slower); default is latency-only")

    # TRT hot-swap
    p.add_argument("--hotswap-int8", action="store_true",
                   help="hot-swap TRT INT8 engines into model._nets.* before timing")
    p.add_argument("--engine-dir", default="trt_engines_int8", type=str,
                   help="directory containing *_int8.engine files")

    return p.parse_args()


# ----------------------------
# Output directory (unchanged)
# ----------------------------
def make_eval_directory(model, savedir, mkdir=True):
    result = os.path.join(savedir, "evals")

    if model.current_epoch is None:
        result = os.path.join(result, "final")
    else:
        result = os.path.join(result, f"epoch_{model.current_epoch}")

    if mkdir:
        os.makedirs(result, exist_ok=True)

    return result


# ----------------------------
# Latency helpers
# ----------------------------
def summarize_times_ms(times_ms: List[float], warmup_drop: int = 0):
    arr = np.asarray(times_ms, dtype=np.float64)
    if warmup_drop and arr.size > warmup_drop:
        arr = arr[warmup_drop:]
    if arr.size == 0:
        return {"n": 0}

    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


# ----------------------------
# Batch -> tensor helper (only for decoder probing in hot-swap)
# ----------------------------
def extract_first_tensor(batch: Any, device="cuda") -> torch.Tensor:
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        for k in ["input", "events", "x", "frame", "data"]:
            if k in batch and torch.is_tensor(batch[k]):
                return batch[k].to(device, non_blocking=True)
        for v in batch.values():
            if torch.is_tensor(v):
                return v.to(device, non_blocking=True)
    if isinstance(batch, (list, tuple)):
        for v in batch:
            if torch.is_tensor(v):
                return v.to(device, non_blocking=True)
    raise RuntimeError("Could not extract a tensor input from batch")


# ----------------------------
# TRT hot-swap machinery (int8)
# ----------------------------
def _trt_dtype_to_torch(dt: trt.DataType) -> torch.dtype:
    np_dt = trt.nptype(dt)
    if np_dt == torch.float32.numpy().dtype:
        return torch.float32
    if np_dt == torch.float16.numpy().dtype:
        return torch.float16
    if np_dt == torch.int8.numpy().dtype:
        return torch.int8
    if np_dt == torch.int32.numpy().dtype:
        return torch.int32
    if np_dt == torch.bool.numpy().dtype:
        return torch.bool
    raise TypeError(f"Unsupported TRT dtype: {dt} / {np_dt}")


class TRTModule(torch.nn.Module):
    """TRT executor: CUDA tensors in/out, execute_async_v3."""
    def __init__(self, engine_path: Union[str, Path], log_level=trt.Logger.ERROR):
        super().__init__()
        engine_path = str(engine_path)
        logger = trt.Logger(log_level)
        runtime = trt.Runtime(logger)

        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TRT execution context")

        self.stream = cuda.Stream()

        self.input_names: List[str] = []
        self.output_names: List[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

    def forward(self, *inputs: torch.Tensor):
        if len(inputs) != len(self.input_names):
            raise RuntimeError(f"Expected {len(self.input_names)} inputs, got {len(inputs)}")

        for name, x in zip(self.input_names, inputs):
            if not x.is_cuda:
                raise RuntimeError("TRT inputs must be CUDA tensors")
            x = x.contiguous()
            self.context.set_input_shape(name, tuple(x.shape))
            self.context.set_tensor_address(name, int(x.data_ptr()))

        outs: List[torch.Tensor] = []
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            torch_dt = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
            y = torch.empty(shape, device="cuda", dtype=torch_dt)
            self.context.set_tensor_address(name, int(y.data_ptr()))
            outs.append(y)

        ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
        if not ok:
            raise RuntimeError("TRT execute_async_v3 failed")

        self.stream.synchronize()
        return outs[0] if len(outs) == 1 else tuple(outs)


class TRTBackboneAdapter(torch.nn.Module):
    """Tensor -> list[Tensor] (expects 3 TRT outputs)."""
    def __init__(self, trt_mod: TRTModule):
        super().__init__()
        self.trt = trt_mod

    def forward(self, x):
        out = self.trt(x)
        if isinstance(out, torch.Tensor):
            raise RuntimeError("Backbone TRT must output 3 tensors, got 1.")
        out = list(out)
        if len(out) != 3:
            raise RuntimeError(f"Backbone TRT must output 3 tensors, got {len(out)}.")
        return out


class TRTEncoderAdapter(torch.nn.Module):
    """list[3] -> list[3]"""
    def __init__(self, trt_mod: TRTModule):
        super().__init__()
        self.trt = trt_mod

    def forward(self, feats):
        if not isinstance(feats, (list, tuple)) or len(feats) != 3:
            raise RuntimeError("Encoder expects list/tuple length 3.")
        out = self.trt(feats[0], feats[1], feats[2])
        if isinstance(out, torch.Tensor):
            raise RuntimeError("Encoder TRT must output 3 tensors, got 1.")
        out = list(out)
        if len(out) != 3:
            raise RuntimeError(f"Encoder TRT must output 3 tensors, got {len(out)}.")
        return out


class TRTDecoderAdapter(torch.nn.Module):
    """list[3] -> match PyTorch decoder output structure (dict/tuple/tensor)."""
    def __init__(self, trt_mod: TRTModule, pyt_decoder: torch.nn.Module, example_feats: List[torch.Tensor]):
        super().__init__()
        self.trt = trt_mod

        with torch.no_grad():
            ref = pyt_decoder(example_feats)

        self.ref_is_dict = isinstance(ref, dict)
        self.ref_keys = list(ref.keys()) if self.ref_is_dict else []
        self.ref_tuple_len = len(ref) if isinstance(ref, (tuple, list)) else (0 if self.ref_is_dict else 1)

    def forward(self, feats):
        if not isinstance(feats, (list, tuple)) or len(feats) != 3:
            raise RuntimeError("Decoder expects list/tuple length 3.")

        out = self.trt(feats[0], feats[1], feats[2])

        if self.ref_is_dict:
            if isinstance(out, torch.Tensor):
                return {self.ref_keys[0]: out}
            out = list(out)
            if len(out) != len(self.ref_keys):
                raise RuntimeError("TRT decoder outputs != PyTorch decoder dict size.")
            return {k: v for k, v in zip(self.ref_keys, out)}

        if self.ref_tuple_len == 1:
            return out if isinstance(out, torch.Tensor) else out[0]

        if isinstance(out, torch.Tensor):
            raise RuntimeError("TRT decoder returned 1 output but PyTorch returns tuple/list.")
        out = list(out)
        if len(out) != self.ref_tuple_len:
            raise RuntimeError("TRT decoder outputs != PyTorch tuple/list length.")
        return tuple(out)


def hotswap_trt_int8(model, engine_dir: Union[str, Path], example_x: torch.Tensor):
    """
    Hot-swap into model._nets.*. Leaves tempenc unchanged.
    Needs example_x only to probe decoder output structure.
    """
    engine_dir = Path(engine_dir)
    nets = model._nets

    bb_engine = engine_dir / "backbone_int8.engine"
    enc_engine = engine_dir / "encoder_int8.engine"
    dec_engine = engine_dir / "decoder_int8.engine"
    for p in [bb_engine, enc_engine, dec_engine]:
        if not p.exists():
            raise FileNotFoundError(p)

    # Probe decoder output structure using current PyTorch nets
    with torch.no_grad():
        bb = getattr(nets.backbone, "_net", nets.backbone)
        feats = bb(example_x)
        feats = nets.encoder(feats)
        if hasattr(nets, "tempenc") and nets.tempenc is not None:
            feats = nets.tempenc(feats)
        example_feats = list(feats)
        if len(example_feats) != 3:
            raise RuntimeError(f"Expected 3 decoder input features, got {len(example_feats)}.")
        pyt_decoder = nets.decoder

    # Swap
    nets.backbone = TRTBackboneAdapter(TRTModule(bb_engine)).cuda().eval()
    nets.encoder = TRTEncoderAdapter(TRTModule(enc_engine)).cuda().eval()
    nets.decoder = TRTDecoderAdapter(TRTModule(dec_engine), pyt_decoder, example_feats).cuda().eval()


# ----------------------------
# Eval single dataset (timed)
# ----------------------------
def eval_single_dataset_latency(
    model, args, data_name, data_config, split, steps, batch_size, workers, data_path,
    do_hotswap: bool, engine_dir: str, warmup_drop: int, time_set_inputs: bool,
    compute_metrics: bool,
):
    # mirror their config overrides
    if batch_size is not None:
        data_config.batch_size = batch_size
    if workers is not None:
        data_config.workers = workers
    if data_path is not None:
        data_config.dataset["path"] = data_path

    args.config.data.eval = {data_name: data_config}
    dl = load_eval_dset(args, split=split)

    # Grab one batch for TRT probing if needed
    if do_hotswap:
        first_batch = next(iter(dl))
        example_x = extract_first_tensor(first_batch, device="cuda")
        hotswap_trt_int8(model, engine_dir, example_x)
        # Re-create dataloader iterator because we consumed one element
        dl = load_eval_dset(args, split=split)

    # Run eval_epoch with timing enabled
    metrics, times = eval_epoch(
        dl, model,
        title=f"Latency: {data_name}",
        steps_per_epoch=steps,
        return_times=True,
        time_set_inputs=time_set_inputs,
        skip_metrics=(not compute_metrics),
    )

    # Summaries
    step_stats = summarize_times_ms(times["eval_step_ms"], warmup_drop=warmup_drop)
    set_stats = None
    if time_set_inputs and "set_inputs_ms" in times:
        set_stats = summarize_times_ms(times["set_inputs_ms"], warmup_drop=warmup_drop)

    return metrics, times, step_stats, set_stats


def main():
    cmdargs = parse_cmdargs()

    args, model = load_model(cmdargs.model, epoch=cmdargs.epoch, device=cmdargs.device)

    data_config_dict = args.config.data.eval
    assert isinstance(data_config_dict, dict)

    if cmdargs.data_name is not None:
        datasets = [cmdargs.data_name]
    else:
        datasets = list(sorted(data_config_dict.keys()))

    with torch.inference_mode():
        for name in datasets:
            metrics, times, step_stats, set_stats = eval_single_dataset_latency(
                model, args, name, data_config_dict[name],
                cmdargs.split, cmdargs.steps, cmdargs.batch_size,
                cmdargs.workers, cmdargs.data_path,
                do_hotswap=cmdargs.hotswap_int8,
                engine_dir=cmdargs.engine_dir,
                warmup_drop=cmdargs.warmup_drop,
                time_set_inputs=cmdargs.time_set_inputs,
                compute_metrics=cmdargs.compute_metrics,
            )

            print(f"\nDataset: {name}")
            if cmdargs.hotswap_int8:
                print(f"  Mode: TRT INT8 hot-swapped  (engines: {cmdargs.engine_dir})")
            else:
                print("  Mode: Baseline (no hot-swap)")

            if step_stats.get("n", 0) == 0:
                print("  No timing samples collected.")
            else:
                print("  eval_step latency (ms/batch):")
                print(f"    n   : {step_stats['n']}")
                print(f"    mean: {step_stats['mean']:.3f}")
                print(f"    p50 : {step_stats['p50']:.3f}")
                print(f"    p90 : {step_stats['p90']:.3f}")
                print(f"    p99 : {step_stats['p99']:.3f}")
                print(f"    min : {step_stats['min']:.3f}")
                print(f"    max : {step_stats['max']:.3f}")

            if set_stats is not None and set_stats.get("n", 0) > 0:
                print("  set_inputs latency (ms/batch):")
                print(f"    mean: {set_stats['mean']:.3f}  p50: {set_stats['p50']:.3f}  p90: {set_stats['p90']:.3f}")

            # If you enabled metrics, you can print them too
            if cmdargs.compute_metrics and metrics is not None:
                try:
                    print("  Metrics:", metrics.get())
                except Exception:
                    print("  Metrics collected (unable to print metrics.get())")


if __name__ == "__main__":
    main()

