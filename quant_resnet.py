import torch
import torchvision
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # initializes CUDA driver
import time
from evlearn.eval.eval   import load_model, load_eval_dset

import argparse



# ----------------------------
# User config
# ----------------------------
BATCH_SIZE = 1
INPUT_SHAPE = (20, 256, 320)
ONNX_FP32_PATH = "resnet50_fp32.onnx"
ENGINE_FP32_PATH = "resnet50_fp32.engine"
ENGINE_FP16_PATH = "resnet50_fp16.engine"
ENGINE_INT8_PATH = "resnet50_int8.engine"
INT8_CALIBRATION_BATCHES = 10

# ----------------------------
# Export FP32 ONNX
# ----------------------------
print("Loading FP32 ResNet50...")

#args, model = load_model(
#    'models/gen1_backup/video_evrtdetr_presnet50', epoch = None, device = 'cuda',
#)
args, model = load_model(
    'models/gen1_lrd/gen1_r0.8_s2', epoch = None, device = 'cuda',
)

# model_fp32 = torchvision.models.resnet50(weights="IMAGENET1K_V1").eval()

model_fp32 = model._nets.backbone._net

dummy_input = torch.randn(BATCH_SIZE, *INPUT_SHAPE).to("cuda")

print(f"Exporting FP32 ONNX to {ONNX_FP32_PATH} ...")
torch.onnx.export(
    model_fp32,
    dummy_input,
    ONNX_FP32_PATH,
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
)
print("FP32 ONNX export done.")

# ----------------------------
# INT8 calibrator
# ----------------------------
class RandomInt8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batches, batch_size, input_shape):
        super().__init__()
        self.batches = batches
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.current = 0
        self.device_input = cuda.mem_alloc(int(batch_size * np.prod(input_shape) * np.float32().nbytes))

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current >= self.batches:
            return None
        data = np.random.randn(self.batch_size, *self.input_shape).astype(np.float32)
        cuda.memcpy_htod(self.device_input, data)
        self.current += 1
        return [int(self.device_input)]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        pass

# ----------------------------
# Build TensorRT engine
# ----------------------------
def build_trt_engine(onnx_path, engine_path, precision="fp32"):
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
    config.set_flag(trt.BuilderFlag.REFIT)  # safe default
    config.set_flag(trt.BuilderFlag.STRICT_NANS)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB

    if precision.lower() == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision.lower() == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        calibrator = RandomInt8Calibrator(INT8_CALIBRATION_BATCHES, BATCH_SIZE, INPUT_SHAPE)
        config.int8_calibrator = calibrator

    print(f"Building {precision.upper()} TensorRT engine...")
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError(f"Failed to build {precision.upper()} engine")
    with open(engine_path, "wb") as f:
        f.write(engine)
    print(f"{precision.upper()} engine saved to {engine_path}")
    return trt.Runtime(logger).deserialize_cuda_engine(engine)

# ----------------------------
# Benchmark
# ----------------------------
def benchmark_trt_engine(
    engine,
    batch_size=1,
    input_shape=(20, 256, 320),
    n_runs=50,
):
    """
    Benchmarks a TensorRT ICudaEngine.
    Returns average latency in milliseconds.
    """

    # ----------------------------
    # Create execution context
    # ----------------------------
    context = engine.create_execution_context()

    # ----------------------------
    # Identify input / output tensors
    # ----------------------------
    input_names = []
    output_names = []

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)

    assert len(input_names) == 1, "Expected exactly one input tensor"

    input_name = input_names[0]

    # ----------------------------
    # Set input shape (EXPLICIT BATCH — REQUIRED)
    # ----------------------------
    input_shape_full = (batch_size, *input_shape)
    context.set_input_shape(input_name, input_shape_full)

    # ----------------------------
    # Allocate input buffer
    # ----------------------------
    input_dtype = trt.nptype(engine.get_tensor_dtype(input_name))
    input_nbytes = np.prod(input_shape_full) * np.dtype(input_dtype).itemsize
    d_input = cuda.mem_alloc(int(input_nbytes))

    # ----------------------------
    # Allocate ALL output buffers
    # ----------------------------
    d_outputs = {}

    for name in output_names:
        out_shape = context.get_tensor_shape(name)
        out_dtype = trt.nptype(engine.get_tensor_dtype(name))
        out_nbytes = np.prod(out_shape) * np.dtype(out_dtype).itemsize
        d_outputs[name] = cuda.mem_alloc(int(out_nbytes))

    # ----------------------------
    # Bind tensor addresses
    # ----------------------------
    context.set_tensor_address(input_name, int(d_input))
    for name, ptr in d_outputs.items():
        context.set_tensor_address(name, int(ptr))

    # ----------------------------
    # Prepare input
    # ----------------------------
    h_input = np.random.randn(*input_shape_full).astype(input_dtype)

    stream = cuda.Stream()

    # ----------------------------
    # Warm-up
    # ----------------------------
    for _ in range(5):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()

    # ----------------------------
    # Benchmark
    # ----------------------------
    start = time.time()
    for _ in range(n_runs):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()
    end = time.time()

    latency_ms = (end - start) / n_runs * 1000.0
    print(f"TensorRT latency: {latency_ms:.2f} ms (batch={batch_size})")

    return latency_ms

# ----------------------------
# Run all benchmarks
# ----------------------------
engines = {
    "FP32": (ONNX_FP32_PATH, ENGINE_FP32_PATH, "fp32"),
    "FP16": (ONNX_FP32_PATH, ENGINE_FP16_PATH, "fp16"),
    "INT8": (ONNX_FP32_PATH, ENGINE_INT8_PATH, "int8"),
}

results = {}
for name, (onnx_path, engine_path, precision) in engines.items():
    engine = build_trt_engine(onnx_path, engine_path, precision)
    latency = benchmark_trt_engine(engine)
    results[name] = latency
    print(f"{name}: {latency:.2f} ms per batch of {BATCH_SIZE}")

print("\nLatency summary:")
for k, v in results.items():
    print(f"{k:>4}: {v:.2f} ms per batch of {BATCH_SIZE}")

