import torch
import torchvision
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # initializes CUDA driver
import time
from evlearn.eval.eval   import load_model, load_eval_dset

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

args, model = load_model(
    'models/gen1_backup/video_evrtdetr_presnet50', epoch = None, device = 'cuda',
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
def benchmark_trt_engine(engine, batch_size=1, input_shape=(3, 224, 224), n_runs=50, fp16_block=False):
    """
    Benchmarks a TensorRT ICudaEngine for FP32, FP16, or FP16 block mode.
    Returns average latency in milliseconds.
    """

    # Identify input and output tensor names
    input_names = []
    output_names = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)

    input_name = input_names[0]
    output_name = output_names[0]

    # Allocate device memory for inputs/outputs
    input_shape_full = (batch_size, *input_shape)
    input_dtype = trt.nptype(engine.get_tensor_dtype(input_name))
    output_dtype = trt.nptype(engine.get_tensor_dtype(output_name))

    input_size = int(np.prod(input_shape_full) * np.dtype(input_dtype).itemsize)
    output_size = int(np.prod(tuple(engine.get_tensor_shape(output_name))) * np.dtype(output_dtype).itemsize)

    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)

    # Prepare dummy input
    dummy_input = np.random.randn(*input_shape_full).astype(input_dtype)

    # Create stream
    stream = cuda.Stream()

    # Create execution context
    context = engine.create_execution_context()

    # Bind device memory
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))

    # Warm-up
    for _ in range(5):
        cuda.memcpy_htod_async(d_input, dummy_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(n_runs):
        cuda.memcpy_htod_async(d_input, dummy_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()
    end = time.time()

    latency_ms = (end - start) / n_runs * 1000

    print(f"TensorRT engine latency: {latency_ms:.2f} ms per batch of {batch_size}")
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

