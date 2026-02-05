"""
EVRT-DETR Complete Model Quantization Script

This script properly extracts the complete EVRT-DETR model by using the
trainer's built-in construct_torch_model() method, which creates a standalone
PyTorch module with all networks properly chained.

The model includes: backbone -> encoder -> temporal encoder -> decoder
"""

import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
from evlearn.eval.eval import load_model
import argparse

# ----------------------------
# Configuration
# ----------------------------
BATCH_SIZE = 1
INPUT_SHAPE = (20, 256, 320)  # Gen1: (10, 240, 304), Gen4: (10, 360, 640)
ONNX_FP32_PATH = "evrt_detr_complete_fp32.onnx"
ENGINE_FP32_PATH = "evrt_detr_complete_fp32.engine"
ENGINE_FP16_PATH = "evrt_detr_complete_fp16.engine"
ENGINE_INT8_PATH = "evrt_detr_complete_int8.engine"
INT8_CALIBRATION_BATCHES = 10

MODEL_PATH = 'models/gen1_lrd/gen1_r0.8_s2'
EPOCH = None
DEVICE = 'cuda'

print("="*80)
print("EVRT-DETR Complete Model Quantization")
print("="*80)

# ----------------------------
# Load Model
# ----------------------------
print(f"\n1. Loading model from: {MODEL_PATH}")
args, trainer = load_model(MODEL_PATH, epoch=EPOCH, device=DEVICE)

print(f"   Trainer type: {type(trainer).__name__}")
print(f"   Trainer module: {type(trainer).__module__}")

# ----------------------------
# Extract Complete Model using construct_torch_model()
# ----------------------------
print(f"\n2. Extracting complete model using construct_torch_model()...")

# Construct the complete torch model
inference_engine = trainer.construct_inference_engine()
model = inference_engine.construct_torch_model()

print(f"   ✓ Model constructed successfully!")
print(f"   Model type: {type(model).__name__}")
print(f"   Model module: {type(model).__module__}")

# Show model components
if hasattr(model, 'backbone'):
    print(f"\n   Model components:")
    components = ['backbone', 'encoder', 'tempenc', 'decoder']
    total_params = 0
    for comp_name in components:
        if hasattr(model, comp_name):
            comp = getattr(model, comp_name)
            n_params = sum(p.numel() for p in comp.parameters())
            total_params += n_params
            print(f"     - {comp_name}: {type(comp).__name__} ({n_params:,} parameters)")
    print(f"   Total parameters: {total_params:,}")

# Set model to eval mode
model.eval()

# ----------------------------
# Prepare Wrapper for ONNX Export
# ----------------------------
print(f"\n3. Creating ONNX-compatible wrapper...")

class EVRTDETRExportWrapper(torch.nn.Module):
    """
    Wrapper for EVRT-DETR that handles the temporal memory state.
    
    The original model expects: (frame, is_new_frame, *memory)
    For ONNX export, we simplify to just take the frame and initialize
    memory internally, since ONNX doesn't handle stateful models well.
    """
    
    def __init__(self, model, memory_spec, frame_shape, device='cuda'):
        super().__init__()
        self.model = model
        self.memory_spec = memory_spec
        self.frame_shape = frame_shape
        self.device = device
        
        # Pre-allocate initial memory state
        self._init_memory_state()
    
    def _init_memory_state(self):
        """Initialize the memory state tensors."""
        self.initial_memory = []
        
        # Create zero-initialized memory based on memory_spec
        for spec in self.memory_spec:
            # spec is typically a tuple like (shape, dtype)
            if isinstance(spec, (list, tuple)):
                shape = spec[0] if len(spec) > 0 else (1, 128, 1, 1)
            else:
                # Fallback if spec format is different
                shape = (1, 128, 1, 1)  # Common shape for temporal memory
            
            memory_tensor = torch.zeros(shape, device=self.device)
            self.initial_memory.append(memory_tensor)
    
    def forward(self, frame):
        """
        Simplified forward pass for ONNX export.
        
        Args:
            frame: Input tensor [batch, time_bins, height, width]
        
        Returns:
            Tuple of (logits, boxes) - the main detection outputs
        """
        batch_size = frame.shape[0]
        
        # Create is_new_frame flag (True for first frame)
        is_new_frame = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        
        # Use initial memory state
        memory = [m.expand(batch_size, -1, -1, -1) if m.ndim == 4 else m 
                  for m in self.initial_memory]
        
        # Run the model
        # Output: (logits, boxes, *updated_memory)
        outputs = self.model(frame, is_new_frame, *memory)
        
        # Extract just logits and boxes (ignore memory updates for export)
        logits = outputs[0]
        boxes = outputs[1]
        
        return logits, boxes


# Get memory spec from trainer
if hasattr(trainer, '_memory_spec'):
    memory_spec = [x[0] for x in trainer._memory_spec]
    print(f"   Memory spec found: {len(memory_spec)} memory tensors")
elif hasattr(model, '_memory_spec'):
    memory_spec = model._memory_spec
    print(f"   Memory spec found: {len(memory_spec)} memory tensors")
else:
    print("   Warning: No memory spec found, using empty list")
    memory_spec = []

# Get frame shape
if hasattr(trainer, '_frame_shape'):
    frame_shape = trainer._frame_shape
    print(f"   Frame shape: {frame_shape}")
else:
    frame_shape = INPUT_SHAPE
    print(f"   Using default frame shape: {frame_shape}")

# Create the wrapper
export_model = EVRTDETRExportWrapper(
    model, 
    memory_spec, 
    frame_shape,
    device=DEVICE
)
export_model.eval()

print(f"   ✓ Export wrapper created")

# ----------------------------
# Test Forward Pass
# ----------------------------
print(f"\n4. Testing forward pass...")
dummy_input = torch.randn(BATCH_SIZE, *INPUT_SHAPE).to(DEVICE)

with torch.no_grad():
    try:
        output = export_model(dummy_input)
        print(f"   ✓ Forward pass successful!")
        print(f"   Input shape: {list(dummy_input.shape)}")
        
        if isinstance(output, (tuple, list)):
            print(f"   Output is {type(output).__name__} with {len(output)} elements:")
            for i, elem in enumerate(output):
                if torch.is_tensor(elem):
                    print(f"     [{i}] {['logits', 'boxes'][i] if i < 2 else 'output'}: shape {list(elem.shape)}, dtype {elem.dtype}")
                else:
                    print(f"     [{i}]: {type(elem)}")
        else:
            print(f"   Output type: {type(output)}")
            
    except Exception as e:
        print(f"   ✗ Forward pass failed!")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        raise

# ----------------------------
# ONNX Export
# ----------------------------
print(f"\n5. Exporting to ONNX: {ONNX_FP32_PATH}")

# Output names
output_names = ["pred_logits", "pred_boxes"]

print(f"   Input names: ['input']")
print(f"   Output names: {output_names}")

try:
    torch.onnx.export(
        export_model,
        dummy_input,
        ONNX_FP32_PATH,
        input_names=["input"],
        output_names=output_names,
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
        # Enable dynamic batch size if needed
        # dynamic_axes={
        #     'input': {0: 'batch_size'},
        #     'pred_logits': {0: 'batch_size'},
        #     'pred_boxes': {0: 'batch_size'}
        # }
    )
    print(f"   ✓ ONNX export successful!")
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(ONNX_FP32_PATH)
        onnx.checker.check_model(onnx_model)
        print(f"   ✓ ONNX model validated")
    except ImportError:
        print(f"   (onnx package not available for validation)")
    
except Exception as e:
    print(f"   ✗ ONNX export failed!")
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()
    raise

# ----------------------------
# INT8 Calibrator
# ----------------------------
class EventDataInt8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, batches, batch_size, input_shape):
        super().__init__()
        self.batches = batches
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.current = 0
        self.device_input = cuda.mem_alloc(
            int(batch_size * np.prod(input_shape) * np.float32().nbytes)
        )
        
        # Generate calibration data
        print(f"   Generating {batches} calibration batches...")
        self.calibration_data = []
        for _ in range(batches):
            data = np.random.randn(batch_size, *input_shape).astype(np.float32)
            self.calibration_data.append(data)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current >= self.batches:
            return None
        data = self.calibration_data[self.current]
        cuda.memcpy_htod(self.device_input, data)
        self.current += 1
        return [int(self.device_input)]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        cache_file = "evrt_detr_int8.cache"
        with open(cache_file, "wb") as f:
            f.write(cache)
        print(f"   ✓ Calibration cache saved: {cache_file}")

# ----------------------------
# Build TensorRT Engines
# ----------------------------
def build_trt_engine(onnx_path, engine_path, precision="fp32"):
    """Build TensorRT engine from ONNX model."""
    
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    print(f"   Parsing ONNX model...")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("   Parsing errors:")
            for i in range(parser.num_errors):
                print(f"     - {parser.get_error(i)}")
            raise RuntimeError("ONNX parsing failed")
    
    print(f"   ✓ ONNX parsed successfully")

    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)  # 8 GB

    if precision.lower() == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print(f"   Using FP16 precision")
        else:
            print(f"   Warning: FP16 not well-supported, using FP32")
            
    elif precision.lower() == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            calibrator = EventDataInt8Calibrator(
                INT8_CALIBRATION_BATCHES, BATCH_SIZE, INPUT_SHAPE
            )
            config.int8_calibrator = calibrator
            print(f"   Using INT8 precision with calibration")
        else:
            print(f"   Warning: INT8 not well-supported, using FP32")

    # Build engine
    print(f"   Building {precision.upper()} engine (this may take 5-15 minutes)...")
    start = time.time()
    
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        raise RuntimeError(f"Failed to build {precision.upper()} engine")
    
    build_time = time.time() - start
    print(f"   ✓ Build completed in {build_time:.1f}s")
    
    # Save engine
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"   ✓ Engine saved: {engine_path}")
    
    # Return runtime engine
    runtime = trt.Runtime(logger)
    return runtime.deserialize_cuda_engine(serialized_engine)

# ----------------------------
# Benchmark
# ----------------------------
def benchmark_trt_engine(engine, batch_size=1, input_shape=(10, 240, 304), n_runs=100):
    """Benchmark TensorRT engine inference latency."""
    
    context = engine.create_execution_context()
    
    # Get input/output tensor names
    input_names = []
    output_names = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_names.append(name)
        else:
            output_names.append(name)
    
    print(f"   Inputs: {input_names}")
    print(f"   Outputs: {output_names}")
    
    # Set input shape
    input_name = input_names[0]
    input_shape_full = (batch_size, *input_shape)
    context.set_input_shape(input_name, input_shape_full)
    
    # Allocate input buffer
    input_dtype = trt.nptype(engine.get_tensor_dtype(input_name))
    input_nbytes = np.prod(input_shape_full) * np.dtype(input_dtype).itemsize
    d_input = cuda.mem_alloc(int(input_nbytes))
    
    # Allocate output buffers
    d_outputs = {}
    for name in output_names:
        out_shape = context.get_tensor_shape(name)
        out_dtype = trt.nptype(engine.get_tensor_dtype(name))
        out_nbytes = np.prod(out_shape) * np.dtype(out_dtype).itemsize
        d_outputs[name] = cuda.mem_alloc(int(out_nbytes))
        print(f"   Output '{name}': shape {out_shape}")
    
    # Bind tensors
    context.set_tensor_address(input_name, int(d_input))
    for name, ptr in d_outputs.items():
        context.set_tensor_address(name, int(ptr))
    
    # Create test input
    h_input = np.random.randn(*input_shape_full).astype(input_dtype)
    stream = cuda.Stream()
    
    # Warmup
    print(f"   Warming up...")
    for _ in range(10):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()
    
    # Benchmark
    print(f"   Running {n_runs} iterations...")
    start = time.time()
    for _ in range(n_runs):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()
    end = time.time()
    
    latency_ms = (end - start) / n_runs * 1000.0
    return latency_ms

# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quantize complete EVRT-DETR model')
    parser.add_argument('--skip-fp32', action='store_true', help='Skip FP32 engine')
    parser.add_argument('--skip-fp16', action='store_true', help='Skip FP16 engine')
    parser.add_argument('--skip-int8', action='store_true', help='Skip INT8 engine')
    parser.add_argument('--benchmark-runs', type=int, default=100, help='Benchmark iterations')
    cmd_args = parser.parse_args()
    
    # Define engines to build
    engines = {}
    if not cmd_args.skip_fp32:
        engines["FP32"] = (ONNX_FP32_PATH, ENGINE_FP32_PATH, "fp32")
    if not cmd_args.skip_fp16:
        engines["FP16"] = (ONNX_FP32_PATH, ENGINE_FP16_PATH, "fp16")
    if not cmd_args.skip_int8:
        engines["INT8"] = (ONNX_FP32_PATH, ENGINE_INT8_PATH, "int8")
    
    # Build and benchmark
    results = {}
    
    for name, (onnx_path, engine_path, precision) in engines.items():
        print(f"\n{'='*80}")
        print(f"{name} Engine")
        print(f"{'='*80}")
        
        try:
            engine = build_trt_engine(onnx_path, engine_path, precision)
            latency = benchmark_trt_engine(
                engine, BATCH_SIZE, INPUT_SHAPE, cmd_args.benchmark_runs
            )
            results[name] = latency
            print(f"   ✓ Average latency: {latency:.2f} ms")
            
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = None
    
    # Summary
    print(f"\n{'='*80}")
    print("FINAL PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"Model: Complete EVRT-DETR (backbone + encoder + tempenc + decoder)")
    print(f"Input: batch_size={BATCH_SIZE}, shape={INPUT_SHAPE}")
    print(f"Total parameters: {total_params:,}\n")
    
    baseline_latency = results.get("FP32")
    
    for name in ["FP32", "FP16", "INT8"]:
        if name in results and results[name] is not None:
            lat = results[name]
            fps = 1000.0 / lat
            speedup = baseline_latency / lat if baseline_latency else 1.0
            print(f"{name:>5}: {lat:7.2f} ms/frame | {fps:6.2f} FPS | {speedup:4.2f}x speedup")
        elif name in results:
            print(f"{name:>5}: FAILED")
    
    print(f"{'='*80}\n")
