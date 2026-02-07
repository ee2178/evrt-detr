import os
import argparse
import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

from evlearn.eval.eval import load_model

# ----------------------------
# Config
# ----------------------------
BATCH_SIZE = 1
INT8_CALIBRATION_BATCHES = 10
WORKSPACE_GB = 4

ENGINE_DIR = "trt_engines_int8"
os.makedirs(ENGINE_DIR, exist_ok=True)

# ----------------------------
# INT8 calibrator
# ----------------------------
class RandomEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, input_shape, batches):
        super().__init__()
        self.input_shape = input_shape
        self.batches = batches
        self.current = 0

        nbytes = int(np.prod(input_shape) * np.float32().nbytes)
        self.device_input = cuda.mem_alloc(nbytes)

    def get_batch_size(self):
        return self.input_shape[0]

    def get_batch(self, names):
        if self.current >= self.batches:
            return None

        data = np.random.randn(*self.input_shape).astype(np.float32)
        cuda.memcpy_htod(self.device_input, data)
        self.current += 1
        return [int(self.device_input)]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        pass

class MultiInputEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, input_shapes, num_batches):
        super().__init__()
        self.input_shapes = input_shapes
        self.num_batches = num_batches
        self.current_batch = 0

        # Allocate device buffers per input
        self.device_buffers = []
        for shape in input_shapes:
            nbytes = int(np.prod(shape) * np.float32().nbytes)
            self.device_buffers.append(cuda.mem_alloc(nbytes))

    def get_batch_size(self):
        return self.input_shapes[0][0]

    def get_batch(self, names):
        if self.current_batch >= self.num_batches:
            return None

        assert len(names) == len(self.device_buffers), (
            f"TRT expects {len(names)} inputs, but calibrator has "
            f"{len(self.device_buffers)} buffers"
        )

        # Fill each input
        for buf, shape in zip(self.device_buffers, self.input_shapes):
            data = np.random.randn(*shape).astype(np.float32)
            cuda.memcpy_htod(buf, data)

        self.current_batch += 1

        # CRITICAL: return pointers in EXACT order of `names`
        return [int(buf) for buf in self.device_buffers]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        pass

# ----------------------------
# TensorRT INT8 builder
# ----------------------------

def build_int8_engine(
    onnx_path,
    engine_path,
    default_batch=1,
    default_hw=(640, 640),
    workspace_gb=4,
    calib_batches=10,
):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed")

    print(f"[TRT] Network has {network.num_inputs} inputs")

    # --- Infer concrete input shapes ---
    input_shapes = []
    for i in range(network.num_inputs):
        t = network.get_input(i)
        shape = list(t.shape)

        # Replace dynamic dims
        for d, v in enumerate(shape):
            if v == -1:
                if d == 0:
                    shape[d] = default_batch
                elif d in (2, 3):
                    shape[d] = default_hw[d - 2]
                else:
                    raise RuntimeError(
                        f"Cannot infer dynamic dim {d} for input {t.name}"
                    )

        input_shapes.append(tuple(shape))
        print(f"[TRT] Input {i}: {t.name} -> {shape}")

    # --- Builder config ---
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_gb << 30
    )

    # --- Calibrator ---
    calibrator = MultiInputEntropyCalibrator(
        input_shapes=input_shapes,
        num_batches=calib_batches,
    )
    config.int8_calibrator = calibrator

    # --- Build ---
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("INT8 engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"[TRT] INT8 engine written to {engine_path}")


# ----------------------------
# ONNX export helper
# ----------------------------
def export_onnx(module, dummy_inputs, onnx_path):
    module.eval()
    with torch.no_grad():
        torch.onnx.export(
            module,
            dummy_inputs,
            onnx_path,
            opset_version=17,
            do_constant_folding=True,
            input_names=[f"input_{i}" for i in range(len(dummy_inputs))],
            output_names=["output"],
        )
    print(f"[ONNX] Exported {onnx_path}")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str)
    args = parser.parse_args()

    args_cfg, model = load_model(args.model_dir, epoch=None, device="cuda")
    nets = model._nets
    # ============================================================
    # Backbone
    # ============================================================
    backbone = nets.backbone._net.cuda().eval()
    backbone_in = torch.randn(BATCH_SIZE, 20, 256, 320, device="cuda")

    backbone_onnx = f"{ENGINE_DIR}/backbone.onnx"
    backbone_engine = f"{ENGINE_DIR}/backbone_int8.engine"
 
    export_onnx(backbone, (backbone_in,), backbone_onnx)
    build_int8_engine(
        backbone_onnx,
        backbone_engine,
        (BATCH_SIZE, 20, 256, 320),
    )


    # ============================================================
    # Encoder (wrapped: consumes backbone features)
    # ============================================================
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        def forward(self, f0, f1, f2):
            return self.enc([f0, f1, f2])

    encoder = EncoderWrapper(nets.encoder).cuda().eval()

    # NOTE: shape must match *actual* backbone output
    
    enc_in0 = torch.randn(1, 512, 32, 40, device="cuda")
    enc_in1 = torch.randn(1, 1024, 16, 20, device="cuda")
    enc_in2 = torch.randn(1, 2048, 8, 10, device="cuda")

    encoder_onnx = f"{ENGINE_DIR}/encoder.onnx"
    encoder_engine = f"{ENGINE_DIR}/encoder_int8.engine"

    export_onnx(encoder, (enc_in0, enc_in1, enc_in2, ), encoder_onnx)
    build_int8_engine(
        encoder_onnx,
        encoder_engine,
        (BATCH_SIZE, 256, 32, 40),
    )
    #with torch.no_grad():
    #    enc_in0 = torch.randn(1, 512, 32, 40, device="cuda")
    #    enc_in1 = torch.randn(1, 1024, 16, 20, device="cuda")
    #    enc_in2 = torch.randn(1, 2048, 8, 10, device="cuda")
    #    feats = nets.encoder([enc_in0, enc_in1, enc_in2])
    #    breakpoint()

    # ============================================================
    # Decoder
    # ============================================================
    class TRTDecoderWrapper(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, feat0, feat1, feat2):
            # Reconstruct expected list
            feats = [feat0, feat1, feat2]
            return self.decoder(feats)

    decoder_wrapper = TRTDecoderWrapper(nets.decoder).cuda().eval()

    dec_in0 = torch.randn(1, 256, 32, 40, device = 'cuda')
    dec_in1 = torch.randn(1, 256, 16, 20, device = 'cuda')
    dec_in2 = torch.randn(1, 256, 8, 10, device = 'cuda')

    decoder_onnx = f"{ENGINE_DIR}/decoder.onnx"
    decoder_engine = f"{ENGINE_DIR}/decoder_int8.engine"

    export_onnx(decoder_wrapper, (dec_in0, dec_in1, dec_in2,), decoder_onnx)
    build_int8_engine(
        decoder_onnx,
        decoder_engine,
        (BATCH_SIZE, 256, 32, 40),
    )

    print("\n✅ INT8 TensorRT engines built")
    print("ℹ️ Temporal encoder intentionally left in PyTorch (FP16)")


if __name__ == "__main__":
    main()

