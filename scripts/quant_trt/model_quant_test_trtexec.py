#!/usr/bin/env python3
"""
使用trtexec测试模型量化（PTQ）的运行速度
"""
# python /home/why/evrt-detr/scripts/lrd/model_quant_test_trtexec.py \
#     /home/why/evrt-detr/video_evrtdetr_presnet50 \
#     --device cuda \
#     --batch-size 1 \
#     --input-shape 20,256,320 \
#     --enable-profiling \
#     --use-cuda-graph \
#     --skip-fp32 \
#     --skip-fp16

import torch
import subprocess
import argparse
import os
import logging
import json
import re

from evlearn.eval.eval import load_model

LOGGER = logging.getLogger('quant_test_trtexec')

DTYPE_MAP = {
    'float32'  : torch.float32,
    'float16'  : torch.float16,
    'bfloat16' : torch.bfloat16,
}

# ----------------------------
# Run trtexec command
# ----------------------------
def run_trtexec(
    onnx_path=None,
    engine_path=None,
    load_engine=None,
    precision="fp32",
    batch_size=1,
    input_shape=None,
    calib_cache=None,
    warmup_ms=200,
    duration_sec=3,
    iterations=None,
    use_cuda_graph=False,
    enable_profiling=False,
    profile_json_path=None,
    save_engine=None,
    verbose=False,
):
    """
    调用trtexec进行engine构建和benchmark
    
    返回: (latency_ms, output_text)
    """
    cmd = ["/share/apps/apptainer/bin/singularity", "exec", "--nv", "docker://nvcr.io/nvidia/tensorrt:24.12-py3",   "/opt/tensorrt/bin/trtexec"]
    
    # Model input
    if onnx_path:
        cmd.append(f"--onnx={onnx_path}")
    elif load_engine:
        cmd.append(f"--loadEngine={load_engine}")
    else:
        raise ValueError("Must specify either onnx_path or load_engine")
    
    # Precision
    if precision.lower() == "fp16":
        cmd.append("--fp16")
    elif precision.lower() == "int8":
        # 使用INT8时，同时启用FP16作为fallback
        cmd.append("--int8")
        cmd.append("--fp16")  # 关键：允许不适合INT8的层使用FP16
        if calib_cache and os.path.exists(calib_cache):
            cmd.append(f"--calib={calib_cache}")
        else:
            print(f"警告: INT8 校准cache不存在或未指定: {calib_cache}")
    
    # Input shape (only for dynamic ONNX, skip for static)
    # trtexec会自动处理静态shape，不需要--shapes
    
    # Inference options
    cmd.append(f"--warmUp={warmup_ms}")
    cmd.append(f"--duration={duration_sec}")
    
    if iterations:
        cmd.append(f"--iterations={iterations}")
    
    # Performance options
    if use_cuda_graph:
        cmd.append("--useCudaGraph")
    
    # Profiling
    if enable_profiling:
        cmd.append("--dumpProfile")
        cmd.append("--separateProfileRun")
        cmd.append("--profilingVerbosity=detailed")
        if profile_json_path:
            cmd.append(f"--exportProfile={profile_json_path}")
    
    # Engine saving
    if save_engine:
        cmd.append(f"--saveEngine={save_engine}")
    
    # Verbose
    if verbose:
        cmd.append("--verbose")
    
    # Run command
    print(f"运行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    output = result.stdout + result.stderr
    
    if result.returncode != 0:
        print(f"trtexec执行失败:")
        print(output)
        raise RuntimeError(f"trtexec failed with return code {result.returncode}")
    
    # Parse latency from output
    latency_ms = parse_latency_from_trtexec_output(output)
    
    return latency_ms, output

# ----------------------------
# Parse latency from trtexec output
# ----------------------------
def parse_latency_from_trtexec_output(output):
    """
    从trtexec输出中提取平均延迟
    查找类似: "mean = 7.75 ms" 的行
    """
    # 查找 Throughput 部分的 mean
    # 格式通常是: "Throughput: ... mean = X.XX ms ..."
    patterns = [
        r"mean\s*=\s*([\d.]+)\s*ms",
        r"Average\s*=\s*([\d.]+)\s*ms",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            latency = float(match.group(1))
            return latency
    
    # 如果没找到，尝试从最后的统计信息中提取
    lines = output.split('\n')
    for line in reversed(lines):
        if 'mean' in line.lower() and 'ms' in line.lower():
            # 尝试提取数字
            numbers = re.findall(r'([\d.]+)\s*ms', line)
            if numbers:
                return float(numbers[0])
    
    print("警告: 无法从trtexec输出中解析延迟")
    return None

# ----------------------------
# Generate INT8 calibration cache using Python
# ----------------------------
def generate_int8_calib_cache(
    onnx_path,
    calib_cache_path,
    batch_size=1,
    input_shape=(20, 256, 320),
    num_batches=10,
    inference_engine=None
):
    """
    使用Python生成INT8校准缓存
    简化版：让TensorRT自动处理校准，不手动设置每个tensor的dynamic range
    """
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    import numpy as np
    
    class MultiInputInt8Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, batches, batch_size, input_shape, inference_engine):
            super().__init__()
            self.batches = batches
            self.batch_size = batch_size
            self.input_shape = input_shape
            self.inference_engine = inference_engine
            self.current = 0
            self.cache_file = None
            
            # 为所有输入准备GPU内存
            self.device_buffers = {}
            
            # frame输入
            frame_size = int(batch_size * np.prod(input_shape) * np.float32().nbytes)
            self.device_buffers['frame'] = cuda.mem_alloc(frame_size)
            
            # is_new_frame输入 (bool)
            is_new_frame_size = int(batch_size * np.dtype(np.bool_).itemsize)
            self.device_buffers['is_new_frame'] = cuda.mem_alloc(is_new_frame_size)
            
            # memory输入
            if inference_engine:
                self.memory_list = inference_engine.init_mem(batch_size)
                input_names_list = [name for name, _ in inference_engine.input_specs]
                for idx, mem_tensor in enumerate(self.memory_list):
                    if isinstance(mem_tensor, torch.Tensor):
                        mem_size = int(mem_tensor.nelement() * mem_tensor.element_size())
                        input_name = input_names_list[2 + idx]  # 跳过frame和is_new_frame
                        self.device_buffers[input_name] = cuda.mem_alloc(mem_size)
                        # 初始化为零
                        mem_numpy = mem_tensor.cpu().numpy().astype(np.float32)
                        cuda.memcpy_htod(self.device_buffers[input_name], mem_numpy)
        
        def set_cache_file(self, cache_file):
            self.cache_file = cache_file
        
        def get_batch_size(self):
            return self.batch_size
        
        def get_batch(self, names):
            if self.current >= self.batches:
                return None
            
            # 生成随机数据
            frame_data = np.random.randn(self.batch_size, *self.input_shape).astype(np.float32)
            is_new_frame_data = np.zeros(self.batch_size, dtype=np.bool_)
            
            # 上传frame数据
            cuda.memcpy_htod(self.device_buffers['frame'], frame_data)
            cuda.memcpy_htod(self.device_buffers['is_new_frame'], is_new_frame_data)
            
            # 准备返回的指针列表（按照TensorRT请求的顺序）
            bindings = []
            for name in names:
                if name in self.device_buffers:
                    bindings.append(int(self.device_buffers[name]))
                else:
                    print(f"警告: 未知输入 {name}")
                    return None  # 失败则返回None
            
            self.current += 1
            return bindings
        
        def read_calibration_cache(self):
            if self.cache_file and os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    return f.read()
            return None
        
        def write_calibration_cache(self, cache):
            if self.cache_file:
                with open(self.cache_file, "wb") as f:
                    f.write(cache)
    
    print(f"生成INT8校准缓存: {calib_cache_path}")
    print("使用增强的INT8校准流程（手动设置dynamic range）...")
    
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
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    
    # 使用多输入校准器
    calibrator = MultiInputInt8Calibrator(num_batches, batch_size, input_shape, inference_engine)
    calibrator.set_cache_file(calib_cache_path)
    config.int8_calibrator = calibrator
    
    print(f"开始INT8校准（将运行{num_batches}批数据）...")
    
    # 【关键】为所有tensor手动设置dynamic range，解决missing scale问题
    print("为所有网络tensor设置dynamic range...")
    
    # 收集所有tensor
    tensor_set = set()
    for i in range(network.num_inputs):
        tensor_set.add(network.get_input(i))
    for i in range(network.num_outputs):
        tensor_set.add(network.get_output(i))
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        for j in range(layer.num_inputs):
            tensor_set.add(layer.get_input(j))
        for j in range(layer.num_outputs):
            tensor_set.add(layer.get_output(j))
    
    # 为所有tensor设置默认dynamic range
    count = 0
    for tensor in tensor_set:
        if tensor is not None:
            tensor_name = tensor.name if hasattr(tensor, 'name') else "unknown"
            
            # 根据tensor名称和类型设置不同的range
            if 'is_new_frame' in tensor_name or tensor.dtype == trt.bool:
                # bool类型，不需要量化
                continue
            elif 'Softmax' in tensor_name or 'softmax' in tensor_name.lower():
                # Softmax输出在[0, 1]之间
                tensor.dynamic_range = (0.0, 1.0)
            elif 'Sigmoid' in tensor_name or 'sigmoid' in tensor_name.lower():
                # Sigmoid输出在[0, 1]之间
                tensor.dynamic_range = (0.0, 1.0)
            elif 'logits' in tensor_name.lower() or 'scores' in tensor_name.lower():
                # 分类logits，范围较大
                tensor.dynamic_range = (-20.0, 20.0)
            elif 'boxes' in tensor_name.lower() or 'bbox' in tensor_name.lower():
                # 边界框坐标，通常[0, 1]归一化
                tensor.dynamic_range = (0.0, 2.0)
            elif 'Constant' in tensor_name:
                # 常量层，使用较小的范围
                tensor.dynamic_range = (-1.0, 1.0)
            else:
                # 其他tensor，使用通用范围
                tensor.dynamic_range = (-10.0, 10.0)
            
            count += 1
    
    print(f"已为 {count} 个tensor设置dynamic range")
    
    # 构建引擎
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        raise RuntimeError("Failed to build engine for calibration")
    
    print(f"INT8校准完成，cache已保存到: {calib_cache_path}")
    
    # 清理临时引擎
    del serialized_engine

# ----------------------------
# Parse command line arguments
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='使用trtexec测试模型量化（PTQ）')
    
    # Model loading
    parser.add_argument(
        'model_dir',
        metavar='MODEL_DIR',
        help='模型目录路径（包含config.json和checkpoints）',
        type=str,
    )
    
    parser.add_argument(
        '-e', '--epoch',
        default=None,
        dest='epoch',
        help='epoch编号',
        type=int,
    )
    
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cuda',
        dest='device',
        help='使用的设备 (cuda/cpu)',
        type=str,
    )
    
    # Quantization parameters
    parser.add_argument(
        '--batch-size',
        default=1,
        dest='batch_size',
        help='batch size',
        type=int,
    )
    
    parser.add_argument(
        '--input-shape',
        default="20,256,320",
        dest='input_shape',
        help='输入形状，格式: C,H,W (例如: 20,256,320)',
        type=str,
    )
    
    parser.add_argument(
        '--int8-calibration-batches',
        default=10,
        dest='int8_calibration_batches',
        help='INT8校准批次数',
        type=int,
    )
    
    parser.add_argument(
        '--onnx-output-dir',
        default=None,
        dest='onnx_output_dir',
        help='ONNX和engine文件输出目录（默认：当前目录）',
        type=str,
    )
    
    parser.add_argument(
        '--onnx-prefix',
        default="model",
        dest='onnx_prefix',
        help='ONNX文件前缀',
        type=str,
    )
    
    # trtexec specific options
    parser.add_argument(
        '--warmup-ms',
        default=200,
        dest='warmup_ms',
        help='trtexec warmup时间（毫秒）',
        type=int,
    )
    
    parser.add_argument(
        '--duration-sec',
        default=3,
        dest='duration_sec',
        help='trtexec benchmark持续时间（秒）',
        type=int,
    )
    
    parser.add_argument(
        '--iterations',
        default=None,
        dest='iterations',
        help='trtexec运行迭代次数（如果指定，则覆盖duration）',
        type=int,
    )
    
    parser.add_argument(
        '--use-cuda-graph',
        action='store_true',
        dest='use_cuda_graph',
        help='使用CUDA Graph加速推理',
    )
    
    parser.add_argument(
        '--enable-profiling',
        action='store_true',
        dest='enable_profiling',
        help='启用逐层profiling（显示每层耗时）',
    )
    
    parser.add_argument(
        '--profile-output-dir',
        default=None,
        dest='profile_output_dir',
        help='Profiling JSON文件输出目录（默认：与ONNX输出目录相同）',
        type=str,
    )
    
    parser.add_argument(
        '--skip-fp32',
        action='store_true',
        dest='skip_fp32',
        help='跳过FP32测试',
    )
    
    parser.add_argument(
        '--skip-fp16',
        action='store_true',
        dest='skip_fp16',
        help='跳过FP16测试',
    )
    
    parser.add_argument(
        '--skip-int8',
        action='store_true',
        dest='skip_int8',
        help='跳过INT8测试',
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        dest='verbose',
        help='trtexec verbose输出',
    )
    
    parser.add_argument(
        '--keep-engines',
        action='store_true',
        dest='keep_engines',
        help='保留生成的engine文件（默认会删除以节省空间）',
    )
    
    return parser.parse_args()

# ----------------------------
# Parse input shape
# ----------------------------
def parse_input_shape(shape_str):
    """解析输入形状字符串"""
    try:
        parts = [int(x.strip()) for x in shape_str.split(',')]
        if len(parts) == 3:
            return tuple(parts)
        else:
            raise ValueError(f"输入形状格式不正确，应为 C,H,W，得到: {shape_str}")
    except ValueError as e:
        raise ValueError(f"无法解析输入形状: {e}")

# ----------------------------
# Main function
# ----------------------------
def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Parse input shape
    input_shape = parse_input_shape(args.input_shape)
    print(f"输入形状: {input_shape}")
    
    # Setup output directory
    if args.onnx_output_dir is None:
        onnx_output_dir = os.getcwd()
    else:
        onnx_output_dir = args.onnx_output_dir
        os.makedirs(onnx_output_dir, exist_ok=True)
    
    # Setup profiling output directory
    if args.profile_output_dir is None:
        profile_output_dir = onnx_output_dir
    else:
        profile_output_dir = args.profile_output_dir
        os.makedirs(profile_output_dir, exist_ok=True)
    
    # ----------------------------
    # Step 1: Load model
    # ----------------------------
    print("\n" + "="*60)
    print("步骤 1: 加载模型")
    print("="*60)
    model_args, model = load_model(args.model_dir, args.epoch, args.device)
    
    # Check model parameters
    total_params = 0
    for net_name, net in model._nets.items():
        net_params = sum(p.numel() for p in net.parameters())
        total_params += net_params
        print(f"  {net_name} 参数量: {net_params:,}")
    print(f"模型总参数量: {total_params:,}")
    
    # ----------------------------
    # Step 2: Export ONNX
    # ----------------------------
    print("\n" + "="*60)
    print("步骤 2: 导出ONNX")
    print("="*60)
    
    # 创建inference engine
    print("创建推理引擎...")
    model.to_dtype(DTYPE_MAP['float32'])  # 确保使用FP32进行导出
    model.eval()
    
    # 使用fuse_postproc=False以便导出纯模型（不包含后处理）
    inference_engine = model.construct_inference_engine(fuse_postproc=False)
    
    # 构建torch模型用于ONNX导出
    torch_model = inference_engine.construct_torch_model().to(args.device)
    torch_model.eval()
    
    # 准备输入
    frame = torch.randn((args.batch_size, *input_shape), 
                       device=args.device, dtype=torch.float32)
    is_new_frame = torch.zeros(args.batch_size, dtype=torch.bool, device=args.device)
    memory = inference_engine.init_mem(args.batch_size)
    
    # 获取输入输出名称
    input_names = [name for name, _ in inference_engine.input_specs]
    output_names = list(inference_engine.output_names)
    
    onnx_fp32_path = os.path.join(onnx_output_dir, f"{args.onnx_prefix}_fp32.onnx")
    
    print(f"导出FP32 ONNX到 {onnx_fp32_path} ...")
    print(f"  输入: {input_names}")
    print(f"  输出: {output_names}")
    
    inputs = (frame, is_new_frame, *memory)
    
    with torch.no_grad():
        torch.onnx.export(
            torch_model,
            inputs,
            onnx_fp32_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=17,
            do_constant_folding=True,
        )
    print("FP32 ONNX导出完成")
    
    # ----------------------------
    # Step 3: 使用trtexec构建引擎并测试
    # ----------------------------
    print("\n" + "="*60)
    print("步骤 3: 使用trtexec构建引擎并测试")
    print("="*60)
    
    # Prepare INT8 calibration cache if needed
    calib_cache_path = None
    if not args.skip_int8:
        calib_cache_path = os.path.join(onnx_output_dir, f"{args.onnx_prefix}_int8_calib.cache")
        if not os.path.exists(calib_cache_path):
            generate_int8_calib_cache(
                onnx_fp32_path,
                calib_cache_path,
                batch_size=args.batch_size,
                input_shape=input_shape,
                num_batches=args.int8_calibration_batches,
                inference_engine=inference_engine
            )
        else:
            print(f"使用已存在的校准缓存: {calib_cache_path}")
    
    engines = {}
    if not args.skip_fp32:
        engines["FP32"] = {
            "precision": "fp32",
            "engine_path": os.path.join(onnx_output_dir, f"{args.onnx_prefix}_fp32.engine"),
            "calib_cache": None,
        }
    if not args.skip_fp16:
        engines["FP16"] = {
            "precision": "fp16",
            "engine_path": os.path.join(onnx_output_dir, f"{args.onnx_prefix}_fp16.engine"),
            "calib_cache": None,
        }
    if not args.skip_int8:
        engines["INT8"] = {
            "precision": "int8",
            "engine_path": os.path.join(onnx_output_dir, f"{args.onnx_prefix}_int8.engine"),
            "calib_cache": calib_cache_path,
        }
    
    results = {}
    for name, config in engines.items():
        print(f"\n处理 {name}...")
        
        # Prepare profiling output path
        profile_json_path = None
        if args.enable_profiling:
            profile_json_path = os.path.join(
                profile_output_dir,
                f"{args.onnx_prefix}_{config['precision']}_profile.json"
            )
        
        # Run trtexec
        latency_ms, output = run_trtexec(
            onnx_path=onnx_fp32_path,
            precision=config["precision"],
            batch_size=args.batch_size,
            input_shape=input_shape,
            calib_cache=config["calib_cache"],
            warmup_ms=args.warmup_ms,
            duration_sec=args.duration_sec,
            iterations=args.iterations,
            use_cuda_graph=args.use_cuda_graph,
            enable_profiling=args.enable_profiling,
            profile_json_path=profile_json_path,
            save_engine=config["engine_path"],
            verbose=args.verbose,
        )
        
        if latency_ms:
            results[name] = latency_ms
            print(f"{name}: {latency_ms:.2f} ms per batch of {args.batch_size}")
        else:
            print(f"{name}: 无法解析延迟")
        
        # Clean up engine file if requested
        if not args.keep_engines and os.path.exists(config["engine_path"]):
            os.remove(config["engine_path"])
            print(f"已删除engine文件: {config['engine_path']}")
    
    # ----------------------------
    # Print summary
    # ----------------------------
    print("\n" + "="*60)
    print("量化测试总结 (使用trtexec)")
    print("="*60)
    print(f"模型: {args.model_dir}")
    print(f"输入形状: {input_shape}")
    print(f"Batch Size: {args.batch_size}")
    print(f"CUDA Graph: {'启用' if args.use_cuda_graph else '禁用'}")
    print(f"Profiling: {'启用' if args.enable_profiling else '禁用'}")
    print("-"*60)
    for k, v in results.items():
        print(f"{k:>4}: {v:.2f} ms per batch of {args.batch_size}")
    print("="*60)
    
    # Calculate speedup if multiple precisions tested
    if len(results) > 1:
        if "FP32" in results:
            fp32_latency = results["FP32"]
            print("\n相对FP32的加速比:")
            for k, v in results.items():
                if k != "FP32":
                    speedup = fp32_latency / v
                    print(f"{k:>4}: {speedup:.2f}x")

if __name__ == '__main__':
    main()
