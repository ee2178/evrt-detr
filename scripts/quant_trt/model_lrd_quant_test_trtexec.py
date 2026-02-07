#!/usr/bin/env python3
"""
整合低秩分解和量化测试的脚本（使用trtexec）- 整个模型版本
先对整个模型进行低秩分解，然后使用trtexec测试量化（PTQ）的运行速度
"""
# python /home/why/evrt-detr/scripts/lrd/model_lrd_quant_test_trtexec.py \
#     /home/why/evrt-detr/video_evrtdetr_presnet50 \
#     --ratio 2.0 \
#     --device cuda \
#     --batch-size 1 \
#     --input-shape 20,256,320 \
#     --enable-profiling \
#     --use-cuda-graph \
#     --skip-fp32 \
#     --skip-fp16

# python /home/why/evrt-detr/scripts/lrd/model_lrd_quant_test_trtexec.py \
#     /home/why/evrt-detr/video_evrtdetr_presnet50 \
#     --whitelist-file /home/why/evrt-detr/scripts/efficiency/conv2d_best_per_layer_bs_4.txt \
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
import real_lrd_resnet as lrd

LOGGER = logging.getLogger('lrd_quant_test_trtexec')

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
    cmd = ["/usr/src/tensorrt/bin/trtexec"]
    
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
        cmd.append("--int8")
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
    num_batches=10
):
    """
    使用Python生成INT8校准缓存
    这部分还是用Python API，因为trtexec不太好做自定义校准数据
    
    改进版：为所有中间tensor（包括Tucker/LRD分解层的中间输出）设置dynamic range
    """
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    import numpy as np
    
    class RandomInt8Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, batches, batch_size, input_shape):
            super().__init__()
            self.batches = batches
            self.batch_size = batch_size
            self.input_shape = input_shape
            self.current = 0
            self.device_input = cuda.mem_alloc(
                int(batch_size * np.prod(input_shape) * np.float32().nbytes)
            )
            self.cache_file = None
        
        def set_cache_file(self, cache_file):
            self.cache_file = cache_file
        
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
            if self.cache_file and os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    return f.read()
            return None
        
        def write_calibration_cache(self, cache):
            if self.cache_file:
                with open(self.cache_file, "wb") as f:
                    f.write(cache)
    
    print(f"生成INT8校准缓存: {calib_cache_path}")
    
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
    
    calibrator = RandomInt8Calibrator(num_batches, batch_size, input_shape)
    calibrator.set_cache_file(calib_cache_path)
    config.int8_calibrator = calibrator
    
    # 【关键改进】为所有中间tensor设置dynamic range
    # 这样Tucker/LRD分解的中间层也能使用INT8
    print("为所有中间tensor设置dynamic range...")
    
    # 收集所有需要量化的tensor
    tensors_to_quantize = []
    tensor_names = set()
    
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        for j in range(layer.num_outputs):
            tensor = layer.get_output(j)
            tensor_name = tensor.name
            if tensor_name not in tensor_names:
                tensor_names.add(tensor_name)
                tensors_to_quantize.append((tensor, tensor_name))
    
    print(f"找到 {len(tensors_to_quantize)} 个tensor需要量化")
    
    # 运行一次推理来收集激活值范围
    print("运行推理以收集激活值范围...")
    
    # 先构建一个临时的FP32引擎来收集激活值
    temp_config = builder.create_builder_config()
    temp_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    
    # 设置所有层的输出为network output（临时）
    original_outputs = []
    for i in range(network.num_outputs):
        original_outputs.append(network.get_output(i))
    
    # 标记所有中间tensor为输出
    for tensor, _ in tensors_to_quantize:
        if not tensor.is_network_output:
            network.mark_output(tensor)
    
    # 构建临时引擎
    print("构建临时FP32引擎用于收集激活范围...")
    temp_engine_data = builder.build_serialized_network(network, temp_config)
    
    if temp_engine_data is None:
        print("警告：无法构建临时引擎，使用默认dynamic range")
        # 使用保守的默认值
        for tensor, tensor_name in tensors_to_quantize:
            if "tucker" in tensor_name.lower() or "lrd" in tensor_name.lower():
                # 为Tucker和LRD的中间层设置更大的范围
                tensor.dynamic_range = (-10.0, 10.0)
                print(f"  设置默认range: {tensor_name} = [-10.0, 10.0]")
    else:
        # 使用临时引擎收集激活值
        import tensorrt as trt
        runtime = trt.Runtime(logger)
        temp_engine = runtime.deserialize_cuda_engine(temp_engine_data)
        context = temp_engine.create_execution_context()
        
        # 准备输入输出缓冲区
        input_binding_idx = temp_engine.get_binding_index("input")
        input_shape = (batch_size, *input_shape)
        input_data = np.random.randn(*input_shape).astype(np.float32)
        d_input = cuda.mem_alloc(input_data.nbytes)
        cuda.memcpy_htod(d_input, input_data)
        
        # 为所有输出分配缓冲区
        bindings = [None] * temp_engine.num_bindings
        bindings[input_binding_idx] = int(d_input)
        
        output_buffers = {}
        for i in range(temp_engine.num_bindings):
            if i == input_binding_idx:
                continue
            binding_name = temp_engine.get_binding_name(i)
            binding_shape = temp_engine.get_binding_shape(i)
            binding_dtype = trt.nptype(temp_engine.get_binding_dtype(i))
            
            # 计算元素数量
            size = 1
            for dim in binding_shape:
                size *= dim
            
            output_buffer = np.empty(size, dtype=binding_dtype)
            d_output = cuda.mem_alloc(output_buffer.nbytes)
            bindings[i] = int(d_output)
            output_buffers[binding_name] = (output_buffer, d_output)
        
        # 运行多次推理收集统计
        print(f"运行 {min(num_batches, 5)} 次推理收集激活统计...")
        activation_ranges = {}
        
        for batch_idx in range(min(num_batches, 5)):
            input_data = np.random.randn(*input_shape).astype(np.float32)
            cuda.memcpy_htod(d_input, input_data)
            
            context.execute_v2(bindings)
            
            # 收集输出
            for name, (buffer, d_buffer) in output_buffers.items():
                cuda.memcpy_dtoh(buffer, d_buffer)
                
                if name not in activation_ranges:
                    activation_ranges[name] = {'min': float('inf'), 'max': float('-inf')}
                
                activation_ranges[name]['min'] = min(activation_ranges[name]['min'], buffer.min())
                activation_ranges[name]['max'] = max(activation_ranges[name]['max'], buffer.max())
        
        # 设置dynamic range
        print("为tensor设置dynamic range:")
        for tensor, tensor_name in tensors_to_quantize:
            if tensor_name in activation_ranges:
                min_val = activation_ranges[tensor_name]['min']
                max_val = activation_ranges[tensor_name]['max']
                # 使用对称范围
                abs_max = max(abs(min_val), abs(max_val))
                if abs_max > 0:
                    tensor.dynamic_range = (-abs_max, abs_max)
                    if "tucker" in tensor_name.lower() or "lrd" in tensor_name.lower():
                        print(f"  ✓ {tensor_name}: [{-abs_max:.4f}, {abs_max:.4f}]")
                else:
                    tensor.dynamic_range = (-1.0, 1.0)
            else:
                # 对于没有收集到的tensor，使用默认值
                tensor.dynamic_range = (-5.0, 5.0)
                if "tucker" in tensor_name.lower() or "lrd" in tensor_name.lower():
                    print(f"  ? {tensor_name}: [-5.0, 5.0] (默认)")
        
        # 清理
        del context
        del temp_engine
        del runtime
    
    # 恢复原始的network outputs
    for i in range(network.num_outputs):
        network.unmark_output(network.get_output(0))
    for output in original_outputs:
        network.mark_output(output)
    
    # 现在构建真正的INT8引擎
    print("执行INT8校准并构建引擎...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        raise RuntimeError("Failed to build engine for calibration")
    
    print(f"INT8校准完成，cache已保存到: {calib_cache_path}")
    print("所有Tucker/LRD分解的中间层现在都应该使用INT8精度")

# ----------------------------
# Parse command line arguments
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='低秩分解后使用trtexec测试量化（PTQ）')
    
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
    
    # Low rank decomposition parameters
    parser.add_argument(
        '--ratio',
        default=0.5,
        dest='ratio',
        help='参数比例或秩比例 (0.0-1.0)',
        type=float,
    )
    
    parser.add_argument(
        '--ratio-mode',
        default="param",
        dest='ratio_mode',
        help='比例模式 (param: 参数比例, rank: 秩比例)',
        type=str,
    )
    
    parser.add_argument(
        '--scheme',
        default=1,
        dest='scheme',
        help='分解方案 (1: SVD, 2: SVD, 3: Tucker-2)',
        type=int,
    )
    
    parser.add_argument(
        '--whitelist-file',
        default=None,
        dest='whitelist_file',
        help='白名单文件: layer_name,method(svd1/svd2/tucker/orig),rank_ratio',
        type=str,
    )
    
    parser.add_argument(
        '--skip-1x1',
        action='store_true',
        dest='skip_1x1',
        help='跳过1x1卷积层的分解',
    )
    
    parser.add_argument(
        '--skip-if-no-mac-reduction',
        action='store_true',
        dest='skip_if_no_mac_reduction',
        help='如果估计的MACs不减少则跳过分解',
    )
    
    parser.add_argument(
        '--skip-if-rank-gt-out',
        action='store_true',
        dest='skip_if_rank_gt_out',
        help='如果rank超过out_channels则跳过scheme2',
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
    
    # Check original parameters
    original_params = sum(p.numel() for p in model._nets.parameters())
    print(f"原始模型参数量: {original_params:,}")
    
    # ----------------------------
    # Step 2: Apply low rank decomposition
    # ----------------------------
    print("\n" + "="*60)
    print("步骤 2: 应用低秩分解")
    print("="*60)
    
    whitelist_map = None
    if args.whitelist_file:
        whitelist_map = {}
        with open(args.whitelist_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    raise ValueError(f"无效的白名单行: {line}")
                name, method, ratio_text = parts[:3]
                method = method.lower()
                if method not in ("svd1", "svd2", "tucker", "orig"):
                    raise ValueError(f"白名单中无效的方法: {method}")
                ratio_val = float(ratio_text) if method != "orig" else 1.0
                whitelist_map[name] = (method, ratio_val)
                if not name.startswith("backbone."):
                    whitelist_map[f"backbone.{name}"] = (method, ratio_val)
    
    if whitelist_map is not None:
        LOGGER.info(
            f"使用白名单应用LRD: {args.whitelist_file} "
            f"(条目数={len(whitelist_map)})"
        )
    else:
        LOGGER.info(f"使用比例应用LRD: ratio={args.ratio}, scheme={args.scheme}")
    
    # Apply low rank decomposition
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
    
    # Print decomposition statistics
    if lrd.total_rank_global > 0:
        rank_ratio = lrd.retained_rank_global / lrd.total_rank_global
        LOGGER.info(
            f"总秩: {lrd.total_rank_global}, 保留秩: {lrd.retained_rank_global}, "
            f"秩比例: {rank_ratio:.3f}"
        )
    else:
        LOGGER.info(
            f"总秩: {lrd.total_rank_global}, 保留秩: {lrd.retained_rank_global}, "
            "秩比例: N/A (无分解层)"
        )
    
    if lrd.total_params_global > 0:
        params_ratio = lrd.retained_params_global / lrd.total_params_global
        LOGGER.info(
            f"总参数: {lrd.total_params_global}, 保留参数: {lrd.retained_params_global}, "
            f"参数比例: {params_ratio:.3f}"
        )
    else:
        LOGGER.info(
            f"总参数: {lrd.total_params_global}, 保留参数: {lrd.retained_params_global}, "
            "参数比例: N/A (无分解层)"
        )
    
    # Check decomposed parameters
    total_params = sum(p.numel() for p in model._nets.parameters())
    param_ratio = total_params / original_params
    print(f"分解后模型参数量: {total_params:,}, 参数比例: {param_ratio:.3f}")
    
    # ----------------------------
    # Step 3: Export ONNX
    # ----------------------------
    print("\n" + "="*60)
    print("步骤 3: 导出ONNX")
    print("="*60)
    
    # 使用整个模型而不是只用 backbone
    model_fp32 = model._nets
    model_fp32.eval()
    
    dummy_input = torch.randn(args.batch_size, *input_shape).to(args.device)
    
    if args.ratio < 1.0:
        args.onnx_prefix = f"{args.onnx_prefix}_lrd"
    else:
        args.onnx_prefix = f"{args.onnx_prefix}"
        
    onnx_fp32_path = os.path.join(onnx_output_dir, f"{args.onnx_prefix}_fp32.onnx")
    
    print(f"导出FP32 ONNX到 {onnx_fp32_path} ...")
    torch.onnx.export(
        model_fp32,
        dummy_input,
        onnx_fp32_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )
    print("FP32 ONNX导出完成")
    
    # ----------------------------
    # Step 4: 使用trtexec构建引擎并测试
    # ----------------------------
    print("\n" + "="*60)
    print("步骤 4: 使用trtexec构建引擎并测试")
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
                num_batches=args.int8_calibration_batches
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
    print("延迟测试总结 (使用trtexec)")
    print("="*60)
    print(f"模型: {args.model_dir}")
    print(f"分解比例: {args.ratio} ({args.ratio_mode})")
    print(f"分解方案: {args.scheme}")
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
