#!/usr/bin/env python3
"""
Low Rank Decomposition for ResNet backbone in EVRT-DETR model.
Performs SVD decomposition on convolutional layers and truncates based on parameter ratio.
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.nn import Conv2d
from collections import OrderedDict
import logging
import pandas as pd

from evlearn.config.args import Args
from evlearn.models import construct_model
from evlearn.bundled.leanbase.torch.funcs import seed_everything
from evlearn.eval.eval   import load_model, load_eval_dset
from evlearn.train.train import eval_epoch

def make_eval_directory(model, savedir, mkdir = True):
    result = os.path.join(savedir, 'evals')

    if model.current_epoch is None:
        result = os.path.join(result, 'final')
    else:
        result = os.path.join(result, f'epoch_{model.current_epoch}')

    if mkdir:
        os.makedirs(result, exist_ok = True)

    return result

def save_metrics(evaldir, data_name, data_path, split, steps, metrics):
    # pylint: disable=too-many-arguments
    fname = f'metrics_data({data_name})'

    if data_path is not None:
        fname += f'_path({data_path})'

    fname += f'_split({split})'

    if steps is not None:
        fname += f'_nb({steps})'

    fname += '.csv'
    path   = os.path.join(evaldir, fname)

    df = pd.Series(metrics).to_frame().T
    df.to_csv(path, index = False)

def eval_single_dataset(
    model, args, data_name, data_config, split, steps, evaldir, batch_size,
    workers, data_path
):
    # pylint: disable=too-many-arguments
    if batch_size is not None:
        data_config.batch_size = batch_size

    if workers is not None:
        data_config.workers = workers

    if data_path is not None:
        data_config.dataset['path'] = data_path

    args.config.data.eval = { data_name : data_config }
    dl = load_eval_dset(args, split = split)

    metrics = eval_epoch(
        dl, model,
        title           = f'Evaluation: {data_name}',
        steps_per_epoch = steps
    )

    save_metrics(evaldir, data_name, data_path, split, steps, metrics.get())

LOGGER = logging.getLogger('lrd_resnet')


def _safe_svd(matrix: torch.Tensor):
    """Run SVD with a safe fallback for older PyTorch versions."""
    try:
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        v = vh.transpose(-2, -1)
        return u, s, v
    except Exception:
        u, s, v = torch.svd(matrix)
        return u, s, v

def parse_cmdargs():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Low Rank Decomposition for ResNet backbone')

    parser.add_argument(
        'model_dir',
        metavar='MODEL_DIR',
        help='model directory containing config.json and checkpoints',
        type=str,
    )

    parser.add_argument(
        '--ratio',
        default=0.5,
        dest='ratio',
        help='parameter ratio or rank ratio for rank truncation (0.0-1.0)',
        type=float,
    )
    
    parser.add_argument(
        '--ratio-mode',
        default="param",
        dest='ratio_mode',
        help='ratio mode for rank truncation (param: parameter ratio, rank: rank ratio)',
        type=str,
    )
    
    parser.add_argument(
        '--scheme',
        default=1,
        dest='scheme',
        help='scheme for rank truncation (1: SVD [out_channels, in_channels * kH * kW], 2: SVD [out_channels * kH, in_channels * kW], 3: Tucker-2 channel-wise decomposition)',
        type=int,
    )
    parser.add_argument(
        '--whitelist-file',
        default=None,
        dest='whitelist_file',
        help='whitelist file: layer_name,method(svd1/svd2/tucker/orig),rank_ratio',
        type=str,
    )
    parser.add_argument(
        '--skip-1x1',
        action='store_true',
        dest='skip_1x1',
        help='skip decomposition for 1x1 conv layers (likely memory-bound)',
    )
    parser.add_argument(
        '--skip-if-no-mac-reduction',
        action='store_true',
        dest='skip_if_no_mac_reduction',
        help='skip decomposition if estimated MACs/pos do not decrease',
    )
    parser.add_argument(
        '--skip-if-rank-gt-out',
        action='store_true',
        dest='skip_if_rank_gt_out',
        help='skip scheme2 if rank per group exceeds out_channels per group',
    )

    parser.add_argument(
        '--epoch',
        default=None,
        dest='epoch',
        help='epoch to load (default: latest)',
        type=int,
    )

    parser.add_argument(
        '--output-dir',
        default=None,
        dest='output_dir',
        help='output directory for decomposed model (default: model_dir + "_lrd")',
        type=str,
    )

    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cuda',
        dest='device',
        help='device to use for decomposition',
        type=str,
    )

    parser.add_argument(
        '--data-name',
        default=None,
        dest='data_name',
        help='name of the dataset to evaluate',
        type=str,
    )

    parser.add_argument(
        '--split',
        default='test',
        dest='split',
        help='dataset split',
        type=str,
    )

    parser.add_argument(
        '--steps',
        default=None,
        dest='steps',
        help='steps for evaluation',
        type=int,
    )

    parser.add_argument(
        '--batch-size',
        default=None,
        dest='batch_size',
        help='batch size for evaluation',
        type=int,
    )

    parser.add_argument(
        '--workers',
        default=None,
        dest='workers',
        help='number of workers to use for evaluation',
        type=int,
    )

    parser.add_argument(
        '--data-path',
        default=None,
        dest='data_path',
        help='path to the new dataset to evaluate',
        type=str,
    )

    return parser.parse_args()

def get_conv_layers(model):
    """Extract all Conv2d layers from ResNet backbone."""
    conv_layers = []

    def traverse_module(module, prefix=""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, Conv2d):
                conv_layers.append((module, name, full_name, child))
            else:
                traverse_module(child, full_name)

    # Focus on backbone - it's in model._nets.backbone
    if hasattr(model, "_nets") and hasattr(model._nets, "backbone"):
        traverse_module(model._nets.backbone, "backbone")
    elif hasattr(model, "backbone") and isinstance(model, nn.Module):
        traverse_module(model.backbone, "backbone")
    elif hasattr(model, "backbone") and isinstance(model.backbone, nn.Module):
        traverse_module(model.backbone, "")
    else:
        # Try to find backbone in model attributes
        for attr_name in dir(model):
            if "backbone" in attr_name.lower():
                backbone = getattr(model, attr_name)
                if isinstance(backbone, nn.Module):
                    traverse_module(backbone, "")
                    break

    return conv_layers

total_rank_global = 0
retained_rank_global = 0

total_params_global = 0
retained_params_global = 0

def _set_module_by_parent(parent_module, child_name, new_module):
    """Replace a direct child module by name."""
    if child_name not in parent_module._modules:
        raise KeyError(f"Module child not found: {child_name}")
    parent_module._modules[child_name] = new_module


def _compute_target_rank(weight_2d, ratio, ratio_mode):
    max_rank = min(weight_2d.shape[0], weight_2d.shape[1])
    orig_params = weight_2d.shape[0] * weight_2d.shape[1]
    if ratio_mode == "param":
        target_rank_float = ratio * orig_params / (weight_2d.shape[0] + weight_2d.shape[1])
    elif ratio_mode == "rank":
        target_rank_float = ratio * max_rank
    else:
        raise ValueError(f"Invalid ratio mode: {ratio_mode}")
    target_rank = max(1, min(max_rank, int(target_rank_float)))
    return target_rank, max_rank, orig_params, target_rank_float


def tucker2_decompose_conv_layer_to_module(
    conv_layer,
    ratio,
    ratio_mode,
    device,
    skip_1x1=False,
):
    """
    Tucker-2 decomposition for Conv2d into 3 conv layers:
      1x1 (in -> r_in) + kxk core (r_in -> r_out) + 1x1 (r_out -> out)
    Uses rank ratio on channel dims; ratio_mode 'param' is treated as rank ratio.
    """
    global total_rank_global, retained_rank_global, total_params_global, retained_params_global

    if conv_layer.groups != 1:
        raise ValueError("Tucker-2 does not support grouped conv layers.")

    weight = conv_layer.weight.data  # [out_channels, in_channels, kH, kW]
    out_channels, in_channels, kH, kW = weight.shape

    if skip_1x1 and kH == 1 and kW == 1:
        LOGGER.info("Skip 1x1 conv due to --skip-1x1")
        return None

    if ratio_mode != "rank":
        LOGGER.info("Tucker-2 uses rank ratio on channels; ignoring ratio_mode='param'.")

    rank_out = max(1, int(out_channels * ratio))
    rank_in = max(1, int(in_channels * ratio))

    # Compute factor matrices on CPU float for stability.
    weight_cpu = weight.detach().float().cpu()
    weight_out = weight_cpu.view(out_channels, -1)
    U_out, _, _ = _safe_svd(weight_out)
    U_out = U_out[:, :rank_out]  # [out_channels, r_out]

    weight_in = weight_cpu.permute(1, 0, 2, 3).contiguous().view(in_channels, -1)
    U_in, _, _ = _safe_svd(weight_in)
    U_in = U_in[:, :rank_in]  # [in_channels, r_in]

    # Core tensor: W x1 U_out^T x2 U_in^T
    core = torch.einsum('ro,oikl->rikl', U_out.t(), weight_cpu)
    core = torch.einsum('ri,oikl->orkl', U_in.t(), core)

    first_conv = nn.Conv2d(
        in_channels,
        rank_in,
        kernel_size=(1, 1),
        stride=1,
        padding=0,
        dilation=1,
        groups=conv_layer.groups,
        bias=False,
    )
    core_conv = nn.Conv2d(
        rank_in,
        rank_out,
        kernel_size=(kH, kW),
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        dilation=conv_layer.dilation,
        groups=conv_layer.groups,
        bias=False,
    )
    last_conv = nn.Conv2d(
        rank_out,
        out_channels,
        kernel_size=(1, 1),
        stride=1,
        padding=0,
        dilation=1,
        groups=conv_layer.groups,
        bias=(conv_layer.bias is not None),
    )

    first_conv.weight.data = U_in.t().view(rank_in, in_channels, 1, 1)
    core_conv.weight.data = core
    last_conv.weight.data = U_out.view(out_channels, rank_out, 1, 1)
    if conv_layer.bias is not None:
        last_conv.bias.data.copy_(conv_layer.bias.data)

    # Update stats
    orig_params = weight.numel()
    decomposed_params = (
        rank_in * in_channels
        + rank_out * rank_in * kH * kW
        + out_channels * rank_out
    )
    total_rank_global += (out_channels + in_channels)
    retained_rank_global += (rank_out + rank_in)
    total_params_global += orig_params
    retained_params_global += decomposed_params

    module = nn.Sequential(OrderedDict([
        ("tucker_conv1", first_conv),
        ("tucker_core", core_conv),
        ("tucker_conv3", last_conv),
    ]))
    return module.to(device)


def svd_lrd_decompose_conv_layer_to_module(
    conv_layer,
    ratio,
    ratio_mode,
    scheme,
    device,
    skip_1x1=False,
    skip_if_no_mac_reduction=False,
    skip_if_rank_gt_out=False,
):
    """
    Perform low rank decomposition on a Conv2d layer using SVD and return a 2-layer module.
    Keeps bias/padding/stride/dilation/groups consistent with the original Conv2d.
    """
    global total_rank_global, retained_rank_global, total_params_global, retained_params_global

    weight = conv_layer.weight.data  # [out_channels, in_channels/groups, kH, kW]
    orig_shape = weight.shape
    out_channels, in_channels_per_group, kH, kW = orig_shape

    groups = conv_layer.groups
    if out_channels % groups != 0:
        raise ValueError(f"out_channels {out_channels} not divisible by groups {groups}")
    out_channels_per_group = out_channels // groups

    # Prepare per-group containers
    rank_per_group = None
    max_rank_per_group = None
    orig_params_per_group = None
    target_rank_float = None

    if scheme == 1:
        # [out_channels_per_group, in_channels_per_group * kH * kW]
        max_rank_per_group = min(out_channels_per_group, in_channels_per_group * kH * kW)
        orig_params_per_group = out_channels_per_group * in_channels_per_group * kH * kW
        if ratio_mode == "param":
            target_rank_float = (
                ratio
                * orig_params_per_group
                / (out_channels_per_group + in_channels_per_group * kH * kW)
            )
        elif ratio_mode == "rank":
            target_rank_float = ratio * max_rank_per_group
        else:
            raise ValueError(f"Invalid ratio mode: {ratio_mode}")
        rank_per_group = max(1, min(max_rank_per_group, int(target_rank_float)))
    elif scheme == 2:
        if kH != kW:
            raise ValueError(f"Scheme 2 expects square kernels, got {kH}x{kW}")
        max_rank_per_group = min(out_channels_per_group * kW, in_channels_per_group * kH)
        orig_params_per_group = out_channels_per_group * in_channels_per_group * kH * kW
        if ratio_mode == "param":
            target_rank_float = (
                ratio
                * orig_params_per_group
                / (out_channels_per_group * kW + in_channels_per_group * kH)
            )
        elif ratio_mode == "rank":
            target_rank_float = ratio * max_rank_per_group
        else:
            raise ValueError(f"Invalid ratio mode: {ratio_mode}")
        rank_per_group = max(1, min(max_rank_per_group, int(target_rank_float)))
    else:
        raise ValueError(f"Invalid scheme: {scheme}")

    if skip_1x1 and kH == 1 and kW == 1:
        LOGGER.info("Skip 1x1 conv due to --skip-1x1")
        return None

    # Estimate per-position MAC ratio to spot cases that may not speed up.
    orig_ops_per_pos = out_channels_per_group * in_channels_per_group * kH * kW
    if scheme == 1:
        decomposed_ops_per_pos = (
            rank_per_group * in_channels_per_group * kH * kW
            + out_channels_per_group * rank_per_group
        )
    else:
        decomposed_ops_per_pos = (
            rank_per_group * in_channels_per_group * kW
            + out_channels_per_group * rank_per_group * kH
        )

    LOGGER.info(
        f"Original shape: {orig_shape}, Rank per group: {rank_per_group}/{max_rank_per_group} "
        f"(target {ratio_mode} ratio: {ratio:.3f}), "
        f"MACs/pos ratio: {decomposed_ops_per_pos / orig_ops_per_pos:.3f}"
    )
    if scheme == 2 and rank_per_group > out_channels_per_group:
        LOGGER.info(
            "Scheme 2 rank exceeds out_channels per group; "
            "intermediate activations may be larger and slow down."
        )
        if skip_if_rank_gt_out:
            LOGGER.info("Skip due to --skip-if-rank-gt-out")
            return None
    if decomposed_ops_per_pos > orig_ops_per_pos:
        LOGGER.info(
            "Decomposed MACs per position exceed original; "
            "speedup is unlikely for this layer."
        )
        if skip_if_no_mac_reduction:
            LOGGER.info("Skip due to --skip-if-no-mac-reduction")
            return None

    total_rank_global += max_rank_per_group * groups
    retained_rank_global += rank_per_group * groups
    total_params_global += orig_params_per_group * groups

    if ratio >= 1.0:
        LOGGER.info(f"Ratio {ratio:.3f} >= 1.0, skipping decomposition for layer")
        return None

    # Create two conv layers
    if scheme == 1:
        first_conv = nn.Conv2d(
            in_channels=conv_layer.in_channels,
            out_channels=rank_per_group * groups,
            kernel_size=(kH, kW),
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=groups,
            bias=False,
        )
        second_conv = nn.Conv2d(
            in_channels=rank_per_group * groups,
            out_channels=conv_layer.out_channels,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            dilation=1,
            groups=groups,
            bias=(conv_layer.bias is not None),
        )
        # Parameter count for scheme 1 (weights only)
        decomposed_params_per_group = (
            rank_per_group * in_channels_per_group * kH * kW
            + out_channels_per_group * rank_per_group
        )
    else:
        # Scheme 2: horizontal then vertical (consistent with reshape)
        first_conv = nn.Conv2d(
            in_channels=conv_layer.in_channels,
            out_channels=rank_per_group * groups,
            kernel_size=(1, kW),
            stride=(1, conv_layer.stride[1]),
            padding=(0, conv_layer.padding[1]),
            dilation=(1, conv_layer.dilation[1]),
            groups=groups,
            bias=False,
        )
        second_conv = nn.Conv2d(
            in_channels=rank_per_group * groups,
            out_channels=conv_layer.out_channels,
            kernel_size=(kH, 1),
            stride=(conv_layer.stride[0], 1),
            padding=(conv_layer.padding[0], 0),
            dilation=(conv_layer.dilation[0], 1),
            groups=groups,
            bias=(conv_layer.bias is not None),
        )
        # Parameter count for scheme 2 (weights only)
        decomposed_params_per_group = (
            rank_per_group * in_channels_per_group * kW
            + out_channels_per_group * rank_per_group * kH
        )

    retained_params_global += decomposed_params_per_group * groups

    # Allocate weights and fill per-group
    first_conv.weight.data.zero_()
    second_conv.weight.data.zero_()

    for g in range(groups):
        out_start = g * out_channels_per_group
        out_end = (g + 1) * out_channels_per_group
        # conv weight has shape [out_channels, in_channels_per_group, kH, kW]
        weight_g = weight[out_start:out_end].to(device)

        if scheme == 1:
            weight_2d_g = weight_g.view(out_channels_per_group, -1)
        else:
            weight_2d_g = weight_g.permute(0, 2, 1, 3).reshape(
                out_channels_per_group * kH, in_channels_per_group * kW
            )

        U, S, V = torch.svd(weight_2d_g)
        U_trunc = U[:, :rank_per_group]
        S_trunc = S[:rank_per_group]
        V_trunc = V[:, :rank_per_group]

        if scheme == 1:
            first_weight_g = (V_trunc.t() * S_trunc.unsqueeze(1)).view(
                rank_per_group, in_channels_per_group, kH, kW
            )
            second_weight_g = U_trunc.unsqueeze(-1).unsqueeze(-1)
        else:
            first_weight_g = (V_trunc.t() * torch.sqrt(S_trunc).unsqueeze(1)).reshape(
                rank_per_group, in_channels_per_group, kW
            ).unsqueeze(2)
            second_weight_g = (U_trunc * torch.sqrt(S_trunc).unsqueeze(0)).reshape(
                out_channels_per_group, kH, rank_per_group
            ).permute(0, 2, 1).unsqueeze(-1)

        # Assign into grouped weights
        first_out_start = g * rank_per_group
        first_out_end = (g + 1) * rank_per_group
        first_conv.weight.data[first_out_start:first_out_end] = first_weight_g.to(
            first_conv.weight.device, dtype=first_conv.weight.dtype
        )
        second_conv.weight.data[out_start:out_end] = second_weight_g.to(
            second_conv.weight.device, dtype=second_conv.weight.dtype
        )

    if conv_layer.bias is not None:
        second_conv.bias.data.copy_(
            conv_layer.bias.data.to(second_conv.bias.device, dtype=second_conv.bias.dtype)
        )

    return nn.Sequential(OrderedDict([("lrd_conv1", first_conv), ("lrd_conv2", second_conv)]))


def tucker_hooi_decompose_conv_layer(conv_layer, ratio, ratio_mode, max_iter=10, tol=1e-4):
    """
    Perform Tucker-2 decomposition using HOOI (Higher-Order Orthogonal Iteration).
    
    HOOI iteratively optimizes factor matrices to minimize reconstruction error.
    This is superior to HOSVD as it finds a local minimum of the approximation error.
    
    For Tucker-2, we only compress channel dimensions:
    W[C_out, C_in, kH, kW] ≈ G[R_out, R_in, kH, kW] ×₁ U₁ ×₂ U₂
    
    Args:
        conv_layer: nn.Conv2d layer
        ratio: rank ratio for truncation
        ratio_mode: 'param' or 'rank'
        max_iter: maximum number of HOOI iterations
        tol: convergence tolerance for relative error change
        
    Returns:
        decomposed weight tensor with same shape as original
    """
    global total_rank_global, retained_rank_global, total_params_global, retained_params_global
    
    weight = conv_layer.weight.data  # [C_out, C_in, kH, kW]
    orig_shape = weight.shape
    C_out, C_in, kH, kW = orig_shape
    orig_params = weight.numel()
    
    # If ratio >= 1.0, skip decomposition
    if ratio >= 1.0:
        LOGGER.info(f"Ratio {ratio:.3f} >= 1.0, skipping Tucker decomposition for layer")
        return weight
    
    # Reshape to 3D tensor: [C_out, C_in, kH*kW]
    weight_3d = weight.view(C_out, C_in, kH * kW)
    
    # ========== Step 1: Initialize using HOSVD (truncated SVD on each mode) ==========
    
    # Mode-1 unfolding: [C_out, C_in * kH*kW]
    W1 = weight_3d.reshape(C_out, -1)
    U1_full, S1, _ = torch.svd(W1)
    
    # Mode-2 unfolding: [C_in, C_out * kH*kW]
    W2 = weight_3d.permute(1, 0, 2).reshape(C_in, -1)
    U2_full, S2, _ = torch.svd(W2)
    
    # Mode-3 (spatial): Keep at full rank for Tucker-2
    # U3 = Identity matrix (no compression)
    
    # Calculate max ranks
    max_rank_1 = min(C_out, C_in * kH * kW)
    max_rank_2 = min(C_in, C_out * kH * kW)
    max_rank_3 = kH * kW  # Full rank for spatial
    
    # Calculate target ranks based on ratio
    if ratio_mode == "param":
        target_rank_1_float = ratio * max_rank_1
        target_rank_1 = max(1, min(max_rank_1, int(target_rank_1_float)))
        target_rank_2_float = ratio * max_rank_2
        target_rank_2 = max(1, min(max_rank_2, int(target_rank_2_float)))
    elif ratio_mode == "rank":
        target_rank_1_float = ratio * max_rank_1
        target_rank_1 = max(1, min(max_rank_1, int(target_rank_1_float)))
        target_rank_2_float = ratio * max_rank_2
        target_rank_2 = max(1, min(max_rank_2, int(target_rank_2_float)))
    else:
        raise ValueError(f"Invalid ratio mode: {ratio_mode}")
    
    R1, R2, R3 = target_rank_1, target_rank_2, max_rank_3
    
    # Initialize factor matrices (truncated from HOSVD)
    U1 = U1_full[:, :R1].clone()
    U2 = U2_full[:, :R2].clone()
    # U3 = Identity (implicit, no compression on spatial dimension)
    
    LOGGER.info(f"HOOI initialization - Target ranks: [{R1}/{max_rank_1}, {R2}/{max_rank_2}, {R3}/{max_rank_3}]")
    
    # ========== Step 2: HOOI Iterations ==========
    prev_error = float('inf')
    
    for iter_idx in range(max_iter):
        # --- Update U1 (output channel mode) ---
        # Project W onto U2 (and U3=I): Y = W ×₂ U2^T
        # Y shape: [C_out, R2, kH*kW]
        Y1 = torch.einsum('oij,ir->orj', weight_3d, U2)
        
        # Mode-1 unfolding of Y: [C_out, R2 * kH*kW]
        Y1_unfold = Y1.reshape(C_out, -1)
        
        # SVD and truncate to R1
        U1_new, _, _ = torch.svd(Y1_unfold)
        U1 = U1_new[:, :R1]
        
        # --- Update U2 (input channel mode) ---
        # Project W onto U1 (and U3=I): Y = W ×₁ U1^T
        # Y shape: [R1, C_in, kH*kW]
        Y2 = torch.einsum('oij,or->rij', weight_3d, U1)
        
        # Mode-2 unfolding of Y: [C_in, R1 * kH*kW]
        Y2_unfold = Y2.permute(1, 0, 2).reshape(C_in, -1)
        
        # SVD and truncate to R2
        U2_new, _, _ = torch.svd(Y2_unfold)
        U2 = U2_new[:, :R2]
        
        # --- Compute reconstruction error for convergence check ---
        # Compute core tensor: G = W ×₁ U1^T ×₂ U2^T
        G = torch.einsum('oij,or->rij', weight_3d, U1)  # [R1, C_in, kH*kW]
        G = torch.einsum('rij,is->rsj', G, U2)           # [R1, R2, kH*kW]
        
        # Reconstruct: W_approx = G ×₁ U1 ×₂ U2
        W_approx = torch.einsum('rsj,or->osj', G, U1)    # [C_out, R2, kH*kW]
        W_approx = torch.einsum('osj,is->oij', W_approx, U2)  # [C_out, C_in, kH*kW]
        
        # Compute relative Frobenius norm error
        error = torch.norm(weight_3d - W_approx) / torch.norm(weight_3d)
        error_change = abs(prev_error - error.item())
        
        LOGGER.info(f"  HOOI iter {iter_idx + 1}/{max_iter}: "
                   f"relative error = {error.item():.6f}, "
                   f"change = {error_change:.6e}")
        
        # Check convergence
        if error_change < tol:
            LOGGER.info(f"  HOOI converged after {iter_idx + 1} iterations")
            break
        
        prev_error = error.item()
    
    # ========== Step 3: Final reconstruction ==========
    # Compute final core tensor
    G_final = torch.einsum('oij,or->rij', weight_3d, U1)
    G_final = torch.einsum('rij,is->rsj', G_final, U2)
    
    # Reconstruct weight
    weight_reconstructed_3d = torch.einsum('rsj,or->osj', G_final, U1)
    weight_reconstructed_3d = torch.einsum('osj,is->oij', weight_reconstructed_3d, U2)
    
    # Reshape back to 4D
    weight_reconstructed = weight_reconstructed_3d.view(orig_shape)
    
    # ========== Update global statistics ==========
    total_rank_global += max_rank_1 + max_rank_2 + max_rank_3
    retained_rank_global += R1 + R2 + R3
    
    # Parameter count for Tucker-2:
    # Core: R1 * R2 * kH * kW
    # U1: C_out * R1
    # U2: C_in * R2
    core_params = R1 * R2 * kH * kW
    u1_params = C_out * R1
    u2_params = C_in * R2
    decomposed_params = core_params + u1_params + u2_params
    
    total_params_global += orig_params
    retained_params_global += decomposed_params
    
    final_error = torch.norm(weight - weight_reconstructed) / torch.norm(weight)
    
    LOGGER.info(f"Tucker-2 (HOOI) - Original shape: {orig_shape}, "
               f"Ranks: [{R1}/{max_rank_1}, {R2}/{max_rank_2}, {R3}/{max_rank_3}], "
               f"Params: {decomposed_params}/{orig_params} ({decomposed_params/orig_params:.3f}), "
               f"Final relative error: {final_error:.6f}")
    
    return weight_reconstructed


def tucker_lrd_decompose_conv_layer(conv_layer, ratio, ratio_mode, scheme):
    """
    Wrapper function for Tucker decomposition to maintain compatibility.
    Now uses HOOI for better reconstruction quality.
    """
    return tucker_hooi_decompose_conv_layer(conv_layer, ratio, ratio_mode)

def apply_lrd_to_model(
    model,
    ratio,
    ratio_mode,
    device,
    scheme,
    skip_1x1=False,
    skip_if_no_mac_reduction=False,
    skip_if_rank_gt_out=False,
    whitelist_map=None,
):
    """Apply low rank decomposition to all conv layers in ResNet backbone."""
    conv_layers = get_conv_layers(model)

    LOGGER.info(f"Found {len(conv_layers)} convolutional layers")

    for parent_module, child_name, full_name, conv_layer in conv_layers:
        # Apply whitelist if provided
        if whitelist_map is not None:
            if full_name not in whitelist_map:
                continue
            wl_method, wl_ratio = whitelist_map[full_name]
            if wl_method == "orig":
                LOGGER.info(f"Skipping layer (whitelist orig): {full_name}")
                continue
            if wl_method == "svd1":
                layer_scheme = 1
            elif wl_method == "svd2":
                layer_scheme = 2
            elif wl_method == "tucker":
                layer_scheme = 3
            else:
                raise ValueError(f"Invalid method in whitelist: {wl_method}")
            layer_ratio = wl_ratio
            layer_ratio_mode = "rank"
        else:
            layer_scheme = scheme
            layer_ratio = ratio
            layer_ratio_mode = ratio_mode

        LOGGER.info(
            f"Processing layer: {full_name} (scheme={layer_scheme}, ratio={layer_ratio})"
        )

        # Move to device for computation
        original_device = conv_layer.weight.device
        original_weight = conv_layer.weight.data.clone()
        conv_layer.weight.data = original_weight.to(device)

        # Perform LRD
        if layer_scheme == 1 or layer_scheme == 2:
            # SVD-LRD
            decomposed_module = svd_lrd_decompose_conv_layer_to_module(
                conv_layer,
                layer_ratio,
                layer_ratio_mode,
                layer_scheme,
                device,
                skip_1x1=skip_1x1,
                skip_if_no_mac_reduction=skip_if_no_mac_reduction,
                skip_if_rank_gt_out=skip_if_rank_gt_out,
            )
        elif layer_scheme == 3:
            # Tucker-2 LRD (3-layer module)
            decomposed_module = tucker2_decompose_conv_layer_to_module(
                conv_layer,
                layer_ratio,
                layer_ratio_mode,
                device,
                skip_1x1=skip_1x1,
            )
        else:
            raise ValueError(f"Invalid scheme: {layer_scheme}")

        if decomposed_module is None:
            LOGGER.info(f"Skipping replacement for {full_name} (no decomposition)")
            continue
        decomposed_module = decomposed_module.to(original_device)
        _set_module_by_parent(parent_module, child_name, decomposed_module)
        if layer_scheme == 3:
            LOGGER.info(f"Replaced {full_name}: {original_weight.shape} -> 3-layer conv (Tucker-2)")
        else:
            LOGGER.info(f"Replaced {full_name}: {original_weight.shape} -> 2-layer conv")

def save_decomposed_model(model, output_dir, args, decomposition_params=None):
    """
    Save the decomposed model.
    
    Args:
        model: The decomposed model to save
        output_dir: Directory to save the model
        args: Model arguments/config
        decomposition_params: Optional dict with decomposition parameters (ratio, scheme, etc.)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Copy config
    import shutil
    config_src = os.path.join(args.savedir, 'config.json')
    config_dst = os.path.join(output_dir, 'config.json')
    if os.path.exists(config_src):
        shutil.copy2(config_src, config_dst)

    # Save a marker file to indicate this is a decomposed model
    marker_file = os.path.join(output_dir, '.decomposed_model')
    with open(marker_file, 'w') as f:
        f.write('This model has been decomposed with LRD/Tucker decomposition.\n')
        f.write('The model structure contains lrd_conv1/lrd_conv2 or tucker_conv1/tucker_core/tucker_conv3 modules.\n')
        if decomposition_params:
            import json
            f.write('\nDecomposition parameters:\n')
            f.write(json.dumps(decomposition_params, indent=2))

    # Temporarily change the model's savedir and save
    original_savedir = model._savedir
    model._savedir = output_dir
    model.save(None)  # Save with epoch=None
    #model._savedir = original_savedir  # Restore original savedir

    LOGGER.info(f"Decomposed model saved to: {output_dir}")

def main():
    """Main function."""
    logging.basicConfig(level=logging.INFO)
    cmdargs = parse_cmdargs()

    # Set output directory
    if cmdargs.output_dir is None:
        cmdargs.output_dir = cmdargs.model_dir + f"_lrd_r{int(cmdargs.ratio*100)}"
    if cmdargs.scheme == 1:
        cmdargs.output_dir += f"_s1"
    elif cmdargs.scheme == 2:
        cmdargs.output_dir += f"_s2"
    elif cmdargs.scheme == 3:
        cmdargs.output_dir += f"_tucker"
    else:
        raise ValueError(f"Invalid scheme: {cmdargs.scheme}")

    # Load model
    args, model = load_model(cmdargs.model_dir, cmdargs.epoch, cmdargs.device)

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

    if whitelist_map is not None:
        LOGGER.info(
            f"Applying LRD with whitelist: {cmdargs.whitelist_file} "
            f"(entries={len(whitelist_map)})"
        )
    else:
        LOGGER.info(f"Applying LRD with ratio: {cmdargs.ratio} and scheme: {cmdargs.scheme}")

    # Apply low rank decomposition
    apply_lrd_to_model(
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

    LOGGER.info(f"Total rank: {total_rank_global}, Retained rank: {retained_rank_global}, Rank ratio: {retained_rank_global / total_rank_global:.3f}")
    LOGGER.info(f"Total params: {total_params_global}, Retained params: {retained_params_global}, Params ratio: {retained_params_global / total_params_global:.3f}")
    
    # Save decomposed model with decomposition parameters
    decomposition_params = {
        'ratio': cmdargs.ratio,
        'ratio_mode': cmdargs.ratio_mode,
        'scheme': cmdargs.scheme,
        'skip_1x1': cmdargs.skip_1x1,
        'skip_if_no_mac_reduction': cmdargs.skip_if_no_mac_reduction,
        'skip_if_rank_gt_out': cmdargs.skip_if_rank_gt_out,
        'whitelist_file': cmdargs.whitelist_file,
    }
    save_decomposed_model(model, cmdargs.output_dir, args, decomposition_params)
    model._savedir = cmdargs.output_dir

    LOGGER.info(f"Low rank decomposition completed successfully with scheme {cmdargs.scheme}!")

    data_config_dict = args.config.data.eval
    assert isinstance(data_config_dict, dict)

    evaldir = make_eval_directory(model, cmdargs.output_dir)

    if cmdargs.data_name is not None:
        datasets = [ cmdargs.data_name ]
    else:
        datasets = list(sorted(data_config_dict.keys()))

    with torch.inference_mode():
        for name in datasets:
            eval_single_dataset(
                model, args, name, data_config_dict[name],
                cmdargs.split, cmdargs.steps, evaldir, cmdargs.batch_size,
                cmdargs.workers, cmdargs.data_path
            )

if __name__ == '__main__':
    main()
