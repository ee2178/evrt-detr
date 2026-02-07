'''by lyuwenyu
'''

import torch 
import torch.nn as nn
from collections import OrderedDict

def compute_lrd_rank(
    ch_in,
    ch_out,
    kernel_size,
    ratio,
    ratio_mode,
    scheme,
):
    groups = 1
    kH, kW = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

    if ch_out % groups != 0 or ch_in % groups != 0:
        raise ValueError("Channels must be divisible by groups")

    C_out_g = ch_out // groups
    C_in_g  = ch_in  // groups

    if scheme == 1:
        max_rank = min(C_out_g, C_in_g * kH * kW)
        orig_params = C_out_g * C_in_g * kH * kW

        if ratio_mode == "param":
            rank_float = ratio * orig_params / (C_out_g + C_in_g * kH * kW)
        elif ratio_mode == "rank":
            rank_float = ratio * max_rank
        else:
            raise ValueError(f"Invalid ratio_mode: {ratio_mode}")

    elif scheme == 2:
        max_rank = min(C_out_g * kW, C_in_g * kH)
        orig_params = C_out_g * C_in_g * kH * kW

        if ratio_mode == "param":
            rank_float = ratio * orig_params / (C_out_g * kW + C_in_g * kH)
        elif ratio_mode == "rank":
            rank_float = ratio * max_rank
        else:
            raise ValueError(f"Invalid ratio_mode: {ratio_mode}")

    else:
        raise ValueError(f"Unsupported LRD scheme: {scheme}")

    rank = max(1, min(max_rank, int(rank_float)))
    return rank



class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None, lrd = False, act_bits = None):
        super().__init__()
        self.act_bits = act_bits

        if lrd == False:
            self.conv = nn.Conv2d(
                ch_in,
                ch_out,
                kernel_size,
                stride,
                padding=(kernel_size - 1) // 2 if padding is None else padding,
                bias=bias,
            )


        else:
            scheme = lrd["scheme"]
            ratio = lrd["ratio"]
            ratio_mode = lrd["ratio_mode"]

            rank = compute_lrd_rank(
                ch_in=ch_in,
                ch_out=ch_out,
                kernel_size=kernel_size,
                ratio=ratio,
                ratio_mode=ratio_mode,
                scheme=scheme
            )


            kH, kW = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
            sH, sW = stride if isinstance(stride, tuple) else (stride, stride)
            pH, pW = (
                ((kH - 1) // 2, (kW - 1) // 2)
                if padding is None
                else (padding, padding) if isinstance(padding, int) else padding
            )

            if scheme == 1:
                # -------------------------
                # Scheme 1: full kernel → 1x1
                # -------------------------
                self.conv = nn.Sequential(OrderedDict([
                    ('lrd_conv1', nn.Conv2d(
                        ch_in,
                        rank,
                        kernel_size=(kH, kW),
                        stride=(sH, sW),
                        padding=(pH, pW),
                        bias=False,
                    )),
                    ('lrd_conv2', nn.Conv2d(
                        rank,
                        ch_out,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=bias,
                    )),
                ])
                )

            elif scheme == 2:
                # -------------------------
                # Scheme 2: (1×kW) → (kH×1)
                # -------------------------
                self.conv = nn.Sequential(OrderedDict([
                    ('lrd_conv1', nn.Conv2d(
                        ch_in,
                        rank,
                        kernel_size=(1, kW),
                        stride=(1, sW),
                        padding=(0, pW),
                        bias=False,
                    )),
                    ('lrd_conv2', nn.Conv2d(
                        rank,
                        ch_out,
                        kernel_size=(kH, 1),
                        stride=(sH, 1),
                        padding=(pH, 0),
                        bias=bias,
                    )),
                ])
                )

            else:
                raise ValueError(f"Unsupported LRD scheme: {scheme}")
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if self.act_bits:
            y = self.quant(self.conv(x), num_bits = self.act_bits)
        else:
            y = self.conv(x)
        return self.act(self.norm(y))

    def quant(self, tensor, num_bits = 4, alpha = 0.95):
        """
        1. Clip to [0.9 * min, 0.9 * max]
        2. Symmetric uniform quantization
        """
        min_val = tensor.min()
        max_val = tensor.max()

        clip_min = alpha * min_val
        clip_max = alpha * max_val

        clipped = torch.clamp(tensor, clip_min, clip_max)

        qmin = -(2 ** (num_bits - 1))
        qmax = (2 ** (num_bits - 1)) - 1

        max_abs = clipped.abs().max()
        if max_abs == 0:
            scale = 1.0
        else:
            scale = max_abs / qmax

        qt = torch.clamp((clipped / scale).round(), qmin, qmax)
        dequant = qt * scale
        return dequant
        


class FrozenBatchNorm2d(nn.Module):
    """copy and modified from https://github.com/facebookresearch/detr/blob/master/models/backbone.py
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """
    def __init__(self, num_features, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        n = num_features
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps
        self.num_features = n 

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def extra_repr(self):
        return (
            "{num_features}, eps={eps}".format(**self.__dict__)
        )


def get_activation(act: str, inpace: bool=True):
    '''get activation
    '''
    act = act.lower()
    
    if act == 'silu':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()
    
    elif act == 'gelu':
        m = nn.GELU()
        
    elif act is None:
        m = nn.Identity()
    
    elif isinstance(act, nn.Module):
        m = act

    else:
        raise RuntimeError('')  

    if hasattr(m, 'inplace'):
        m.inplace = inpace
    
    return m 
