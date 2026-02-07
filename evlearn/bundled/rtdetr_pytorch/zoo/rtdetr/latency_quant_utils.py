import torch
import torch.nn as nn
import torch.nn.functional as F

import quarot
# import gemm_int8

class PackedQuantizedTensor:
    def __init__(self, 
                 quantized_x: torch.Tensor, 
                 scales_x: torch.Tensor):
        self.quantized_x = quantized_x
        self.scales_x = scales_x

    def size(self):
        return self.quantized_x.size()
    
    @property
    def device(self):
        return self.quantized_x.device
    
    @property
    def dtype(self):
        return self.quantized_x.dtype


def sym_dequant_pytorch(q: torch.Tensor,
                        scale_row: torch.Tensor,
                        scale_col: torch.Tensor,
                        bits: int = 32):
    """
    Pure PyTorch symmetric dequant for int32 GEMM accumulators.

    q:          int32 tensor of shape [..., C]  (last dim = columns/out_features)
    scale_row:  fp16 tensor of shape [prod(q.shape[:-1])]  (per flattened row)
    scale_col:  fp16 tensor of shape [C]  (per column/out_feature)

    Returns: fp16 tensor with same shape as q.
    """
    assert q.dtype == torch.int32
    assert scale_row.dtype == torch.float16
    assert scale_col.dtype == torch.float16

    # flatten to [R, C]
    q_shape = q.shape
    q2 = q.reshape(-1, q_shape[-1])  # [R, C]
    R, C = q2.shape

    scale_row = scale_row.reshape(-1)
    scale_col = scale_col.reshape(-1)
    assert scale_row.numel() == R, f"scale_row {scale_row.numel()} != rows {R}"
    assert scale_col.numel() == C, f"scale_col {scale_col.numel()} != cols {C}"

    # Fast variant: do multiplications in fp16 (may be less numerically safe than fp32)
    out = q2.to(torch.float16)
    out = out * scale_col.unsqueeze(0)          # broadcast over rows
    out = out * scale_row.unsqueeze(1)          # broadcast over cols

    return out.view(*q_shape).contiguous()


class Quantizer8bit(torch.nn.Module):
    def __init__(self, input_clip_ratio=1.0):
        super().__init__()
        self.input_clip_ratio = input_clip_ratio
    
    def forward(self, x):
        scales_x = (torch.max(torch.abs(x), dim=-1)[0].unsqueeze(-1)/128).to(torch.float16) * self.input_clip_ratio
        quantized_x = torch.clamp(x/scales_x, -128, 127).to(torch.int8)
        packed_tensor = PackedQuantizedTensor(quantized_x, scales_x)
        return packed_tensor


class Linear8bit(nn.Module):
    def __init__(self, in_features, out_features, bias=False, dtype=torch.float16):
        '''
        Symmetric 8-bit Linear Layer.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer('weight_scales',
                             torch.zeros((self.out_features, 1), requires_grad=False))
        self.register_buffer('weight', (torch.randint(-8, 7, (self.out_features, self.in_features),
                                                             # SubByte weight
                                                             dtype=torch.int8, requires_grad=False)))
        if bias:                                                        
            self.register_buffer('bias', torch.zeros((self.out_features), dtype=dtype))
        else:
            self.bias = None
        
    def forward(self, x):
        #if torch.cuda.current_device() != x.device:
        #    torch.cuda.set_device(x.device)
        
        
        assert type(x) == PackedQuantizedTensor #Quantized input is given
        x, scales_x = x.quantized_x, x.scales_x
        #shape_handler = ShapeHandler(quantized_x)
        #quantized_x = shape_handler.flatten(quantized_x)
        # x = quarot.matmul(x, self.weight)
        #breakpoint()
        B, M, K_in = x.shape
        N_out = self.weight.shape[0] # 等同于 self.out_features
        
        # 将输入激活 reshape 为 2D: (B*M, K_in)
        x = x.reshape(-1, K_in).contiguous()
        x = torch.matmul(x, self.weight, alpha=1).to(torch.int32)
        x = x.reshape(B, M, N_out)
        #out = shape_handler.unflatten(
        #    quarot.sym_dequant(int_result, scales_x, self.weight_scales))
        if self.bias is not None:
            return quarot.sym_dequant(x, scales_x, self.weight_scales) + self.bias
        else:
            #breakpoint()
            return quarot.sym_dequant(x, scales_x, self.weight_scales)
        # if self.bias is not None:
        #     return sym_dequant_pytorch(x, scales_x, self.weight_scales) + self.bias
        # else:
        #     #breakpoint()
        #     return sym_dequant_pytorch(x, scales_x, self.weight_scales)

    @staticmethod
    def from_float(module: torch.nn.Linear, weight_scales=None,):
        '''
        Generate a new Linear8bit module from a FP16 Linear module.
        The weight matrix should have the same shape as the weight matrix of the FP16 Linear module and rounded using torch.round()
        routine. We will convert it to subByte representation and save it in the int_weight buffer.
        '''
        weight_matrix = module.weight.data
        
        
        int_module = Linear8bit(module.in_features, module.out_features, bias=module.bias is not None, dtype=weight_matrix.dtype).to(weight_matrix.dtype)
        if weight_scales is not None:
            assert weight_scales.shape == (module.out_features, 1), 'weight_scales should have shape (out_features, 1)'
            weight_matrix = weight_matrix.cuda()
            int_module.weight_scales.copy_(weight_scales.to(weight_matrix.dtype))
            int_rounded_weight = (weight_matrix/weight_scales.cuda()).round()
            # int_module.weight.copy_(quarot.functional.pack_i4(int_rounded_weight.to(torch.int8)).cpu())
            int_module.weight.copy_(int_rounded_weight.to(torch.int8).cpu())
        
            if module.bias is not None:
                int_module.bias.copy_(module.bias)
        else:
            weight_scales = (torch.max(torch.abs(weight_matrix), dim=-1)[0].unsqueeze(-1)/7).to(torch.float16).cuda()
            assert weight_scales.shape == (module.out_features, 1), 'weight_scales should have shape (out_features, 1)'
            weight_matrix = weight_matrix.cuda()
            int_module.weight_scales.copy_(weight_scales.to(weight_matrix.dtype))
            int_rounded_weight = (weight_matrix/weight_scales.cuda()).round()
            # int_module.weight.copy_(quarot.functional.pack_i4(int_rounded_weight.to(torch.int8)).cpu())
            int_module.weight.copy_(int_rounded_weight.to(torch.int8).cpu())
        
            if module.bias is not None:
                int_module.bias.copy_(module.bias)
        return int_module
