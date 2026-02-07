"""
Custom MultiheadAttention module that uses nn.Linear for qkv_fuse and out_proj,
and torch's scaled_dot_product_attention for attention computation.

This design makes it easier to:
- Use linear hooks to log inputs/outputs
- Implement fake quantization using nn.Linear
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from latency_quant_utils import Linear8bit, Quantizer8bit
import torch.cuda.nvtx as nvtx


class CustomMultiheadAttention(nn.Module):
    """
    Custom MultiheadAttention that uses nn.Linear layers for qkv_fuse and out_proj,
    and torch.nn.functional.scaled_dot_product_attention for attention computation.
    
    This is a drop-in replacement for nn.MultiheadAttention with the same interface,
    but uses standard nn.Linear layers that can be easily hooked and quantized.
    
    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of parallel attention heads
        dropout: Dropout probability on attention weights (default: 0.0)
        bias: If specified, adds bias to input/output projection layers (default: True)
        add_bias_kv: If specified, adds bias to the key and value sequences (default: False)
        add_zero_attn: If specified, adds a batch of zeros to the key and value sequences (default: False)
        kdim: Total number of features for keys (default: None, uses embed_dim)
        vdim: Total number of features for values (default: None, uses embed_dim)
        batch_first: If True, then the input and output tensors are provided as (batch, seq, feature) (default: False)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn
        self.batch_first = batch_first
        
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # QKV fused projection using nn.Linear
        # This projects input to (Q, K, V) all at once: [batch, seq, embed_dim] -> [batch, seq, 3*embed_dim]
        # self.qkv_fuse = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Output projection using nn.Linear
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Optional bias for key and value
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.zeros((1, 1, embed_dim)))
            self.bias_v = nn.Parameter(torch.zeros((1, 1, embed_dim)))
        else:
            self.register_parameter('bias_k', None)
            self.register_parameter('bias_v', None)
        
        # self._reset_parameters()
        self.quant_forward = False
    
    def _reset_parameters(self):
        """Initialize parameters similar to nn.MultiheadAttention"""
        # Use xavier_uniform for qkv_fuse
        # nn.init.xavier_uniform_(self.qkv_fuse.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0.)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0.)
        
        # Use xavier_uniform for out_proj
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)
        
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)
    
    def init_quant(self):
        self.q_proj = Linear8bit.from_float(self.q_proj)
        self.k_proj = Linear8bit.from_float(self.k_proj)
        self.v_proj = Linear8bit.from_float(self.v_proj)
        self.out_proj = Linear8bit.from_float(self.out_proj)
        self.quantizer = Quantizer8bit()
        self.quant_forward = True
    
    def from_nn_mha(self, nn_mha: nn.MultiheadAttention):
        """
        Copy weights and biases from an nn.MultiheadAttention module to this CustomMultiheadAttention.
        
        Args:
            nn_mha: The nn.MultiheadAttention module to copy weights from
        
        Returns:
            self (for method chaining)
        
        Raises:
            ValueError: If the dimensions don't match
        """
        # Copy qkv_fuse weights and bias (clone to avoid sharing tensors)
        if nn_mha.in_proj_weight is not None:
            # fuse_input, but model have different v input so deprecate this
            # self.qkv_fuse.weight.data = nn_mha.in_proj_weight.data.clone()
            fuse_weight = torch.chunk(nn_mha.in_proj_weight.data.clone(), 3, dim=0)
            self.q_proj.weight.data = fuse_weight[0]
            self.k_proj.weight.data = fuse_weight[1]
            self.v_proj.weight.data = fuse_weight[2]
            del fuse_weight

        
        if nn_mha.in_proj_bias is not None:
            # if self.qkv_fuse.bias is not None:
            if self.q_proj.bias is not None:
                # self.qkv_fuse.bias.data = nn_mha.in_proj_bias.data.clone()
                fuse_bias = torch.chunk(nn_mha.in_proj_bias.data.clone(), 3, dim=0)
                self.q_proj.bias.data = fuse_bias[0]
                self.k_proj.bias.data = fuse_bias[1]
                self.v_proj.bias.data = fuse_bias[2]
                del fuse_bias
        elif self.q_proj.bias is not None:
            self.q_proj.bias.data.zero_()
            self.k_proj.bias.data.zero_()
            self.v_proj.bias.data.zero_()
        
        # Copy out_proj weights and bias (don't replace the module, just copy the data)
        if hasattr(nn_mha, 'out_proj') and isinstance(nn_mha.out_proj, nn.Module):
            self.out_proj.weight.data = nn_mha.out_proj.weight.data.clone()
            if nn_mha.out_proj.bias is not None:
                if self.out_proj.bias is not None:
                    self.out_proj.bias.data = nn_mha.out_proj.bias.data.clone()
            elif self.out_proj.bias is not None:
                self.out_proj.bias.data.zero_()
        
        # Copy bias_k and bias_v if they exist
        if nn_mha.bias_k is not None and self.bias_k is not None:
            self.bias_k.data = nn_mha.bias_k.data.clone()
        if nn_mha.bias_v is not None and self.bias_v is not None:
            self.bias_v.data = nn_mha.bias_v.data.clone()
        
        
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the custom multihead attention.
        
        Args:
            query: Query tensor of shape (N, L, E) if batch_first=True, else (L, N, E)
            key: Key tensor of shape (N, S, E) if batch_first=True, else (S, N, E). If None, uses query.
            value: Value tensor of shape (N, S, E) if batch_first=True, else (S, N, E). If None, uses query.
            key_padding_mask: Mask to exclude keys that are pads, of shape (N, S)
            need_weights: If True, returns attention weights (default: True)
            attn_mask: 2D or 3D mask to prevent attention to certain positions
            average_attn_weights: If True, returns averaged attention weights (default: True)
        
        Returns:
            attn_output: Output tensor of shape (N, L, E) if batch_first=True, else (L, N, E)
            attn_output_weights: Attention weights if need_weights=True, else None
        """
        
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # Project Q, K, V using fused linear layer
        # Split qkv_fuse weight into Q, K, V parts (similar to nn.MultiheadAttention's in_proj_weight)
        # This works for both self-attention (query=key=value) and cross-attention
        # q = self.qkv_fuse(query)
        # q, k, v = torch.chunk(q, 3, dim=-1)
        if self.quant_forward:
            query = self.quantizer(query)
            value = self.quantizer(value)
        nvtx.range_push('mha_forward_q_proj')
        # torch.cuda.synchronize()
        q = self.q_proj(query)
        # torch.cuda.synchronize()
        nvtx.range_pop()
        k = self.k_proj(query)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention: [batch, seq, num_heads, head_dim]
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        
        # Transpose to [batch, num_heads, seq, head_dim] for scaled_dot_product_attention
        q = q.transpose(1, 2)  # [batch, num_heads, seq_q, head_dim]
        k = k.transpose(1, 2)  # [batch, num_heads, seq_k, head_dim]
        v = v.transpose(1, 2)  # [batch, num_heads, seq_k, head_dim]
        
        # Use torch's scaled_dot_product_attention
        # This function expects (batch, num_heads, seq, head_dim) format
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,  # Set to True if you want causal masking
        )
        
        # attn_output: [batch, num_heads, seq_q, head_dim]
        # Reshape back: [batch, seq_q, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2)
        # Flatten: [batch, seq_q, embed_dim]
        attn_output = attn_output.contiguous().view(batch_size, seq_len_q, self.embed_dim)
        
        # Apply output projection
        if self.quant_forward:
            attn_output = self.quantizer(attn_output)
        attn_output = self.out_proj(attn_output)
        
        # Handle batch_first format for output
        if not self.batch_first:
            # Convert from (N, L, E) back to (L, N, E)
            attn_output = attn_output.transpose(0, 1)
        
        # Compute attention weights if needed
        attn_output_weights = None
        if need_weights:
            # Compute attention weights manually for compatibility
            # This is approximate since scaled_dot_product_attention doesn't return weights by default
            # We compute them separately
            with torch.no_grad():
                # Compute attention scores
                scale = 1.0 / (self.head_dim ** 0.5)
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                
                # Apply masks (use the combined mask that was used in scaled_dot_product_attention)
                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        # Boolean mask: True means keep, False means mask out
                        attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))
                    else:
                        # Float mask: add to scores
                        attn_scores = attn_scores + attn_mask
                
                # Apply softmax
                attn_weights = F.softmax(attn_scores, dim=-1)
                
                # Apply dropout
                if self.dropout > 0.0 and self.training:
                    attn_weights = F.dropout(attn_weights, p=self.dropout, training=True)
                
                # Average over heads if requested
                if average_attn_weights:
                    attn_weights = attn_weights.mean(dim=1)  # [batch, seq_q, seq_k]
                else:
                    attn_weights = attn_weights  # [batch, num_heads, seq_q, seq_k]
                
                attn_output_weights = attn_weights
        
        return attn_output, attn_output_weights
    



def replace_mha_with_custom_mha(model: nn.Module):
    """
    Replace nn.MultiheadAttention modules with CustomMultiheadAttention
    while preserving weights.
    """
    to_replace = []

    # 1) Collect first (do NOT mutate while iterating)
    for name, m in model.named_modules():
        if isinstance(m, nn.MultiheadAttention):
            to_replace.append((name, m))

    # 2) Replace
    for full_name, mha in to_replace:
        # Split path
        *parent_path, attr_name = full_name.split(".")

        # Resolve parent module
        parent = model
        for p in parent_path:
            parent = getattr(parent, p)

        # Create custom MHA and copy weights
        custom_mha = CustomMultiheadAttention(
            embed_dim=mha.embed_dim,
            num_heads=mha.num_heads,
            dropout=mha.dropout,
            batch_first=getattr(mha, "batch_first", False),
        )
        custom_mha.from_nn_mha(mha)

        # Replace
        setattr(parent, attr_name, custom_mha)
