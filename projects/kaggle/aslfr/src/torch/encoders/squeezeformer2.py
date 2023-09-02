#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   conformer.py
#        \author   chenghuige
#          \date   2023-08-05 09:02:16.428718
#   \Description modified from 
#   name = 'conformer',
#   packages = find_packages(),
#   version = '0.3.2',
#   license='MIT',
#   description = 'The convolutional module from the Conformer paper',
#   author = 'Phil Wang',
#   author_email = 'lucidrains@gmail.com',
#   url = 'https://github.com/lucidrains/conformer',
# try rope embedding
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import *
import melt as mt
from src.config import *
from src.torch.embedding import *
from src.torch.layers import InstanceDropout
from src import util
from src.torch.layers import TimeReductionModule, TimeReduction2Module, AvgPoolingModule, MaxPoolingModule

import torch
from torch import nn, einsum
import torch.nn.functional as F
from typing import Union


class Swish(nn.SiLU):
    """
    Swish activation function introduced in 'https://arxiv.org/abs/1710.05941'
    Mathematically identical to SiLU. See note in nn.SiLU for references.
    """
    
    
class CausalConv1D(nn.Conv1d):
    """
    A causal version of nn.Conv1d where each step would have limited access to locations on its right or left
    All arguments are the same as nn.Conv1d except padding.

    If padding is set None, then paddings are set automatically to make it a causal convolution where each location would not see any steps on its right.

    If padding is set as a list (size of 2), then padding[0] would be used as left padding and padding[1] as right padding.
    It would make it possible to control the number of steps to be accessible on the right and left.
    This mode is not supported when stride > 1. padding[0]+padding[1] should be equal to (kernel_size - 1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
    ) -> None:
        self.cache_drop_size = None
        if padding is None:
            self._left_padding = kernel_size - 1
            self._right_padding = stride - 1
        else:
            if stride != 1 and padding != kernel_size - 1:
                raise ValueError("No striding allowed for non-symmetric convolutions!")
            if isinstance(padding, int):
                self._left_padding = padding
                self._right_padding = padding
            elif isinstance(padding, list) and len(padding) == 2 and padding[0] + padding[1] == kernel_size - 1:
                self._left_padding = padding[0]
                self._right_padding = padding[1]
            else:
                raise ValueError(f"Invalid padding param: {padding}!")

        self._max_cache_len = self._left_padding

        super(CausalConv1D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )

    def update_cache(self, x, cache=None):
        if cache is None:
            new_x = F.pad(x, pad=(self._left_padding, self._right_padding))
            next_cache = cache
        else:
            new_x = F.pad(x, pad=(0, self._right_padding))
            new_x = torch.cat([cache, new_x], dim=-1)
            if self.cache_drop_size > 0:
                next_cache = new_x[:, :, : -self.cache_drop_size]
            else:
                next_cache = new_x
            next_cache = next_cache[:, :, -cache.size(-1) :]
        return new_x, next_cache

    def forward(self, x, cache=None):
        x, cache = self.update_cache(x, cache=cache)
        x = super().forward(x)
        if cache is None:
            return x
        else:
            return x, cache

class ConformerFeedForward(nn.Module):
    """
    feed-forward module of Conformer model.
    """

    def __init__(self, d_model, d_ff, dropout, activation=Swish()):
        super(ConformerFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

    def reset_parameters_ff(self):
        ffn1_max = self.d_model ** -0.5
        ffn2_max = self.d_ff ** -0.5
        with torch.no_grad():
            nn.init.uniform_(self.linear1.weight, -ffn1_max, ffn1_max)
            nn.init.uniform_(self.linear1.bias, -ffn1_max, ffn1_max)
            nn.init.uniform_(self.linear2.weight, -ffn2_max, ffn2_max)
            nn.init.uniform_(self.linear2.bias, -ffn2_max, ffn2_max)

activation_registry = {
    "identity": nn.Identity,
    "hardtanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}

class ConformerConvolution(nn.Module):
    """The convolution module for the Conformer model.
    Args:
        d_model (int): hidden dimension
        kernel_size (int): kernel size for depthwise convolution
        pointwise_activation (str): name of the activation function to be used for the pointwise conv.
            Note that Conformer uses a special key `glu_` which is treated as the original default from
            the paper.
    """

    def __init__(
        self, d_model, kernel_size, norm_type='batch_norm', conv_context_size=None, pointwise_activation='glu_'
    ):
        super(ConformerConvolution, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.norm_type = norm_type

        if conv_context_size is None:
            conv_context_size = (kernel_size - 1) // 2

        if pointwise_activation in activation_registry:
            self.pointwise_activation = activation_registry[pointwise_activation]()
            dw_conv_input_dim = d_model * 2

            if hasattr(self.pointwise_activation, 'inplace'):
                self.pointwise_activation.inplace = True
        else:
            self.pointwise_activation = pointwise_activation
            dw_conv_input_dim = d_model

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model, out_channels=d_model * 2, kernel_size=1, stride=1, padding=0, bias=True
        )

        self.depthwise_conv = CausalConv1D(
            in_channels=dw_conv_input_dim,
            out_channels=dw_conv_input_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=conv_context_size,
            groups=dw_conv_input_dim,
            bias=True,
        )

        if norm_type == 'batch_norm':
            self.batch_norm = nn.BatchNorm1d(dw_conv_input_dim)
        elif norm_type == 'instance_norm':
            self.batch_norm = nn.InstanceNorm1d(dw_conv_input_dim)
        elif norm_type == 'layer_norm':
            self.batch_norm = nn.LayerNorm(dw_conv_input_dim)
        elif norm_type.startswith('group_norm'):
            num_groups = int(norm_type.replace("group_norm", ""))
            self.batch_norm = nn.GroupNorm(num_groups=num_groups, num_channels=d_model)
        else:
            raise ValueError(f"conv_norm_type={norm_type} is not valid!")

        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=dw_conv_input_dim, out_channels=d_model, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, x, pad_mask=None, cache=None):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)

        # Compute the activation function or use GLU for original Conformer
        if self.pointwise_activation == 'glu_':
            x = nn.functional.glu(x, dim=1)
        else:
            x = self.pointwise_activation(x)

        if pad_mask is not None:
            x = x.float().masked_fill(pad_mask.unsqueeze(1), 0.0)

        x = self.depthwise_conv(x, cache=cache)
        if cache is not None:
            x, cache = x

        if self.norm_type == "layer_norm":
            x = x.transpose(1, 2)
            x = self.batch_norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.batch_norm(x)

        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        if cache is None:
            return x
        else:
            return x, cache

    def reset_parameters_conv(self):
        pw1_max = pw2_max = self.d_model ** -0.5
        dw_max = self.kernel_size ** -0.5

        with torch.no_grad():
            nn.init.uniform_(self.pointwise_conv1.weight, -pw1_max, pw1_max)
            nn.init.uniform_(self.pointwise_conv1.bias, -pw1_max, pw1_max)
            nn.init.uniform_(self.pointwise_conv2.weight, -pw2_max, pw2_max)
            nn.init.uniform_(self.pointwise_conv2.bias, -pw2_max, pw2_max)
            nn.init.uniform_(self.depthwise_conv.weight, -dw_max, dw_max)
            nn.init.uniform_(self.depthwise_conv.bias, -dw_max, dw_max)
            
class ScaleBiasLayer(nn.Module):
    """
    Computes an affine transformation y = x * scale + bias, either learned via adaptive weights, or fixed.
    Efficient alternative to LayerNorm where we can avoid computing the mean and variance of the input, and
    just rescale the output of the previous layer.

    Args:
        d_model (int): input dimension of layer.
        adaptive_scale (bool): whether to learn the affine transformation parameters or not. If set to False,
            the scale is fixed to 1 and bias to 0, effectively performing a No-Op on the input.
            This is done for export compatibility.
    """

    def __init__(self, d_model: int, adaptive_scale: bool):
        super().__init__()
        self.adaptive_scale = adaptive_scale
        if adaptive_scale:
            self.scale = nn.Parameter(torch.ones(d_model))
            self.bias = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_buffer('scale', torch.ones(d_model), persistent=True)
            self.register_buffer('bias', torch.zeros(d_model), persistent=True)

    def forward(self, x):
        scale = self.scale.view(1, 1, -1)
        bias = self.bias.view(1, 1, -1)
        return x * scale + bias

class DepthWiseConv1d(nn.Module):

  def __init__(self, chan_in, chan_out, kernel_size, padding):
    super().__init__()
    self.padding = padding
    self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

  def forward(self, x):
    x = F.pad(x, self.padding)
    return self.conv(x)
  
# attention, feedforward, and conv module


class Scale(nn.Module):

  def __init__(self, scale, fn):
    super().__init__()
    self.fn = fn
    self.scale = scale

  def forward(self, x, **kwargs):
    return self.fn(x, **kwargs) * self.scale


class PreNorm(nn.Module):

  def __init__(self, dim, fn):
    super().__init__()
    self.fn = fn
    self.norm = nn.LayerNorm(dim)

  def forward(self, x, **kwargs):
    x = self.norm(x)
    return self.fn(x, **kwargs)

#https://github.com/lucidrains/perceiver-ar-pytorch/blob/685d77d152c55ef7210336566b952de7da631f68/perceiver_ar_pytorch/perceiver_ar_pytorch.py#L276
from einops import rearrange, repeat

# rotary positional embedding
# https://arxiv.org/abs/2104.09864

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    # return torch.cat((-x2, x1), dim=-1)
    # return torch.cat((torch.neg(x2), x1), dim=-1)
    return torch.cat((x2 * (-1.), x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# attention

class Attention(nn.Module):
  def __init__(self, dim, dim_head=64, 
                heads=8, dropout=0., max_pos_emb=512, 
                relpos_att=True, rope=False):
    super().__init__()
    self.scale = dim_head ** -0.5
    self.heads = heads
    inner_dim = heads * dim_head

    # self.norm = nn.LayerNorm(dim)
    self.dropout = nn.Dropout(dropout)
    self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
    self.to_out = nn.Linear(inner_dim, dim, bias = False)
    self.max_position_embeddings = max_pos_emb
    self.dim_head = dim_head
    self.relpos_att = relpos_att
    self.rope = rope
    if relpos_att:
      if rope:
        self._init_rope()
      else:
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)
        # ic(self.rel_pos_emb)

  def _init_rope(self):
    scaling_type = FLAGS.scaling_type
    if scaling_type is None:
        self.rotary_emb = LlamaRotaryEmbedding(self.dim_head, max_position_embeddings=self.max_position_embeddings)
    else:
        scaling_factor = FLAGS.scaling_factor
        if scaling_type == "linear":
            self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                self.dim_head, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
            )
        elif scaling_type == "dynamic":
            self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                self.dim_head, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
      
    # ic(self.rotary_emb)

  def forward(self, x):
    # x = self.norm(x)
    n, device, h = x.shape[-2], x.device, self.heads

    q, k, v = self.to_qkv(x).chunk(3, dim = -1)
    
    q = q.view(q.shape[0], q.shape[1], h, -1).permute(0, 2, 1, 3)
    k = k.view(k.shape[0], k.shape[1], h, -1).permute(0, 2, 1, 3)
    v = v.view(v.shape[0], v.shape[1], h, -1).permute(0, 2, 1, 3)

    if self.relpos_att:
      cos, sin = self.rotary_emb(v, seq_len=n)
      position_ids = torch.arange(0, n, dtype=torch.long, device=device)
      position_ids = position_ids.unsqueeze(0).view(-1, n)
      q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
      sim = torch.matmul(q, k.transpose(2, 3)) * self.scale
    else:
      sim = torch.matmul(q, k.permute(0, 1, 3, 2)) * self.scale
    
    attn = F.softmax(sim, dim=-1, dtype=torch.float32).to(q.dtype)
    # attn = sim.softmax(dim = -1)
    attn = self.dropout(attn)
    
    out = torch.matmul(attn, v)
    # out = einsum('b h i j, b h j d -> b h i d', attn, v)    
    out = out.permute(0, 2, 1, 3)
    out = out.reshape(out.shape[0], out.shape[1], -1)
    # out = rearrange(out, 'b h n d -> b n (h d)')
    out = self.to_out(out)
    out = self.dropout(out)
    return out

class SqueezeformerBlock(nn.Module):

  def __init__(self,
               *,
               dim,
               dim_head=64,
               heads=8,
               ff_mult=4,
               conv_expansion_factor=2,
               conv_kernel_size=31,
               attn_dropout=0.,
               ff_dropout=0.,
               conv_dropout=0.,
               conv_causal=False,
               relpos_att=True,
               rope=False,
               inst_drop=None,
               skip_factor=None):
    super().__init__()
    # first feed forward module
    self.norm_feed_forward1 = nn.LayerNorm(dim)
    self.feed_forward1 = ConformerFeedForward(d_model=dim, d_ff=dim * ff_mult, dropout=ff_dropout)
    self.feed_forward1_scale = ScaleBiasLayer(d_model=dim, adaptive_scale=True)
    
    # convolution module
    self.norm_conv = nn.LayerNorm(dim)
    self.conv = ConformerConvolution(
        d_model=dim, kernel_size=conv_kernel_size, norm_type='batch_norm', pointwise_activation='swish'
    )
    self.conv_scale = ScaleBiasLayer(d_model=dim, adaptive_scale=True)
    
    # multi-headed self-attention module
    self.norm_self_att = nn.LayerNorm(dim)
    self.self_attn = Attention(dim=dim,
                          dim_head=dim_head,
                          heads=heads,
                          dropout=attn_dropout,
                          max_pos_emb=FLAGS.n_frames,
                          relpos_att=relpos_att,
                          rope=rope)
    self.self_attn_scale = ScaleBiasLayer(d_model=dim, adaptive_scale=True)

    # second feed forward module
    self.norm_feed_forward2 = nn.LayerNorm(dim)
    self.feed_forward2 = ConformerFeedForward(d_model=dim, d_ff=dim * ff_mult, dropout=ff_dropout)
    self.feed_forward2_scale = ScaleBiasLayer(d_model=dim, adaptive_scale=True)
    
    self.inst_drop = inst_drop if inst_drop is not None else FLAGS.inst_drop_rate
    # 0.2
    self.dropout = InstanceDropout(self.inst_drop)
    self.skip_factor = skip_factor if skip_factor is not None else FLAGS.skip_factor
    
    self.fc_factor = 0.5
    
    self.reset_parameters()
    
  def reset_parameters(self):
    # Used for Squeezeformer initialization only
    self.feed_forward1.reset_parameters_ff()
    self.feed_forward2.reset_parameters_ff()
    self.conv.reset_parameters_conv()

  def forward(self, x):
    residual = x
    x = self.self_attn_scale(x)
    x = self.self_attn(x)
    x = residual + self.dropout(x) * self.skip_factor
    
    x = self.norm_self_att(x)
    residual = x
    x = self.feed_forward1_scale(x)
    x = self.feed_forward1(x)
    x = residual + self.dropout(x) * self.skip_factor * self.fc_factor
    x = self.norm_feed_forward1(x)
    residual = x

    x = self.conv_scale(x)
    x = self.conv(x)
    x = residual + self.dropout(x) * self.skip_factor
    x = self.norm_conv(x)
    residual = x
    
    x = self.feed_forward2_scale(x)
    x = self.feed_forward2(x)
    x = residual + self.dropout(x) * self.skip_factor * self.fc_factor
    x = self.norm_feed_forward2(x)

    return x


# Conformer


class Squeezeformer(nn.Module):

  def __init__(self,
               dim,
               *,
               depth,
               dim_head=64,
               heads=8,
               ff_mult=4,
               conv_expansion_factor=2,
               conv_kernel_size=31,
               attn_dropout=0.,
               ff_dropout=0.,
               conv_dropout=0.,
               conv_causal=False):
    super().__init__()
    self.dim = dim
    self.layers = nn.ModuleList([])
    self.inst_drops = [None] * depth
    if FLAGS.dynamic_inst_drop:
      self.inst_drops = util.compute_stochastic_depth_drop_probs(depth, FLAGS.inst_drop_rate)
    heads_ = heads
    conv_kernel_size_ = conv_kernel_size
    for i in range(depth):    
      if FLAGS.relpos_combine_mode == 0:
        relpos_att = True
        dim_head_ = dim_head
        if i < depth - FLAGS.nonrope_layers:
          rope = True
        else:
          rope = False
          # // 2 如果是dimhead是64 如果是32则保持不变 
          if dim_head > 32:
            dim_head_ = dim_head // FLAGS.relpos_att_stride
      elif FLAGS.relpos_combine_mode == 1:
        rope = False
        relpos_att = False
        dim_head_ = dim_head
        if i >= depth - FLAGS.relpos_att_layers:
          relpos_att = True
          # // 2 如果是dimhead是64 如果是32则保持不变 
          if dim_head > 32:
            dim_head_ = dim_head // FLAGS.relpos_att_stride
      elif FLAGS.relpos_combine_mode == 2:
        assert FLAGS.time_reduce
        relpos_att = True
        dim_head_ = dim_head
        rope = True
        heads_ = heads
        if i < FLAGS.time_reduce_idx:
          if heads > 4:
            heads_ = heads // 2
        else:
          conv_kernel_size_ = conv_kernel_size // 2
      elif FLAGS.relpos_combine_mode == 3:
        assert FLAGS.time_reduce
        relpos_att = True
        dim_head_ = dim_head
        conv_kernel_size_ = conv_kernel_size
        rope = True
        if i < FLAGS.time_reduce_idx:
          if dim_head > 32:
            dim_head_ = dim_head // 2
        else:
          conv_kernel_size_ = conv_kernel_size // 2
      else:
        raise ValueError(f'not support relpos_combine_mode {FLAGS.relpos_combine_mode}')
      
      self.layers.append(SqueezeformerBlock(dim=dim,
                          dim_head=dim_head_,
                          heads=heads_,
                          ff_mult=ff_mult,
                          conv_expansion_factor=conv_expansion_factor,
                          conv_kernel_size=conv_kernel_size_,
                          conv_causal=conv_causal,
                          attn_dropout=attn_dropout,
                          ff_dropout=ff_dropout,
                          conv_dropout=conv_dropout,
                          relpos_att=relpos_att,
                          rope=rope,
                          inst_drop=self.inst_drops[i]))
      
    if FLAGS.time_reduce:
      if FLAGS.time_reduce_idx < depth - 1:
        if FLAGS.time_reduce_method == 'conv':
          reduction_module = TimeReductionModule(dim, dim, kernel_size=FLAGS.time_kernel_size, stride=FLAGS.time_stride) 
        elif FLAGS.time_reduce_method == 'conv2':
          reduction_module = TimeReduction2Module(dim, dim, kernel_size=FLAGS.time_kernel_size, stride=FLAGS.time_stride) 
        elif FLAGS.time_reduce_method == 'avg':
          reduction_module = AvgPoolingModule(FLAGS.time_stride)
        elif FLAGS.time_reduce_method == 'max':
          reduction_module = MaxPoolingModule(FLAGS.time_stride)
        else:
          raise ValueError(f'not support time_reduce_method {FLAGS.time_reduce_method}')
        
        if FLAGS.share_reduce:
          gezi.set('time_reduce_module', reduction_module)
        self.layers.insert(FLAGS.time_reduce_idx, reduction_module)
      
  def forward(self, x):
    for i, layer in enumerate(self.layers):
      x = layer(x)
      if FLAGS.inter_ctc:
        if self.training and i == FLAGS.time_reduce_idx:
          gezi.set('inter_ctc_out', x)
    return x



class Encoder(nn.Module):

  def __init__(self):
    super(Encoder, self).__init__()
    if FLAGS.trans_emb:
      self.embedding = get_embeddding() if FLAGS.embedding else SimpleEmbedding()
    if FLAGS.global_drop is None:
      attn_dropout, ff_dropout, conv_dropout = FLAGS.attn_drop, FLAGS.ff_drop, FLAGS.conv_drop
    else:
      attn_dropout, ff_dropout, conv_dropout = FLAGS.global_drop, FLAGS.global_drop, FLAGS.global_drop
    self.encoder = Squeezeformer(
        dim=FLAGS.encoder_units,
        depth=FLAGS.encoder_layers,
        dim_head=FLAGS.mhatt_dimhead,
        heads=FLAGS.mhatt_heads,
        ff_mult=FLAGS.ff_mult,
        conv_expansion_factor=FLAGS.conv1d_expansion_factor,  # 2
        conv_kernel_size=FLAGS.conv1d_ksize_vals[0],
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
        conv_dropout=conv_dropout)

  def forward(self, x):
    if FLAGS.trans_emb:
      x = self.embedding(x)
    x = self.encoder(x)
    return x
