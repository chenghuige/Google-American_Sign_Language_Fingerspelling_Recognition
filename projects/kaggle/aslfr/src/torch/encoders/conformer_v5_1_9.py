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
from src.torch.layers import TimeReductionModule

import torch
from torch import nn, einsum
import torch.nn.functional as F

# helper functions

def calc_same_padding(kernel_size):
  pad = kernel_size // 2
  return (pad, pad - (kernel_size + 1) % 2)

# einsum('b h n d, n r d -> b h n r', x, y)
# @torch.jit.script
def relpos_att(x, y):
  b, h, n, d = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
  r = y.shape[1]
  
  x = x.permute(2, 0, 1, 3).reshape(n, -1, d)
  x = torch.matmul(x, y.permute(0,2,1)).view(n, b, h, r).permute(1,2,0,3)
  return x
# helper classes

class Swish(nn.Module):

  def forward(self, x):
    return x * x.sigmoid()

# GLU and ECA which is better ? TODO 
class GLU(nn.Module):

  def __init__(self, dim):
    super().__init__()
    self.dim = dim

  def forward(self, x):
    out, gate = x.chunk(2, dim=self.dim)
    return out * gate.sigmoid()


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
      if self.rope:
        cos, sin = self.rotary_emb(v, seq_len=n)
        position_ids = torch.arange(0, n, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, n)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        sim = torch.matmul(q, k.transpose(2, 3)) * self.scale
      else:
        # shaw's relpos att which is slow and not work better then rope so just use rope is fine
        dots = torch.matmul(q, k.permute(0, 1, 3, 2)) * self.scale
        max_pos_emb = self.max_position_embeddings
        seq = torch.arange(n, device=device)
        dist = seq.unsqueeze(-1) - seq.unsqueeze(0)
        # dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        if FLAGS.allow_einsum:
          pos_attn = einsum('b h n d, n r d -> b h n r', q, rel_pos_emb) * self.scale
        else:
          pos_attn = relpos_att(q, rel_pos_emb) * self.scale 
        sim = dots + pos_attn
    else:
      sim = torch.matmul(q, k.permute(0, 1, 3, 2)) * self.scale
    
    # seems causal not good
    if FLAGS.causal_mask:
      i, j = sim.shape[-2:]
      causal_mask = torch.ones((i, j), device = x.device, dtype = torch.bool).triu(j - i + 1)
      sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

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

class FeedForward(nn.Module):

  def __init__(self, dim, mult=4, dropout=0.):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(dim, dim * mult), 
                             Swish(),
                             nn.Dropout(dropout), 
                             nn.Linear(dim * mult, dim),
                             nn.Dropout(dropout))

  def forward(self, x):
    return self.net(x)


class SwapChannels(nn.Module):

  def __int__(self):
    super().__init__()

  def forward(self, x):
    return x.permute(0, 2, 1)


class ConformerConvModule(nn.Module):

  def __init__(self,
               dim,
               causal=False,
               expansion_factor=2,
               kernel_size=31,
               dropout=0.):
    super().__init__()

    inner_dim = dim * expansion_factor
    padding = calc_same_padding(kernel_size) if not causal else (kernel_size -
                                                                 1, 0)

    self.net = nn.Sequential(
        nn.LayerNorm(dim),
        SwapChannels(),
        # Rearrange('b n c -> b c n'),
        nn.Conv1d(dim, inner_dim * 2, 1),
        GLU(dim=1),
        DepthWiseConv1d(inner_dim,
                        inner_dim,
                        kernel_size=kernel_size,
                        padding=padding),
        nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
        Swish(),
        nn.Conv1d(inner_dim, dim, 1),
        # Rearrange('b c n -> b n c'),
        SwapChannels(),
        nn.Dropout(dropout))

  def forward(self, x):
    return self.net(x)


# Conformer Block


class ConformerBlock(nn.Module):

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
    self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
    self.attn = Attention(dim=dim,
                          dim_head=dim_head,
                          heads=heads,
                          dropout=attn_dropout,
                          max_pos_emb=FLAGS.n_frames,
                          relpos_att=relpos_att,
                          rope=rope)
    self.conv = ConformerConvModule(dim=dim,
                                    causal=conv_causal,
                                    expansion_factor=conv_expansion_factor,
                                    kernel_size=conv_kernel_size,
                                    dropout=conv_dropout)
    self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

    self.attn = PreNorm(dim, self.attn)
    self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
    self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

    self.post_norm = nn.LayerNorm(dim)
    self.inst_drop = inst_drop if inst_drop is not None else FLAGS.inst_drop_rate
    # 0.2
    self.dropout = InstanceDropout(self.inst_drop)
    self.skip_factor = skip_factor if skip_factor is not None else FLAGS.skip_factor

  def forward(self, x):
    # instance dropout有效降低过拟合, skip_factor也有1-2个点提升
    # TODO 注意上面 self.ff1 已经scale 0.5 所以这里相当于整体 *0.25
    x = self.dropout(self.ff1(x)) * self.skip_factor + x
    x = self.dropout(self.attn(x)) * self.skip_factor + x
    x = self.dropout(self.conv(x)) * self.skip_factor + x
    x = self.dropout(self.ff2(x)) * self.skip_factor + x
    x = self.post_norm(x)
    return x


# Conformer


class Conformer(nn.Module):

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

    relpos_att = True
    rope = True
    # if like 6, 100 then whill have heads 4, 8
    # elif like 6, 10 then have heads 4, 8, 16
    # elif like 6, 6 then only heads 4
    # or like reduce_idx == 100 then [100, 100] will not reduce any more
    reduce_idxes = FLAGS.reduce_idxes or [FLAGS.reduce_idx, FLAGS.reduce_idx]
    assert len(reduce_idxes) == 2
    ic(reduce_idxes)
    for i in range(depth):    
      if i in reduce_idxes:
        self.layers.append(TimeReductionModule(dim, dim, kernel_size=FLAGS.time_kernel_size, stride=FLAGS.time_stride))
        
      if i < reduce_idxes[0]:
        dim_head = 32
        heads = 4
      elif i < reduce_idxes[1]:
        dim_head = 32
        heads = 8
      else:
        if reduce_idxes[1] > reduce_idxes[0]:
          dim_head = 32
          heads = 16
        
      self.layers.append(ConformerBlock(dim=dim,
                          dim_head=dim_head,
                          heads=heads,
                          ff_mult=ff_mult,
                          conv_expansion_factor=conv_expansion_factor,
                          conv_kernel_size=conv_kernel_size,
                          conv_causal=conv_causal,
                          attn_dropout=attn_dropout,
                          ff_dropout=ff_dropout,
                          conv_dropout=conv_dropout,
                          relpos_att=relpos_att,
                          rope=rope,
                          inst_drop=self.inst_drops[i]))
    
  
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
    self.embedding = get_embeddding() if FLAGS.embedding else SimpleEmbedding()
    if FLAGS.global_drop is None:
      attn_dropout, ff_dropout, conv_dropout = FLAGS.attn_drop, FLAGS.ff_drop, FLAGS.conv_drop
    else:
      attn_dropout, ff_dropout, conv_dropout = FLAGS.global_drop, FLAGS.global_drop, FLAGS.global_drop
    self.encoder = Conformer(
        dim=FLAGS.encoder_units,
        depth=FLAGS.encoder_layers,
        dim_head=FLAGS.mhatt_dimhead,
        heads=FLAGS.mhatt_heads,
        ff_mult=4,
        conv_expansion_factor=FLAGS.conv1d_expansion_factor,  # 2
        conv_kernel_size=FLAGS.conv1d_ksize_vals[0],
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
        conv_dropout=conv_dropout)

  def forward(self, x_inp):
    x = self.embedding(x_inp)
    x = self.encoder(x)
    return x
