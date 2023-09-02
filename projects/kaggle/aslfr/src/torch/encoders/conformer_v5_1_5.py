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
# add time reduction
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import *
import melt as mt
from src.config import *
from src.torch.embedding import *
from src.torch.layers import InstanceDropout

import torch
from torch import nn, einsum
import torch.nn.functional as F

# simplfiy from NEMO
class TimeReductionModule(nn.Module):
    """
    Squeezeformer Time Reduction procedure. Downsamples the audio by `stride` in the time dimension.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        out_dim (int): Output dimension of the module.
        kernel_size (int): Conv kernel size for depthwise convolution in convolution module
        stride (int): Downsampling factor in time dimension.
    """

    def __init__(self, d_model: int, out_dim: int, kernel_size: int = 5, stride: int = 2):
        super().__init__()

        self.d_model = d_model
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = max(0, self.kernel_size - self.stride)

        self.dw_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            groups=d_model,
        )

        self.pw_conv = nn.Conv1d(
            in_channels=d_model, out_channels=out_dim, kernel_size=1, stride=1, padding=0, groups=1,
        )

        self.reset_parameters()

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.dw_conv(x)
        x = self.pw_conv(x)

        x = x.transpose(1, 2)  # [B, T, C]
        return x

    def reset_parameters(self):
        dw_max = self.kernel_size ** -0.5
        pw_max = self.d_model ** -0.5

        with torch.no_grad():
            torch.nn.init.uniform_(self.dw_conv.weight, -dw_max, dw_max)
            torch.nn.init.uniform_(self.dw_conv.bias, -dw_max, dw_max)
            torch.nn.init.uniform_(self.pw_conv.weight, -pw_max, pw_max)
            torch.nn.init.uniform_(self.pw_conv.bias, -pw_max, pw_max)


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


class Attention(nn.Module):

  def __init__(self, dim, heads=8, dim_head=64, 
               dropout=0., max_pos_emb=512, 
               relpos_att=False):
    super().__init__()
    inner_dim = dim_head * heads
    self.heads = heads
    self.scale = dim_head**-0.5
    self.qkv = nn.Linear(dim, inner_dim * 3, bias=False)
    self.to_out = nn.Linear(inner_dim, dim)

    self.max_pos_emb = max_pos_emb
    self.relpos_att = relpos_att
    if self.relpos_att:
      self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

    self.dropout = nn.Dropout(dropout)

  def forward(
      self,
      x,
  ):
    n, device, h, max_pos_emb = x.shape[
        -2], x.device, self.heads, self.max_pos_emb

    q, k, v = self.qkv(x).chunk(3, dim=-1)
    q = q.view(q.shape[0], q.shape[1], h, -1).permute(0, 2, 1, 3)
    k = k.view(k.shape[0], k.shape[1], h, -1).permute(0, 2, 1, 3)
    v = v.view(v.shape[0], v.shape[1], h, -1).permute(0, 2, 1, 3)
        
    dots = torch.matmul(q, k.permute(0, 1, 3, 2)) * self.scale
    
    # ## 这里相对位置编码的重要性 影响很大...
    ##  keras to tflite 转换einsum似乎有问题。。。 带来极大的不一致性 因此手动改写ensim操作
    # # shaw's relative positional embedding 
    if self.relpos_att:
      seq = torch.arange(n, device=device)
      dist = seq.unsqueeze(-1) - seq.unsqueeze(0)
      # dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
      dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
      rel_pos_emb = self.rel_pos_emb(dist).to(q)
      if FLAGS.allow_einsum:
        pos_attn = einsum('b h n d, n r d -> b h n r', q, rel_pos_emb) * self.scale
      else:
        pos_attn = relpos_att(q, rel_pos_emb) * self.scale 
      dots = dots + pos_attn

    # TODO F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype) FIX for awp nan of fp16?
    # attn = dots.softmax(dim=-1)
    attn = F.softmax(dots, dim=-1, dtype=torch.float32).to(q.dtype)

    out = torch.matmul(attn, v)
    out = out.permute(0, 2, 1, 3)
    out = out.reshape(out.shape[0], out.shape[1], -1)
    out = self.to_out(out)
    return self.dropout(out)


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
               relpos_att=False):
    super().__init__()
    self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
    self.attn = Attention(dim=dim,
                          dim_head=dim_head,
                          heads=heads,
                          dropout=attn_dropout,
                          relpos_att=relpos_att,
                          max_pos_emb=FLAGS.n_frames)
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
    self.dropout = InstanceDropout(0.2)
    

  def forward(self, x):
    # instance dropout有效降低过拟合, skip_factor也有1-2个点提升
    x = self.dropout(self.ff1(x)) * FLAGS.skip_factor + x
    x = self.dropout(self.attn(x)) * FLAGS.skip_factor + x
    x = self.dropout(self.conv(x)) * FLAGS.skip_factor + x
    x = self.dropout(self.ff2(x)) * FLAGS.skip_factor + x
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
    for i in range(depth):
      # here put relpos att to last layers better then first layers
      if i <= depth - (FLAGS.relpos_att_layers + 1):
        relpos_att = False
        dim_head_ = dim_head
      else:
        relpos_att = True 
        dim_head_ = dim_head // 2
        
      self.layers.append(ConformerBlock(dim=dim,
                          dim_head=dim_head_,
                          heads=heads,
                          ff_mult=ff_mult,
                          conv_expansion_factor=conv_expansion_factor,
                          conv_kernel_size=conv_kernel_size,
                          conv_causal=conv_causal,
                          relpos_att=relpos_att))
    if FLAGS.time_reduce:
      ## TODO 这里再实验一些 也许效果不好是kernel_size参数的问题?
      # self.layers.insert(FLAGS.time_reduce_idx, TimeReductionModule(dim, dim, kernel_size=15, stride=2))
      self.layers.insert(FLAGS.time_reduce_idx, TimeReductionModule(dim, dim, kernel_size=3, stride=2))
      
  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x


class Encoder(nn.Module):

  def __init__(self):
    super(Encoder, self).__init__()
    self.embedding = get_embeddding() if FLAGS.embedding else SimpleEmbedding()
    self.encoder = Conformer(
        dim=FLAGS.encoder_units,
        depth=FLAGS.encoder_layers,
        dim_head=FLAGS.mhatt_dimhead,
        heads=FLAGS.mhatt_heads,
        ff_mult=4,
        conv_expansion_factor=FLAGS.conv1d_expansion_factor,  # 2
        conv_kernel_size=FLAGS.conv1d_ksize_vals[0],
        attn_dropout=0.2,
        ff_dropout=0.2,
        conv_dropout=0.2)

  def forward(self, x_inp):
    x = self.embedding(x_inp)
    x = self.encoder(x)
    return x
