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
# this one combine with nemo relpos attention implementation
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

try:
  from nemo.collections.asr.parts.submodules.multi_head_attention import (
      LocalAttRelPositionalEncoding,
      MultiHeadAttention,
      PositionalEncoding,
      RelPositionalEncoding,
      RelPositionMultiHeadAttention,
      RelPositionMultiHeadAttentionLongformer,
  )
except Exception:
  pass

# helper functions

def calc_same_padding(kernel_size):
  pad = kernel_size // 2
  return (pad, pad - (kernel_size + 1) % 2)

# einsum('b h n d, n r d -> b h n r', x, y)
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

  def __init__(self, dim, heads=8, dim_head=64, dropout=0., max_pos_emb=512):
    super().__init__()
    inner_dim = dim_head * heads
    self.heads = heads
    self.scale = dim_head**-0.5
    self.qkv = nn.Linear(dim, inner_dim * 3, bias=False)
    self.to_out = nn.Linear(inner_dim, dim)
    self.attn = RelPositionMultiHeadAttention(
                n_head=heads,
                n_feat=inner_dim,
                dropout_rate=0.1,
                pos_bias_u=None,
                pos_bias_v=None,
            )

  def forward(
      self,
      x,
      pos_emb
  ):
    q, k, v = self.qkv(x).chunk(3, dim=-1)
    out = self.attn(q, k, v, None, pos_emb)
    out = self.to_out(out)
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
               conv_causal=False):
    super().__init__()
    self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
    self.pos_enc = RelPositionalEncoding(
                d_model=dim_head * heads,
                dropout_rate=0.1,
                max_len=512,
                xscale=dim**0.5,
                dropout_rate_emb=0.,
            )
    self.norm = nn.LayerNorm(dim)
    self.attn = Attention(dim=dim,
                          dim_head=dim_head,
                          heads=heads,
                          dropout=attn_dropout)
    self.conv = ConformerConvModule(dim=dim,
                                    causal=conv_causal,
                                    expansion_factor=conv_expansion_factor,
                                    kernel_size=conv_kernel_size,
                                    dropout=conv_dropout)
    self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

    # self.attn = PreNorm(dim, self.attn)
    self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
    self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

    self.post_norm = nn.LayerNorm(dim)
    self.dropout = InstanceDropout(0.2)
    

  def forward(self, x):
    # instance dropout有效降低过拟合, skip_factor也有1-2个点提升
    x = self.dropout(self.ff1(x)) * FLAGS.skip_factor + x
    self.pos_enc.extend_pe(FLAGS.n_frames, x.device)
    x_, pos_emb = self.pos_enc(x)
    x = self.dropout(self.norm(self.attn(x_, pos_emb))) * FLAGS.skip_factor + x
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
    self.encoder = nn.Sequential(*[
      nn.Sequential(
        ConformerBlock(dim=dim,
                      dim_head=dim_head,
                      heads=heads,
                      ff_mult=ff_mult,
                      conv_expansion_factor=conv_expansion_factor,
                      conv_kernel_size=conv_kernel_size,
                      conv_causal=conv_causal)) for _ in range(depth)])

  def forward(self, x):
    x = self.encoder(x)
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
