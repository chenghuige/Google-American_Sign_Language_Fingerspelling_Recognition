#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model2.py
#        \author   chenghuige
#          \date   2023-07-13 00:12:33.561837
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import *
import melt as mt
from src.config import *
from src.torch.embedding import *
from src.torch.layers import Conv1DBlocks, InstanceDropout

class MultiHeadSelfAttention(nn.Module):
  def __init__(self, input_dim, num_heads=4, dim_head=64, dropout=0, **kwargs):
    super().__init__(**kwargs)
    self.dim = dim_head * num_heads
    # TODO or self.scale = dim_head**-0.5
    self.scale = self.dim**-0.5
    self.num_heads = num_heads
    self.qkv = nn.Linear(input_dim, 3 * self.dim, bias=False)
    self.drop = nn.Dropout(dropout)
    self.softmax = nn.Softmax(dim=-1)
    self.proj = nn.Linear(self.dim, input_dim, bias=False)
    
  def forward(self, inputs):
    qkv = self.qkv(inputs)
    qkv = qkv.view(inputs.shape[0], inputs.shape[1], self.num_heads, self.dim * 3 // self.num_heads)
    qkv = qkv.permute(0, 2, 1, 3)
    q, k, v = torch.split(qkv, [self.dim // self.num_heads] * 3, dim=-1)

    attn = torch.matmul(q, k.permute(0, 1, 3, 2)) * self.scale

    attn = self.softmax(attn)
    attn = self.drop(attn)

    x = attn @ v
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(x.shape[0], x.shape[1], self.dim)
    x = self.proj(x)
    return x
  
class TransformerBlock(nn.Module):
  def __init__(self, 
                input_dim,
                dim_head=64,
                num_heads=4,
                expand=4,
                attn_dropout=0.2,
                drop_rate=0.2):
    super().__init__()
    
    self.bn = BatchNorm(input_dim, momentum=0.05, eps=1e-3)
    self.mhsa = MultiHeadSelfAttention(input_dim, dim_head=dim_head, num_heads=num_heads, dropout=attn_dropout)
    self.bn2 = BatchNorm(input_dim, momentum=0.05, eps=1e-3)
    self.fc = nn.Sequential(
                  nn.Linear(input_dim, input_dim * expand, bias=False),
                  nn.SiLU(),
                  nn.Linear(input_dim * expand, input_dim, bias=False),
                  )
    if not FLAGS.inst_drop:
      self.drop = nn.Dropout(drop_rate)
    else:
      self.drop = InstanceDropout(drop_rate)
  
  def forward(self, inputs):
    x = inputs
    x = self.bn(x)
    x = self.mhsa(x)
    x = self.drop(x)
    x = x * FLAGS.skip_factor + inputs
    attn_out = x
    x = self.bn2(x)
    x = self.fc(x)
    x = self.drop(x)
    x = x * FLAGS.skip_factor + attn_out
    return x

class Encoder(nn.Module):

  def __init__(self):
    super(Encoder, self).__init__()
    self.embedding = get_embeddding() if FLAGS.embedding else SimpleEmbedding()
    self.encoder = nn.Sequential(*[
      nn.Sequential(
        Conv1DBlocks(FLAGS.encoder_units, ksize_vals=FLAGS.conv1d_ksize_vals),
        TransformerBlock(FLAGS.encoder_units, 
                         dim_head=FLAGS.mhatt_dimhead,
                         num_heads=FLAGS.mhatt_heads, 
                         expand=FLAGS.conv1d_expansion_factor),
      ) for _ in range(FLAGS.encoder_layers)
    ])
    gezi.set('torch2tf', True)
    
  def forward(self, x_inp):
    x = self.embedding(x_inp)
    x = self.encoder(x)
    return x
  