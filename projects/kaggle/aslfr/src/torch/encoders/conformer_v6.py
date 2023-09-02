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
# this one attention part using openspeech 
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
  
## layers for conv1d_transformer asl1 1st place solution
#Copied from previous comp 1st place model: https://www.kaggle.com/code/hoyso48/1st-place-solution-training
class ECA(nn.Module):

  def __init__(self, kernel_size=5, **kwargs):
    super().__init__(**kwargs)
    self.kernel_size = kernel_size
    self.conv = nn.Conv1d(1,
                          1,
                          kernel_size=kernel_size,
                          stride=1,
                          padding='same',
                          bias=False)
    self.act = nn.Sigmoid()

  def forward(self, inputs):
    x = torch.mean(inputs, dim=-1)
    x = x.unsqueeze(1)
    x = self.conv(x)
    x = x.squeeze(1)
    x = self.act(x)
    x = x.unsqueeze(-1)
    return inputs * x

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
  

# attention code from openspeech
import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    r"""
    Positional Encoding proposed in "Attention Is All You Need".
    Since transformer contains no recurrence and no convolution, in order for the model to make
    use of the order of the sequence, we must add some positional information.

    "Attention Is All You Need" use sine and cosine functions of different frequencies:
        PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
        PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))
    """

    def __init__(self, d_model: int = 512, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, length: int) -> Tensor:
        return self.pe[:, :length]


class RelPositionalEncoding(nn.Module):
    """
    Relative positional encoding module
    Args:
        d_model: Embedding dimension.
        max_len: Maximum input length.
    """

    def __init__(self, d_model: int = 512, max_len: int = 512) -> None:
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.pe = None
        # self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : Input tensor B X T X C
        Returns:
            torch.Tensor: Encoded tensor B X T X C
        """
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return pos_emb

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class Linear(nn.Module):
    r"""
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)


class RelativeMultiHeadAttention(nn.Module):
    r"""
    Multi-head attention with relative positional encoding.
    This concept was proposed in the "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

    Args:
        dim (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.
    """

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 16,
        dropout_p: float = 0.1,
    ) -> None:
        super(RelativeMultiHeadAttention, self).__init__()
        assert dim % num_heads == 0, "d_model % num_heads should be zero."

        self.dim = dim
        self.d_head = int(dim / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(self.d_head)

        self.query_proj = Linear(dim, dim)
        self.key_proj = Linear(dim, dim)
        self.value_proj = Linear(dim, dim)
        self.pos_proj = Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(p=dropout_p)
        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)

        self.out_proj = Linear(dim, dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        pos_embedding: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = value.size(0)

        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = self.key_proj(key).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        value = self.value_proj(value).view(batch_size, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        pos_embedding = self.pos_proj(pos_embedding).view(batch_size, -1, self.num_heads, self.d_head)

        content_score = torch.matmul((query + self.u_bias).transpose(1, 2), key.transpose(2, 3))
        pos_score = torch.matmul((query + self.v_bias).transpose(1, 2), pos_embedding.permute(0, 2, 3, 1))
        pos_score = self._relative_shift(pos_score)

        score = (content_score + pos_score) / self.sqrt_dim

        if mask is not None:
            mask = mask.unsqueeze(1)
            score.masked_fill_(mask, -1e4)

        attn = F.softmax(score, -1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.dim)

        return self.out_proj(context)

    def _relative_shift(self, pos_score: Tensor) -> Tensor:
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        # zeros = pos_score.new_zeros(batch_size, num_heads, seq_length1, 1)
        zeros = torch.zeros(batch_size, num_heads, seq_length1, 1, dtype=pos_score.dtype, device=pos_score.device)
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)

        padded_pos_score = padded_pos_score.view(batch_size, num_heads, seq_length2 + 1, seq_length1)
        # pos_score = padded_pos_score[:, :, 1:].view_as(pos_score)[:, :, :, : seq_length2 // 2 + 1]
        pos_score = padded_pos_score[:, :, 1:].view(pos_score.size())[:, :, :, : seq_length2 // 2 + 1]

        return pos_score


from typing import Optional

import torch.nn as nn
from torch import Tensor

class MultiHeadedSelfAttentionModule(nn.Module):
    r"""
    Conformer employ multi-headed self-attention (MHSA) while integrating an important technique from Transformer-XL,
    the relative sinusoidal positional encoding scheme. The relative positional encoding allows the self-attention
    module to generalize better on different input length and the resulting encoders is more robust to the variance of
    the utterance length. Conformer use prenorm residual units with dropout which helps training
    and regularizing deeper models.

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_p: float = 0.1,
    ) -> None:
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, inputs: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        r"""
        Forward propagate of conformer's multi-headed self attention module.

        Inputs: inputs, mask
            - **inputs** (batch, time, dim): Tensor containing input vector
            - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

        Returns:
            - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
        """
        batch_size = inputs.size(0)
        pos_embedding = self.positional_encoding(inputs)
        pos_embedding = pos_embedding.repeat(batch_size, 1, 1)

        inputs = self.layer_norm(inputs)
        outputs = self.attention(inputs, inputs, inputs, pos_embedding=pos_embedding, mask=mask)

        return self.dropout(outputs)


class Attention(nn.Module):

  def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
    super().__init__()
    inner_dim = dim_head * heads
    self.qkv = nn.Linear(dim, inner_dim, bias=False)
    self.fc = nn.Linear(inner_dim, dim)
    self.att = MultiHeadedSelfAttentionModule(inner_dim, heads, dropout)
    
  def forward(
      self,
      x,
  ):
    x = self.qkv(x)       
    x = self.att(x)
    x = self.fc(x)
    return x


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
        # ECA(),
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

    self.attn = PreNorm(dim, self.attn)
    self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
    self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))

    self.post_norm = nn.LayerNorm(dim)
    self.dropout = InstanceDropout(0.2)
    self.skip_factors = nn.ParameterList([
      nn.Parameter(torch.tensor(FLAGS.skip_factor)),
      nn.Parameter(torch.tensor(FLAGS.skip_factor)),
      nn.Parameter(torch.tensor(FLAGS.skip_factor)),
      nn.Parameter(torch.tensor(FLAGS.skip_factor))])

  def forward(self, x):
    # instance dropout有效降低过拟合, skip_factor也有3个点提升
    x = self.dropout(self.ff1(x)) * self.skip_factors[0] + x
    x = self.dropout(self.attn(x)) * self.skip_factors[1] + x
    x = self.dropout(self.conv(x)) * self.skip_factors[2] + x
    x = self.dropout(self.ff2(x)) * self.skip_factors[3] + x
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
