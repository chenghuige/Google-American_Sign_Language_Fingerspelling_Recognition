#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   layers.py
#        \author   chenghuige  
#          \date   2023-07-06 21:00:01.077276
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
import tensorflow as tf
from src.tf.util import *
from src.config import *

class BatchNorm(nn.Module):
  def __init__(self, num_features, momentum=0.1, eps=1e-5):
    super().__init__()
    self.bn = nn.BatchNorm1d(num_features, momentum=momentum, eps=eps)
  
  def forward(self, x):
    x = x.permute(0, 2, 1)
    x = self.bn(x)
    x = x.permute(0, 2, 1)
    return x
  
## layers for transformer 

# Initiailizers
INIT_HE_UNIFORM = tf.keras.initializers.he_uniform
INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform
# Activations
GELU = tf.keras.activations.gelu

# based on: https://stackoverflow.com/questions/67342988/verifying-the-implementation-of-multihead-attention-in-transformer
# replaced softmax with softmax layer to support masked softmax
def scaled_dot_product(q, k, v, softmax, attention_mask):
  #calculates Q . K(transpose)
  qkt = tf.matmul(q, k, transpose_b=True)
  #caculates scaling factor
  dk = tf.math.sqrt(tf.cast(q.shape[-1], dtype=q.dtype))
  # ic(q.dtype, dk.dtype)
  scaled_qkt = qkt / dk
  softmax = softmax(scaled_qkt, mask=attention_mask)
  z = tf.matmul(softmax, v)
  #shape: (m,Tx,depth), same shape as q,k,v
  return z

class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, d_model, num_of_heads, dropout=0., d_out=None):
    super(MultiHeadAttention, self).__init__()
    self.d_model = d_model
    self.num_of_heads = num_of_heads
    self.depth = d_model // num_of_heads
    self.depth = int(self.depth * FLAGS.mhatt_depth_ratio)
    self.wq = [
        tf.keras.layers.Dense(self.depth, use_bias=False)
        for i in range(num_of_heads)
    ]
    self.wk = [
        tf.keras.layers.Dense(self.depth, use_bias=False)
        for i in range(num_of_heads)
    ]
    self.wv = [
        tf.keras.layers.Dense(self.depth, use_bias=False)
        for i in range(num_of_heads)
    ]
    self.wo = tf.keras.layers.Dense(d_model if d_out is None else d_out,
                                    use_bias=False)
    self.softmax = tf.keras.layers.Softmax()
    self.drop = tf.keras.layers.Dropout(dropout)
    self.supports_masking = True

  def call(self, q, k, v, attention_mask=None, training=False):

    multi_attn = []
    for i in range(self.num_of_heads):
      Q = self.wq[i](q)
      K = self.wk[i](k)
      V = self.wv[i](v)
      multi_attn.append(
          scaled_dot_product(Q, K, V, self.softmax, attention_mask))

    multi_head = tf.concat(multi_attn, axis=-1)
    multi_head_attention = self.wo(multi_head)
    multi_head_attention = self.drop(multi_head_attention, training=training)

    return multi_head_attention
  
class CrossAttention(tf.keras.layers.Layer):
  def __init__(self, units, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(key_dim=units, num_heads=1, **kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

  def call(self, x, context):
    shape_checker = ShapeChecker()

    shape_checker(x, 'batch t units')
    shape_checker(context, 'batch s units')

    attn_output, attn_scores = self.mha(
        query=x,
        value=context,
        return_attention_scores=True)

    shape_checker(x, 'batch t units')
    shape_checker(attn_scores, 'batch heads t s')

    # Cache the attention scores for plotting later.
    attn_scores = tf.reduce_mean(attn_scores, axis=1)
    shape_checker(attn_scores, 'batch t s')
    self.last_attention_weights = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x
  
class InstanceDropout(nn.Module):
  def __init__(self, p=0.5):
    super().__init__()
    self.p = p
    self.dropout = nn.Dropout(p)

  def forward(self, x):
    mask = torch.ones_like(x[:,:1,:1])
    mask = self.dropout(mask)
    return x * mask
  
  
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

# https://openspeech-team.github.io/openspeech/_modules/openspeech/modules/conv_base.html#BaseConv1d
class BaseConv1d(nn.Module):
    """ Base convolution module. """
    def __init__(self):
        super(BaseConv1d, self).__init__()

    def _get_sequence_lengths(self, seq_lengths):
        return (
            (seq_lengths + 2 * self.conv.padding[0]
             - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) - 1) // self.conv.stride[0] + 1
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError

# https://openspeech-team.github.io/openspeech/_modules/openspeech/modules/depthwise_conv1d.html#DepthwiseConv1d
class DepthwiseConv1d(BaseConv1d):
    r"""
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True

    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 1-D convolution.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding, 
            bias=bias,
        )

    def forward(self, inputs, input_lengths=None):
        if input_lengths is None:
            return self.conv(inputs)
        else:
            return self.conv(inputs), self._get_sequence_lengths(input_lengths)

      
class CausalDWConv1D(nn.Module):

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size=17,
               bias=False,
               **kwargs):
    super().__init__(**kwargs)
    self.kernel_size = kernel_size
    self.dw_conv = DepthwiseConv1d(in_channels,
                                   out_channels,
                                   kernel_size=kernel_size,
                                   bias=bias)
                                   
  def forward(self, x):
    dilation_rate = 1
    kernel_size = self.kernel_size
    x = F.pad(x, (dilation_rate * (kernel_size - 1), 0)) 
    x = self.dw_conv(x)
    return x

class Conv1dBlock(nn.Module):
  def __init__(self, channel_size, kernel_size, dilation_rate=1, drop_rate=0.0, expand_ratio=2, se_ratio=0.25, activation='swish', name=None):
    super().__init__()
    channel_expand = channel_size * expand_ratio
    self.expand_fc = nn.Sequential(
      nn.Linear(channel_size, channel_expand, bias=True),
      nn.SiLU())
    self.conv = CausalDWConv1D(channel_expand, channel_expand, kernel_size=kernel_size, bias=False)
    self.bn = nn.BatchNorm1d(channel_expand, momentum=0.05, eps=1e-3)
    if FLAGS.use_eca:
      self.eca = ECA()
    self.project_fc = nn.Linear(channel_expand, channel_size, bias=True)
    if not FLAGS.inst_drop:
      self.drop = nn.Dropout(drop_rate)
    else:
      self.drop = InstanceDropout(drop_rate)
    
  def forward(self, x):
    skip = x
    x = self.expand_fc(x)
    x = x.permute(0, 2, 1)
    x = self.conv(x)
    x = self.bn(x)
    if FLAGS.use_eca:
      x = self.eca(x)
    x = x.permute(0, 2, 1)
    x = self.project_fc(x)
    x = self.drop(x)
    x = x * FLAGS.skip_factor + skip
    return x
  
class Conv1DBlocks(nn.Module):
  def __init__(self, channel_size, ksize_vals=[11,11,11]):
    super(Conv1DBlocks, self).__init__()
    conv1d_blocks = []
    for ksize in ksize_vals:
      conv1d_blocks.append(Conv1dBlock(channel_size, ksize, drop_rate=0.2))
    self.conv1d_blocks = nn.Sequential(*conv1d_blocks)
    
  def forward(self, x):
    x = self.conv1d_blocks(x)
    return x

class MultiHeadSelfAttention(nn.Module):
  def __init__(self, input_dim, dim=256, num_heads=4, dropout=0, **kwargs):
    super().__init__(**kwargs)
    self.dim = dim
    self.scale = self.dim**-0.5
    self.num_heads = num_heads
    self.qkv = nn.Linear(input_dim, 3 * dim, bias=False)
    self.drop = nn.Dropout(dropout)
    self.softmax = nn.Softmax(dim=-1)
    self.proj = nn.Linear(dim, dim, bias=False)
    
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
                dim=256,
                num_heads=4,
                expand=4,
                attn_dropout=0.2,
                drop_rate=0.2):
    super().__init__()
    
    self.bn = BatchNorm(input_dim, momentum=0.05, eps=1e-3)
    self.mhsa = MultiHeadSelfAttention(input_dim, dim=dim, num_heads=num_heads, dropout=attn_dropout)
    self.bn2 = BatchNorm(input_dim, momentum=0.05, eps=1e-3)
    self.fc = nn.Sequential(
                  nn.Linear(dim, dim * expand, bias=False),
                  nn.SiLU(),
                  nn.Linear(dim * expand, dim, bias=False),
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
        ## NOTICE modify here so can dived by 2...
        # self.padding = max(0, self.kernel_size - self.stride) 
        ##  # like k=5, stride=2 here padding is 2 which make 320 -> 160 -> 80
        self.padding = (self.kernel_size + 1) // self.stride - 1

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
            
class TimeReduction2Module(nn.Module):
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
        ## NOTICE here is as original NEMO version if you want to fall back just --time_reduce_method == 'conv2'
        # like k=5, stride=2 here padding is 3 which make 320 -> 161 -> 81 
        self.padding = max(0, self.kernel_size - self.stride) 
        # self.padding = (self.kernel_size + 1) // self.stride - 1

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

class AvgPoolingModule(nn.Module):
  def __init__(self, pool_size, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.pooling = nn.AvgPool1d(pool_size)
    
  def forward(self, x):
    x = x.permute(0, 2, 1)
    x = self.pooling(x)
    x = x.permute(0, 2, 1)
    return x
  
class MaxPoolingModule(nn.Module):
  def __init__(self, pool_size, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.pooling = nn.MaxPool1d(pool_size)
    
  def forward(self, x):
    x = x.permute(0, 2, 1)
    x = self.pooling(x)
    x = x.permute(0, 2, 1)
    return x
  
# from espnet
class PositionalEncoding(torch.nn.Module):
    """Positional encoding.

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
        reverse (bool): Whether to reverse the input position. Only for
        the class LegacyRelPositionalEncoding. We remove it in the current
        class RelPositionalEncoding.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, reverse=False):
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.reverse = reverse
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        self._register_load_state_dict_pre_hook(_pre_hook)

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)
        else:
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)
      
class Conv1dSubsampling2(torch.nn.Module):
    """Convolutional 1D subsampling (to 1/2 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv1dSubsampling2 object."""
        super(Conv1dSubsampling2, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim, odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask=None):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = x.transpose(2, 1)  # (#batch, idim, time)
        x = self.conv(x)
        b, c, t = x.size()
        x = self.out(x.transpose(1, 2).contiguous())
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:1][:, :, :-2:2]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]

class Conv2dSubsampling2(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/2 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling2 object."""
        super(Conv2dSubsampling2, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 1),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 2)), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask=None):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:1]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]
      