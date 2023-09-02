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
  
  
## layers for conv1d_transformer asl1 1st place solution
#Copied from previous comp 1st place model: https://www.kaggle.com/code/hoyso48/1st-place-solution-training
class ECA(tf.keras.layers.Layer):

  def __init__(self, kernel_size=5, **kwargs):
    super().__init__(**kwargs)
    self.supports_masking = True
    self.kernel_size = kernel_size
    self.conv = tf.keras.layers.Conv1D(1,
                                       kernel_size=kernel_size,
                                       strides=1,
                                       padding="same",
                                       use_bias=False)

  def call(self, inputs, mask=None):
    nn = tf.keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)
    nn = tf.expand_dims(nn, -1)
    nn = self.conv(nn)
    nn = tf.squeeze(nn, -1)
    nn = tf.nn.sigmoid(nn)
    nn = nn[:, None, :]
    return inputs * nn


class CausalDWConv1D(tf.keras.layers.Layer):

  def __init__(self,
               kernel_size=17,
               dilation_rate=1,
               use_bias=False,
               depthwise_initializer='glorot_uniform',
               name='',
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.causal_pad = tf.keras.layers.ZeroPadding1D(
        (dilation_rate * (kernel_size - 1), 0), name=name + '_pad')
    self.dw_conv = tf.keras.layers.DepthwiseConv1D(
        kernel_size,
        strides=1,
        dilation_rate=dilation_rate,
        padding='valid',
        use_bias=use_bias,
        depthwise_initializer=depthwise_initializer,
        name=name + '_dwconv')
    self.supports_masking = True

  def call(self, inputs):
    x = self.causal_pad(inputs)
    x = self.dw_conv(x)
    return x


def Conv1DBlock(channel_size,
                kernel_size,
                dilation_rate=1,
                drop_rate=0.0,
                expand_ratio=2,
                se_ratio=0.25,
                activation='swish',
                name=None):
  '''
    efficient conv1d block, @hoyso48
    '''
  if name is None:
    name = str(tf.keras.backend.get_uid("mbblock"))
  # Expansion phase
  def apply(inputs):
    channels_in = tf.keras.backend.int_shape(inputs)[-1]
    channels_expand = channels_in * expand_ratio

    skip = inputs

    x = tf.keras.layers.Dense(channels_expand,
                              use_bias=True,
                              activation=activation,
                              name=name + '_expand_conv')(inputs)

    # Depthwise Convolution
    x = CausalDWConv1D(kernel_size,
                       dilation_rate=dilation_rate,
                       use_bias=False,
                       name=name + '_dwconv')(x)

    x = tf.keras.layers.BatchNormalization(momentum=0.95, name=name + '_bn')(x)

    if FLAGS.use_eca:
      x = ECA()(x)

    x = tf.keras.layers.Dense(channel_size,
                              use_bias=True,
                              name=name + '_project_conv')(x)

    if drop_rate > 0:
      if FLAGS.inst_drop:
        x = tf.keras.layers.Dropout(drop_rate,
                                    noise_shape=(None, 1, 1),
                                    name=name + '_drop')(x)
      else:
        x = tf.keras.layers.Dropout(drop_rate, name=name + '_drop')(x)

    if (channels_in == channel_size):
      x = tf.keras.layers.add([x, skip], name=name + '_add')
    return x

  return apply


class MultiHeadSelfAttention(tf.keras.layers.Layer):

  def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
    super().__init__(**kwargs)
    self.dim = dim
    self.scale = self.dim**-0.5
    self.num_heads = num_heads
    ic(self.num_heads)
    self.qkv = tf.keras.layers.Dense(3 * dim, use_bias=False)
    self.drop1 = tf.keras.layers.Dropout(dropout)
    self.proj = tf.keras.layers.Dense(dim, use_bias=False)
    self.supports_masking = True

  def call(self, inputs, mask=None):
    #-> [128, 256, 384]
    qkv = self.qkv(inputs)
    #-> [128, 256, 4, 96] -> [128, 4, 256, 96]
    qkv = tf.keras.layers.Permute((2, 1, 3))(tf.keras.layers.Reshape(
        (-1, self.num_heads, self.dim * 3 // self.num_heads))(qkv))
    # q,k,v [128, 4, 256, 32]
    q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)
    # [128, 4, 256, 256]
    attn = tf.matmul(q, k, transpose_b=True) * self.scale

    if mask is not None:
      mask = mask[:, None, None, :]

    attn = tf.keras.layers.Softmax(axis=-1)(attn, mask=mask)
    attn = self.drop1(attn)

    # [128, 4, 256, 256] * [128, 4, 256, 32] -> [128, 4, 256, 32]
    x = attn @ v
    # ->[128, 256, 4, 32] -> [128, 256, 128]
    x = tf.keras.layers.Reshape((-1, self.dim))(tf.keras.layers.Permute(
        (2, 1, 3))(x))
    # -> [128, 256, 128]
    x = self.proj(x)
    return x

def TransformerBlock(dim=256,
                     num_heads=4,
                     expand=4,
                     attn_dropout=0.2,
                     drop_rate=0.2,
                     activation='swish'):

  def apply(inputs):
    x = inputs
    x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
    x = MultiHeadSelfAttention(dim=dim,
                               num_heads=num_heads,
                               dropout=attn_dropout)(x)
    if FLAGS.inst_drop:
      x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
    else:
      x = tf.keras.layers.Dropout(drop_rate)(x)
    x = tf.keras.layers.Add()([inputs, x])
    attn_out = x

    x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
    x = tf.keras.layers.Dense(dim * expand,
                              use_bias=False,
                              activation=activation)(x)
    x = tf.keras.layers.Dense(dim, use_bias=False)(x)
    if FLAGS.inst_drop:
      x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
    else:
      x = tf.keras.layers.Dropout(drop_rate)(x)
    x = tf.keras.layers.Add()([attn_out, x])
    return x

  return apply

def get_conv1d_blocks(dim=384, ksize=11, ksize_vals=[], name='conv1d_blocks'):
  if FLAGS.pad_frames:
    input_shape = (FLAGS.n_frames, dim)
  else:
    input_shape = (None, dim)
  inp = tf.keras.Input(shape=input_shape)
  x = inp
  if FLAGS.use_masking:
    x = tf.keras.layers.Masking(input_shape=input_shape)(x)
  
  for ksize in ksize_vals:
    x = Conv1DBlock(dim, ksize, drop_rate=0.2)(x)

  model = tf.keras.Model(inp, x, name=name)
  return model

def get_transformer_block(dim=384, num_heads=4, expand=2, name='transformer_block'):
  if FLAGS.pad_frames:
    input_shape = (FLAGS.n_frames, dim)
  else:
    input_shape = (None, dim)
  inp = tf.keras.Input(shape=input_shape)
  x = inp
  if FLAGS.use_masking:
    x = tf.keras.layers.Masking(input_shape=input_shape)(x)
  
  x = TransformerBlock(dim, num_heads=num_heads, expand=expand)(x)

  model = tf.keras.Model(inp, x, name=name)
  return model

class Conv1DBlocks(tf.keras.layers.Layer):
  def __init__(self, dim=384, ksize_vals=[11,11,11]):
    super(Conv1DBlocks, self).__init__()
    self.dim = dim
    self.conv1d_blocks = get_conv1d_blocks(dim, ksize_vals=ksize_vals, name='conv1d_blocks')
    
  def call(self, x):
    x = self.conv1d_blocks(x)
    return x
  
class TransformerBlocks(tf.keras.layers.Layer):
  def __init__(self, dim=384, num_heads=4, expand=2, num_layers=1):
    super(TransformerBlocks, self).__init__()
    self.dim = dim
    self.transformer_blocks = tf.keras.Sequential([
      get_transformer_block(dim, num_heads, expand, name=f'transformer_block_{layer}') for layer in range(num_layers)
      ], name='transformer_blocks')
    
  def call(self, x):
    x = self.transformer_blocks(x)
    return x

class Conv1dTransformerBlock(tf.keras.layers.Layer):
  def __init__(self, dim=384, dropout_step=0, expand=2, num_conv1d_layers=3, num_transofrmer_layers=1):
    super(Conv1dTransformerBlock, self).__init__()
    self.conv1d_blocks = tf.keras.Sequential([
      get_conv1d_block(dim, dropout_step, name=f'conv1d_block_{layer}') for layer in range(num_conv1d_layers)
      ])
    self.transformer_blocks = tf.keras.Sequential([
      get_transformer_block(dim, expand, name=f'transformer_block_{layer}') for layer in range(num_transofrmer_layers)
      ])
    
  def call(self, x):
    x = self.conv1d_blocks(x)
    x = self.transformer_blocks(x)
    return x
