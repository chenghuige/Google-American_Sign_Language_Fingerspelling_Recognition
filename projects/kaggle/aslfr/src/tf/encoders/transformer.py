#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   transformer.py
#        \author   chenghuige  
#          \date   2023-07-06 20:51:44.575354
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import *
import melt as mt
from src.config import *
from src.tf.preprocess import PreprocessLayer
from src.tf.util import *
from src.tf import loss
from src.tf.embedding import *
from src.tf.decode import decode_phrase
from src.tf.layers import *

# Encoder based on multiple transformer blocks
class Encoder(tf.keras.Model):

  def __init__(self):
    super(Encoder, self).__init__(name='transformer_encoder')
    self.embedding = get_embeddding() if FLAGS.embedding else PositionEmbedding()
    self.num_blocks = FLAGS.encoder_layers
    self.supports_masking = True
    
  def build(self, input_shape):
    self.ln_1s = []
    self.mhas = []
    self.ln_2s = []
    self.mlps = []
    # Make Transformer Blocks
    for i in range(self.num_blocks):
      # First Layer Normalisation
      self.ln_1s.append(
          tf.keras.layers.LayerNormalization(epsilon=FLAGS.layer_norm_eps))
      # Multi Head Attention
      self.mhas.append(MultiHeadAttention(FLAGS.encoder_units, FLAGS.mhatt_heads, FLAGS.mhatt_drop))
      # Second Layer Normalisation
      self.ln_2s.append(
          tf.keras.layers.LayerNormalization(epsilon=FLAGS.layer_norm_eps))
      # Multi Layer Perception
      self.mlps.append(
          tf.keras.Sequential([
              tf.keras.layers.Dense(FLAGS.encoder_units * FLAGS.mlp_ratio,
                                    activation=GELU,
                                    kernel_initializer=INIT_GLOROT_UNIFORM),
              tf.keras.layers.Dropout(FLAGS.mlp_drop),
              tf.keras.layers.Dense(FLAGS.encoder_units,
                                    kernel_initializer=INIT_HE_UNIFORM),
          ]))

  def call(self, x_inp):
    if FLAGS.ignore_nan_frames:
      # Attention mask to ignore missing frames
      attention_mask = tf.where(
          tf.math.reduce_sum(x_inp, axis=[2]) == 0.0, 0.0, 1.0)
      attention_mask = tf.expand_dims(attention_mask, axis=1)
      attention_mask = tf.repeat(attention_mask, repeats=FLAGS.n_frames, axis=1)
    else:
      attention_mask = None

    x = self.embedding(x_inp)
    
    # Iterate input over transformer blocks
    for ln_1, mha, ln_2, mlp in zip(self.ln_1s, self.mhas, self.ln_2s,
                                    self.mlps):
      x = ln_1(x + mha(x, x, x, attention_mask=attention_mask))
      x = ln_2(x + mlp(x))

    return x
    
