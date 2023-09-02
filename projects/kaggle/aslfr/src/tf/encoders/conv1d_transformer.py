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
from src.tf.preprocess import PreprocessLayer
from src.tf.util import *
from src.tf import loss
from src.tf.embedding import *
from src.tf.layers import *
from src.tf.decode import decode_phrase

class Encoder(tf.keras.Model):

  def __init__(self):
    super(Encoder, self).__init__(name='conv1d_transformer_encoder')
    self.embedding = get_embeddding() if FLAGS.embedding else SimpleEmbedding()
    self.supports_masking = True
    
    self.encoder = tf.keras.Sequential([
          tf.keras.Sequential([
          Conv1DBlocks(FLAGS.encoder_units, ksize_vals=FLAGS.conv1d_ksize_vals),
          TransformerBlocks(FLAGS.encoder_units, 
                            num_heads=FLAGS.mhatt_heads, 
                            expand=FLAGS.conv1d_expansion_factor, 
                            num_layers=FLAGS.transformer_layers),
        ], name=f'conv1d_transformer_layer_{layer}') for layer in range(FLAGS.encoder_layers)
      ], name='encoder')
    
  def call(self, x_inp):
    x = self.embedding(x_inp)
    x = self.encoder(x)
    return x
  