#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   rnn.py
#        \author   chenghuige  
#          \date   2023-07-06 20:51:37.888389
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
import melt as mt
from src.config import *
from src.tf.util import *  
from src.tf.embedding import *
from src.tf.layers import CrossAttention
  
class Encoder(tf.keras.layers.Layer):
  def __init__(self):
    super(Encoder, self).__init__()
    # Embedding layer for Landmarks
    self.embedding = get_embeddding() if FLAGS.embedding else SimpleEmbedding()
    RNN = getattr(tf.keras.layers, FLAGS.rnn)
    # The RNN layer processes those vectors sequentially.
    self.encoder = tf.keras.Sequential(
        [          
          tf.keras.layers.Bidirectional(
            merge_mode=FLAGS.rnn_merge,
            layer=RNN(FLAGS.encoder_units,
                      # Return the sequence and state
                      return_sequences=True,
                      dropout=FLAGS.rnn_drop, 
                      # recurrent_dropout=0.2,
                      recurrent_initializer='glorot_uniform'), 
            name='rnn_{}'.format(i)) for i in range(FLAGS.encoder_layers)
        ], name='encoder')
    self.supports_masking = True

  def call(self, x):
    x = self.embedding(x)
    x = self.encoder(x)
    return x
    