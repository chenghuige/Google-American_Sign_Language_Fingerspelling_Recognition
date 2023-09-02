#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   encoder.py
#        \author   chenghuige  
#          \date   2023-07-16 07:41:37.198413
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
import tensorflow as tf
from src import util

class Encoder(tf.keras.Model):

  def __init__(self):
    super(Encoder, self).__init__(name='encoder')
    self.encoder = util.get_encoder()
    ic(self.encoder)
    if FLAGS.encode_pool_size > 1:
      self.pooling = tf.keras.layers.AveragePooling1D(FLAGS.encode_pool_size)
    self.supports_masking = True
    
  def call(self, frames):
    x = self.encoder(frames)  
    if FLAGS.encode_pool_size > 1:
      x = self.pooling(x)
    return x
  
