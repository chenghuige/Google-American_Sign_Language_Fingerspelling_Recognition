#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   decoder.py
#        \author   chenghuige  
#          \date   2023-07-16 07:47:42.615107
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
import tensorflow as tf
from src import util

class Decoder(tf.keras.Model):

  def __init__(self):
    super(Decoder, self).__init__(name='decoder')
    self.decoder = util.get_decoder()
    ic(self.decoder)
    self.supports_masking = True
    
  def call(self, encoder_outputs, phrase):
    x = self.decoder(encoder_outputs, phrase)  
    return x
  
