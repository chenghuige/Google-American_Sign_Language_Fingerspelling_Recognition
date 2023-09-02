#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dataset.py
#        \author   chenghuige  
#          \date   2023-06-26 17:16:21.385835
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
from src.config import *
from src.tf.preprocess import PreprocessLayer
import melt as mt

class Dataset(mt.Dataset):
  def __init__(self, subset='valid', **kwargs):
    super(Dataset, self).__init__(subset, **kwargs)
    self.prepocess = PreprocessLayer(FLAGS.n_frames)

  def parse(self, example):
    dynamic_keys = []
    # dynamic_keys = ['frames']
    self.auto_parse(exclude_keys=dynamic_keys)
    self.adds(dynamic_keys)
    fe = self.parse_(serialized=example)
    # fe['frames'] = self.prepocess(tf.reshape(fe['frames'], [-1, N_COLS]))
    mt.try_append_dim(fe) 
    if FLAGS.decode_phrase_type:
      fe['phrase'] = tf.concat([fe['phrase_type_'] + VOCAB_SIZE - 3, fe['phrase']], axis=-1)
    x = fe
    y = fe['phrase']
    return x, y
