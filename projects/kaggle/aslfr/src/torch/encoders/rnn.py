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

from torch import nn
from gezi.common import * 
import melt as mt
from src.config import *
from src.torch.embedding import *
from src.torch.layers import BatchNorm

class RnnBlock(nn.Module):
  def __init__(self, n_layers=1):
    super(RnnBlock, self).__init__()
    RNN = getattr(nn, FLAGS.rnn)
    self.rnn = RNN(FLAGS.encoder_units,
                   FLAGS.encoder_units, 
                   n_layers,
                   dropout=FLAGS.rnn_drop, 
                   bidirectional=True, 
                   batch_first=True)
    if FLAGS.rnn_batchnorm:
      self.bn = BatchNorm(FLAGS.encoder_units, momentum=0.05)
    
  def forward(self, x):
    x = self.rnn(x)[0]
    x = x[..., :FLAGS.encoder_units] + x[..., FLAGS.encoder_units:]
    if FLAGS.rnn_batchnorm:
      x = self.bn(x)
    return x
  
class Encoder(nn.Module):
  def __init__(self):
    super(Encoder, self).__init__()
    # Embedding layer for Landmarks
    self.embedding = get_embeddding() if FLAGS.embedding else SimpleEmbedding()
    RNN = getattr(nn, FLAGS.rnn)
    self.encoder = RnnBlock(FLAGS.encoder_layers)

  def forward(self, x):
    x = self.embedding(x)
    x = self.encoder(x)
    return x
    