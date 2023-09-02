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

from torch import nn
from gezi.common import * 
from src import util
from src.torch.layers import AvgPoolingModule

class Encoder(nn.Module):

  def __init__(self):
    super(Encoder, self).__init__()
    self.encoder = util.get_encoder()
    if FLAGS.encode_pool_size > 1:
      if FLAGS.share_reduce:
        self.pooling = gezi.get('time_reduce_module')
      self.pooling = AvgPoolingModule(FLAGS.encode_pool_size)

  def forward(self, frames):
    x = self.encoder(frames)  
    if FLAGS.encode_pool_size > 1:
      x = self.pooling(x)
    if FLAGS.inter_ctc and FLAGS.inter_ctc_pooling:
      if self.training:
        x_ = gezi.get('inter_ctc_out')
        if x_ is not None:
          x_ = self.pooling(x_)
        gezi.set('inter_ctc_out', x_)
    return x
  
