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
from src.torch.embedding import *
from src.torch.layers import Conv1DBlocks, TransformerBlock

class Encoder(nn.Module):

  def __init__(self):
    super(Encoder, self).__init__()
    self.embedding = get_embeddding() if FLAGS.embedding else SimpleEmbedding()
    self.encoder = nn.Sequential(*[
      nn.Sequential(
        Conv1DBlocks(FLAGS.encoder_units, ksize_vals=FLAGS.conv1d_ksize_vals),
        TransformerBlock(FLAGS.encoder_units, 
                         FLAGS.encoder_units, 
                         expand=2,
                         attn_dropout=FLAGS.attn_drop),
      ) for _ in range(FLAGS.encoder_layers)
    ])
    # gezi.set('torch2tf', True)
    
  def forward(self, x_inp):
    x = self.embedding(x_inp)
    x = self.encoder(x)
    return x
  