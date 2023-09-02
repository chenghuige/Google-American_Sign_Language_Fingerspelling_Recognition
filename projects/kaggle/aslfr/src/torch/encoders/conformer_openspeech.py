#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   conformer.py
#        \author   chenghuige  
#          \date   2023-08-05 09:02:16.428718
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import *
import melt as mt
from src.config import *
from src.torch.embedding import *
from src.torch.layers import Conv1DBlocks

class Encoder(nn.Module):

  def __init__(self):
    super(Encoder, self).__init__()
    self.embedding = get_embeddding() if FLAGS.embedding else SimpleEmbedding()
    from openspeech.encoders.conformer_encoder import ConformerEncoder
    self.encoder =  ConformerEncoder(
          input_dim=FLAGS.encoder_units,
          encoder_dim=FLAGS.encoder_units,
          num_layers=FLAGS.encoder_layers,          
          num_attention_heads=FLAGS.mhatt_heads,
          conv_expansion_factor=FLAGS.conv1d_expansion_factor,
          conv_kernel_size=FLAGS.conv1d_ksize_vals[0],
    )
    # gezi.set('torch2tf', True)
  def forward(self, x_inp):
    x = self.embedding(x_inp)
    x = self.encoder(x)
    return x  
