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
from src.torch.layers import Conv1DBlocks, TransformerBlock

class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ConformerConvBlock(nn.Module):
    """A single convolution block for the Conformer encoder.

    Args:
        d_model (int): input/output dimension
        kernel_size (int): kernel size in depthwise convolution
        param_init (str): parameter initialization method
        normalization (str): batch_norm/group_norm/layer_norm
        causal (bool): causal mode for streaming infernece

    """

    def __init__(self, d_model, kernel_size=11, param_init=None, normalization='batch_norm',
                 causal=False):

        super().__init__()

        assert (kernel_size - 1) % 2 == 0, 'kernel_size must be the odd number.'
        assert kernel_size >= 3, 'kernel_size must be larger than 3.'
        self.kernel_size = kernel_size
        self.causal = causal

        if causal:
            self.padding = (kernel_size - 1)
        else:
            self.padding = (kernel_size - 1) // 2

        self.pointwise_conv1 = nn.Conv1d(in_channels=d_model,
                                         out_channels=d_model * 2,  # for GLU
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        self.depthwise_conv = nn.Conv1d(in_channels=d_model,
                                        out_channels=d_model,
                                        kernel_size=kernel_size,
                                        stride=1,
                                        padding=self.padding,
                                        groups=d_model,  # depthwise
                                        bias=True)

        if normalization == 'batch_norm':
            self.norm = nn.BatchNorm1d(d_model)
        elif normalization == 'group_norm':
            num_groups = 2
            self.norm = nn.GroupNorm(num_groups=max(1, d_model // num_groups),
                                     num_channels=d_model)
        elif normalization == 'layer_norm':
            self.norm = nn.LayerNorm(d_model, eps=1e-12)
        else:
            raise NotImplementedError(normalization)
        logger.info('normalization: %s' % normalization)
        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(in_channels=d_model,
                                         out_channels=d_model,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)

        if param_init == 'xavier_uniform':
            self.reset_parameters_xavier_uniform()
        elif param_init == 'lecun':
            self.reset_parameters_lecun()
        else:
            logger.info('Parameter initialization is skipped.')

    def reset_parameters_xavier_uniform(self):
        """Initialize parameters with Xavier uniform distribution."""
        logger.info('===== Initialize %s with Xavier uniform distribution =====' %
                    self.__class__.__name__)
        for conv_layer in [self.pointwise_conv1, self.pointwise_conv2, self.depthwise_conv]:
            for n, p in conv_layer.named_parameters():
                init_with_xavier_uniform(n, p)

    def reset_parameters_lecun(self, param_init=0.1):
        """Initialize parameters with lecun style.."""
        logger.info('===== Initialize %s with lecun style =====' %
                    self.__class__.__name__)
        for conv_layer in [self.pointwise_conv1, self.pointwise_conv2, self.depthwise_conv]:
            for n, p in conv_layer.named_parameters():
                init_with_lecun_normal(n, p, param_init)

    def forward(self, xs):
        """Forward pass.

        Args:
            xs (FloatTensor): `[B, T, d_model]`
        Returns:
            xs (FloatTensor): `[B, T, d_model]`

        """
        bs, xmax, dim = xs.size()

        xs = xs.transpose(2, 1).contiguous()  # `[B, C, T]`
        xs = self.pointwise_conv1(xs)  # `[B, 2 * C, T]`
        xs = F.glu(xs, dim=1)  # `[B, C, T]`

        xs = self.depthwise_conv(xs)  # `[B, C, T]`
        if self.causal:
            xs = xs[:, :, :-self.padding]

        xs = xs.transpose(2, 1)
        if isinstance(self.norm, nn.LayerNorm):
            xs = self.activation(self.norm(xs))  # `[B, T, C]`
        else:
            # time-independent normalization
            xs = xs.contiguous().view(bs * xmax, -1, 1)
            xs = self.activation(self.norm(xs))  # `[B * T, C, 1]`
            xs = xs.view(bs, xmax, -1)
        xs = xs.transpose(2, 1)
        xs = self.pointwise_conv2(xs)  # `[B, C, T]`

        xs = xs.transpose(2, 1).contiguous()  # `[B, T, C]`
        return xs


class Encoder(nn.Module):

  def __init__(self):
    super(Encoder, self).__init__()
    self.embedding = get_embeddding() if FLAGS.embedding else SimpleEmbedding()
    self.encoder = nn.Sequential(*[
      nn.Sequential(
        ConformerConvBlock(d_model=FLAGS.encoder_units, 
                          kernel_size=FLAGS.conv1d_ksize_vals[0]),
        TransformerBlock(FLAGS.encoder_units, FLAGS.encoder_units, expand=2),
      ) for _ in range(FLAGS.encoder_layers)
    ])
  def forward(self, x_inp):
    x = self.embedding(x_inp)
    x = self.encoder(x)
    return x  
