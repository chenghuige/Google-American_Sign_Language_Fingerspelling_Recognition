#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   preprocess.py
#        \author   chenghuige
#          \date   2023-06-25 16:34:33.330160
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import *
from src.config import *
"""
    Tensorflow layer to process data in TFLite
    Data needs to be processed in the model itself, so we can not use Python
"""

class PreprocessLayerNonNaN(tf.keras.layers.Layer):

  def __init__(self, n_hand_cols=84):
    super(PreprocessLayerNonNaN, self).__init__()
    self.n_hand_cols = n_hand_cols

  @tf.function(input_signature=(tf.TensorSpec(shape=[None, N_COLS],
                                              dtype=tf.float32),),)
  def call(self, data0):
    # Fill NaN Values With 0
    data = tf.where(tf.math.is_nan(data0), 0.0, data0)

    # Hacky
    data = data[None]

    # Empty Hand Frame Filtering
    hands = tf.slice(data, [0, 0, 0], [-1, -1, self.n_hand_cols])
    hands = tf.abs(hands)
    mask = tf.reduce_sum(hands, axis=2)
    mask = tf.not_equal(mask, 0)
    data = data[mask][None]
    data = tf.squeeze(data, axis=[0])

    return data


"""
    Tensorflow layer to process data in TFLite
    Data needs to be processed in the model itself, so we can not use Python
"""


class PreprocessLayer(tf.keras.layers.Layer):
  # 128 if add z
  def __init__(self, n_frames=128, n_hand_cols=84):
    super(PreprocessLayer, self).__init__()
    self.n_frames = n_frames
    self.n_hand_cols = n_hand_cols
    ic(N_COLS, self.n_frames)
    
  @tf.function(input_signature=(tf.TensorSpec(shape=[None, N_COLS],
                                              dtype=tf.float32),),)
  def call(self, data0, trunct_method=None):
    trunct_method = trunct_method or FLAGS.trunct_method
    # Fill NaN Values With 0
    data = tf.where(tf.math.is_nan(data0), 0.0, data0)
    # Hacky
    data = data[None]
        
    n_frames = len(data[0])
    
    if FLAGS.filter_nan_frames:
      if n_frames > self.n_frames or FLAGS.always_filter_nan_frames:
        frames = tf.abs(data)
        mask = tf.reduce_sum(frames, axis=2)
        mask = tf.not_equal(mask, 0)
        data = data[mask][None]
    
    n_frames = len(data[0])
    if n_frames > self.n_frames:
      if FLAGS.filter_nan_hands:
        # Empty Hand Frame Filtering
        hands = tf.slice(data, [0, 0, 0], [-1, -1, self.n_hand_cols])
        hands = tf.abs(hands)
        mask = tf.reduce_sum(hands, axis=2)
        mask = tf.not_equal(mask, 0)
        data = data[mask][None]

    if not FLAGS.always_resize:
      # Pad Zeros  TODO dynamic type so not pad? dynanmic seq2seq input
      n_frames = len(data[0])
      if n_frames < self.n_frames:
        data = tf.concat(
            (data,
            tf.zeros([1, self.n_frames - n_frames, N_COLS], dtype=tf.float32)),
            axis=1)
      
      n_frames = len(data[0])
      if n_frames > self.n_frames:
        if trunct_method == 'resize':
          # Downsample
          data = tf.image.resize(
              data,
              [1, self.n_frames],
              method=tf.image.ResizeMethod.BILINEAR,
          )
        else:
          data = data[:, :self.n_frames, :]
    else:
      data = tf.image.resize(
              data,
              [1, self.n_frames],
              method=tf.image.ResizeMethod.BILINEAR,
          )
          
    data = tf.reshape(data, [1, self.n_frames, N_COLS])
    
    # Squeeze Batch Dimension
    data = tf.squeeze(data, axis=[0])

    return data
