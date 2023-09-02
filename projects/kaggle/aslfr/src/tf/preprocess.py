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
import melt as mt
from src.config import *
from src.tf.augment import *

# # TODO Simplified InferOnlyPrePorcssLayer?
# class PreprocessLayer(tf.keras.layers.Layer):

#   def __init__(self, n_frames=128, training=False):
#     super(PreprocessLayer, self).__init__()
#     self.n_frames = n_frames
#     self.n_cols = get_n_cols(no_motion=True, use_z=True)
#     self.training = training
#     self.means = gezi.load(f'{FLAGS.root}/means.npy')
#     self.stds = gezi.load(f'{FLAGS.root}/stds.npy')
#     ic(self.means.shape, self.stds.shape)

#   @tf.function(input_signature=(tf.TensorSpec(shape=[None, N_COLS],
#                                               dtype=tf.float32),),)
#   def call(self, data):
#     # Hacky (add dim in front line shape [3,4] to shape [1, 3, 4])
#     data = data[None]
#     dtype = data.dtype
         
#     # seems norm before resize produce better results
#     # -------norm frames
    
#     l = []
#     l.append(
#         (data - self.means) / self.stds,
#     )
    
#     if FLAGS.concat_frames:   
#       data = tf.concat([data] + l, axis=-1)
#     elif FLAGS.norm_frames:
#       data = l[0]
    
        
#     # Fill NaN Values With 0
#     data = tf.where(tf.math.is_nan(data), tf.zeros_like(data), data)
                   
#     n_frames = len(data[0])
#     pos = get_pos(n_frames, dtype)
#     data = tf.concat([data, pos], axis=-1)
    
#     # ------add addional info like add pos after resample, actually similar results before or after resample
#     n_frames = len(data[0])
    
#     # Pad Zeros  TODO dynamic type so not pad? dynanmic seq2seq input
#     n_frames = len(data[0])
#     if n_frames < self.n_frames:
#       data = tf.concat((data,
#                         tf.zeros([1, self.n_frames - n_frames, self.n_cols],
#                                   dtype=dtype)),
#                       axis=1)

#     n_frames = len(data[0])
#     if n_frames > self.n_frames:
#       # Downsample
#       data = tf.image.resize(
#           data,
#           [1, self.n_frames],
#           method=FLAGS.resize_method,
#       )
#       # For resize The return value has type float32, unless the method is ResizeMethod.NEAREST_NEIGHBOR
#       data = tf.cast(data, dtype)

#     data = tf.reshape(data, [1, self.n_frames, self.n_cols])
      
#     # Squeeze Batch Dimension
#     data = tf.squeeze(data, axis=[0])

#     return data

"""
    Tensorflow layer to process data in TFLite
    Data needs to be processed in the model itself, so we can not use Python
    [None, N_COLS] -> [1, NONE, N_COLS] -> [1, N_FRAMES, N_COLS]
"""
class PreprocessLayer(tf.keras.layers.Layer):

  def __init__(self, n_frames=128, training=False):
    super(PreprocessLayer, self).__init__()
    self.n_frames = n_frames
    self.n_cols = get_n_cols(no_motion=True, use_z=True)
    ic(self.n_cols)
    self.means = gezi.load(f'{FLAGS.root}/means.npy')
    self.stds = gezi.load(f'{FLAGS.root}/stds.npy')
    ic(self.means.shape, self.stds.shape)
    self.training = training

  ## seems not needed as training perf is similar
  @tf.function(input_signature=(tf.TensorSpec(shape=[None, N_COLS],
                                              dtype=tf.float32),),)
  ## this will cause error
  # @tf.function()
  def call(self, data):
    trunct_method = FLAGS.trunct_method
    training = self.training

    # Hacky (add dim in front line shape [None,N_COLS] to shape [1, None, N_COLS])
    data = data[None]
    dtype = data.dtype
    
    if FLAGS.clip_input:
      mask = tf.cast(data > 0, dtype)
      abs_data = tf.math.abs(data)
      abs_data = tf.clip_by_value(abs_data, clip_value_min=1e-6, clip_value_max=100)
      data = abs_data * mask + (- abs_data) * (1 - mask)      
    
    # [1, None, N_COLS//3, 3]
    data = reshape(data)
    if FLAGS.clip_xy:
      data_xy, data_z = tf.split(data, [2, 1], axis=-1)
      data_xy = tf.clip_by_value(data_xy, clip_value_min=1e-6, clip_value_max=1.0 - 1e-6)
      data = tf.concat([data_xy, data_z], axis=-1)

    ## Notice for flip lr you need 1 - x, so must put it before turn nan to 0
    if not FLAGS.dominant_flip:
      if training and FLAGS.use_aug:
        # flip not affect much 
        data = mt.Apply(flip_lr, FLAGS.flip_rate)(data)    
        
    # TODO check
    if FLAGS.dominant_flip or (not training and FLAGS.pred_flip):
      left_hand = data[:, :N_HAND_POINTS]
      right_hand = data[:, N_HAND_POINTS:N_HAND_POINTS * 2]
      right_dominant = tf.reduce_mean(tf.abs(right_hand)) > tf.reduce_mean(tf.abs(left_hand))
      data = tf.cond(right_dominant, 
                     lambda: data, 
                     lambda: flip_lr(data))
    if FLAGS.force_flip:
      data = flip_lr(data)
    
    if training and FLAGS.use_aug:
      data = mt.Apply(scale, FLAGS.scale_rate)(data)
      data = mt.Apply(rotate, FLAGS.rotate_rate)(data)
      # shift not affect much..
      data = mt.Apply(shift, FLAGS.shift_rate)(data)
      data = mt.OneOf([scale, rotate, shift], FLAGS.affine_rate)(data)
      data = mt.Apply(temporal_seq_mask, FLAGS.temporal_seq_mask_rate)(data)
    
    data_ = data
    data = reshape_back(data)
      
    # seems norm before resize produce better results
    # -------norm frames
    l = []
    if FLAGS.norm_frames:
      if FLAGS.concat_frames:
        l.append(data_)
      l.append(
          (data - self.means) / self.stds,
      )
      l[-1] = reshape(l[-1])
    else:
      l.append(data_)
    
    # # work worse
    # if FLAGS.norm_frame:
    #   l.append(norm_frame(data_))
      
    if FLAGS.norm_hands:
      l.append(norm_hands(data_))
      
    if FLAGS.add_motion:
      l.append(add_motion(data_))
            
    # [1, None, feats, 3], concat_frames_dim = -2
    data = tf.concat(l, axis=FLAGS.concat_frames_dim)

    n_cols = self.n_cols if not FLAGS.add_pos else self.n_cols - 1
    data = tf.reshape(data, [1, -1, n_cols])
    # Fill NaN Values With 0
    data = tf.where(tf.math.is_nan(data), tf.zeros_like(data), data)
                       
    # add_pos helps a lot and add_pos_before_resample also +0.002 compare to add_pos_after_resample
    if FLAGS.add_pos and FLAGS.add_pos_before_resample:
      # n_frames = len(data[0])
      n_frames = tf.shape(data)[1]
      pos = get_pos(n_frames, dtype)
      data = tf.concat([data, pos], axis=-1)
      
    #  now basic features done, we resample for frames with basic features
    if training and FLAGS.use_aug:
      # resample for time range, like 120 frames to 135 or 90 frames
      # resample helps a lot
      data = mt.Apply(resample, FLAGS.resample_rate)(data)
      data = tf.cast(data, dtype)
      
    # ------add addional info like add pos after resample, actually similar results before or after resample
    n_frames = tf.shape(data)[1]
    
    # do it after expand dim 0, and do not use expand_dims before...
    if FLAGS.add_pos and (not FLAGS.add_pos_before_resample):
      pos = get_pos(n_frames, dtype)
      data = tf.concat([data, pos], axis=-1)

    # -------------  try padding to max len if less frames exists, then resize to n_frames if more frames exits
    if n_frames > self.n_frames:
      if FLAGS.filter_nan_hands or FLAGS.mask_nan_hands:
        # Empty Hand Frame Filtering
        # hands = tf.slice(data, [0, 0, 0], [-1, -1, self.n_hand_cols])
        hands = data[:, :, :N_HAND_POINTS * 2 * 3]
        hands = tf.abs(hands)
        mask = tf.reduce_sum(hands, axis=2)
        mask = tf.not_equal(mask, 0)
        if FLAGS.filter_nan_hands:
          if FLAGS.nan_hands_method == 1:
            # https://www.kaggle.com/competitions/asl-fingerspelling/discussion/434353
            alternating_tensor = tf.math.equal(tf.cumsum(
              tf.ones_like(hands[:,:,0]), axis=1)%2, 1.0)
            mask = tf.math.logical_or(mask, alternating_tensor)
          elif FLAGS.nan_hands_method == 2:
            mask2 = tf.concat([tf.ones_like(mask[:,0:1], mask.dtype), mask[:,:-1]], axis=1)
            mask = tf.math.logical_or(mask, mask2)
          data = data[mask][None]            
        else:
          data = data * tf.expand_dims(tf.cast(mask, dtype), axis=-1)

    if not FLAGS.always_resize:
      # Pad Zeros  TODO dynamic type so not pad? dynanmic seq2seq input
      # n_frames = len(data[0])
      n_frames = tf.shape(data)[1]
      if n_frames < self.n_frames:
        if FLAGS.pad_frames:
          if FLAGS.pad_method == 'zero':
            data = tf.concat((data,
                              tf.zeros([1, self.n_frames - n_frames, self.n_cols],
                                        dtype=dtype)),
                            axis=1)
          else:
            data = tf.image.resize(
              data,
              [1, self.n_frames],
              method=FLAGS.pad_resize_method,
            )
        elif n_frames < FLAGS.encode_pool_size:
          data = tf.concat(
              (data,
               tf.zeros([1, FLAGS.encode_pool_size - n_frames, self.n_cols],
                        dtype=dtype)),
              axis=1)
        elif n_frames < FLAGS.min_frames:
          data = tf.concat(
              (data,
               tf.zeros([1, FLAGS.min_frames - n_frames, self.n_cols],
                        dtype=dtype)),
              axis=1)

      n_frames = tf.shape(data)[1]
      if n_frames > self.n_frames:
        if trunct_method == 'resize':
          # Downsample
          data = tf.image.resize(
              data,
              [1, self.n_frames],
              method=FLAGS.resize_method,
          )
          # For resize The return value has type float32, unless the method is ResizeMethod.NEAREST_NEIGHBOR
          data = tf.cast(data, dtype)
        else:
          data = data[:, :self.n_frames, :]
    else:
      data = tf.image.resize(
          data,
          [1, self.n_frames],
          method=FLAGS.resize_method,
      )
      data = tf.cast(data, dtype)

    if FLAGS.pad_frames:
      data = tf.reshape(data, [1, self.n_frames, self.n_cols])
      
    # 这个mask是否放在最后最好 是否放在前面 然后filter 全0的帧？那样比较麻烦 感觉这样还行...
    if training and FLAGS.use_aug:
      # temperal mask help a lot
      data = mt.Apply(temporal_mask, FLAGS.temporal_mask_rate)(data)
      data = mt.Apply(spatio_mask, FLAGS.spatio_mask_rate)(data)
      
    if not FLAGS.use_z:
      if FLAGS.add_pos:
        pos_data = data[...,-2:-1]
        data = data[...,:-1]
      data = tf.reshape(data, [1, self.n_frames, -1, 3])
      data = data[...,:2]
      data = tf.reshape(data, [1, self.n_frames, -1])
      if FLAGS.add_pos:
        data = tf.concat([data, pos_data], axis=-1)
      
    # l = []
    # if FLAGS.add_motion:
    #   data_ = tf.where(tf.math.equal(data, 0), tf.zeros_like(data) + float('NaN'), data)
    #   data_motion = tf.concat([tf.zeros_like(data_[:, :1]), data_[:, 1:] - data_[:, :-1]], axis=1)
    #   data_motion = tf.where(tf.math.is_nan(data_motion), tf.zeros_like(data_motion), data_motion)
    #   l.append(data_motion)
    #   if FLAGS.add_motion2:
    #     data_motion = tf.concat([tf.zeros_like(data_[:, :2]), data_[:, 2:] - data_[:, :-2]], axis=1)
    #     data_motion = tf.where(tf.math.is_nan(data_motion), tf.zeros_like(data_motion), data_motion)
    #     l.append(data_motion)
      
    # if l:
    #   data = tf.concat([data] + l, axis=-1)
             
    # Squeeze Batch Dimension
    data = tf.squeeze(data, axis=[0])
    # tf.print('----', data.shape)
    return data, n_frames

class PreProcssor(object):
  def __init__(self, subset='valid', squeeze=False):
    training = subset == 'train'
    self.prepocess = PreprocessLayer(FLAGS.n_frames, training=training)
    self.squeeze = squeeze

  def __call__(self, fe):
    # # TODO HACK for hug datasets.. ds = ds.to_tf_dataset(batch_size=1)
    if self.squeeze:
      for key in fe:
        fe[key] = tf.squeeze(fe[key], axis=0)
    x = fe
    weights = fe['weight']
    fe['frames'] = tf.reshape(fe['frames'], [fe['n_frames'], N_COLS])
    fe['frames'], fe['n_frames_'] = self.prepocess(fe['frames']) 
    fe['phrase_len'] = tf.cond(tf.greater(fe['phrase_len'], MAX_PHRASE_LEN), lambda: tf.constant(MAX_PHRASE_LEN, fe['phrase_len'].dtype), lambda: fe['phrase_len'])
    if FLAGS.task == 'seq':
      if FLAGS.no_eos:
        # as in tfrecords we have eos, but in training if we don't have eos we change it to pad
        mask = tf.logical_or(fe['phrase_'] == PAD_IDX, fe['phrase_'] == EOS_IDX)
        mask = tf.cast(mask, fe['phrase_'].dtype)
        fe['phrase_'] = fe['phrase_'] * (1 - mask) + mask * PAD_IDX
      
      # # TODO do not need this if gen records set phrase_type_ = 'sup'..
      # if FLAGS.decode_phrase_type:
      #   # TODO here not work for torch try_append_dim
      #   # mt.try_append_dim(fe)
      #   # fe['phrase_'] = tf.expand_dims(fe['phrase_'], axis=-1)
      #   weights_mask = tf.cast(weights != 1, fe['phrase_type_'].dtype)
      #   fe['phrase_type_'] =  weights_mask * (N_TYPES - 1) + (1 - weights_mask) * fe['phrase_type_']
      #   fe['phrase_'] = tf.concat([fe['phrase_type_'] + VOCAB_SIZE - N_TYPES, fe['phrase_']], axis=-1)
      
      y = fe['phrase_']
    else:
      y = fe['cls_label']
      
    if FLAGS.mix_sup:
      weights_mask = tf.cast(weights != 1, tf.float32)
      weights = weights_mask * FLAGS.sup_weight + (1 - weights_mask)
      x['weight'] = weights
    return x, y
  