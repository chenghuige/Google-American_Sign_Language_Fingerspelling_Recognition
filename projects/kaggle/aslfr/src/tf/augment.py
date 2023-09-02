#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   augment.py
#        \author   chenghuige  
#          \date   2023-07-26 23:57:52.313671
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
from gezi.common import *
import melt as mt
from src.config import *

# TODO 一维度展平的输入按照先x再y 这样可以展开[...,2] 处理x y 方便很多 暂时先不改...
def reshape(data):
  data = tf.reshape(data, (1, -1, N_COLS // 3, 3))
  return data

def reshape_back(data):
  data = tf.reshape(data, (1, -1, N_COLS))
  return data

def try_reshape(data):
  if mt.get_shape(data, -1) == N_COLS:
    data = reshape(data)
  return data

def splits(data):
  # l = [
  #     N_HAND_POINTS, N_HAND_POINTS, 
  #     N_POSE_POINTS, N_POSE_POINTS, 
  #     N_LIP_POINTS, N_LIP_POINTS,
  # ]
  l = [
      N_HAND_POINTS, N_HAND_POINTS, 
      N_POSE_POINTS, N_POSE_POINTS, 
      N_LIP_POINTS, N_LIP_POINTS,
      N_EYE_POINTS, N_EYE_POINTS,
      N_NOSE_POINTS, N_NOSE_POINTS,
      N_MID_POINTS, 
  ]
  return tf.split(data, l, axis=-2)

def split(data):
  x, y, z = tf.split(data, 3, axis=-1)
  return x, y, z

def concat(x, y, z):
  data = tf.concat([x, y, z], axis=-1)
  return data

def flip_lr(data):
  x, y, z = split(data)
  x = 1. - x
  data = concat(x, y, z)
  # lhand, rhand, lpose, rpose, llip, rlip = splits(data)
  # data = tf.concat([rhand, lhand, rpose, lpose, rlip, llip], axis=-2)
  lhand, rhand, lpose, rpose, llip, rlip, leye, reye, lnose, rnose, mid = splits(data)
  data = tf.concat([rhand, lhand, rpose, lpose, rlip, llip, reye, leye, rnose, lnose, mid], axis=-2)
  return data

def zero_mean(x, axis=0, keepdims=False):
  upper = tf.reduce_sum(x, axis=axis, keepdims=keepdims)
  bottom = tf.reduce_sum(tf.where(tf.math.equal(x, 0.), tf.zeros_like(x), tf.ones_like(x)), axis=axis, keepdims=keepdims)
  return upper / bottom

def zero_std(x, center=None, axis=0, keepdims=False):
  if center is None:
    center = zero_mean(x, axis=axis, keepdims=True)
  d = x - center
  return tf.math.sqrt(zero_mean(d * d, axis=axis, keepdims=keepdims))

@tf.function()
def interp1d(x, target_len, method='random'):
  target_len = tf.maximum(1, target_len)
  if method == 'random':
    if tf.random.uniform(()) < 0.33:
      x = tf.image.resize(x, (1, target_len), 'bilinear')
    else:
      if tf.random.uniform(()) < 0.5:
        x = tf.image.resize(x, (1, target_len), 'bicubic')
      else:
        x = tf.image.resize(x, (1, target_len), 'nearest')
  else:
    x = tf.image.resize(x, (1, target_len), method)
  x = tf.reshape(x, (1, target_len, -1))
  return x

# x [None, n_frames, n_cols]
def resample(x, rate=(0.5, 1.5)):
  rate = tf.random.uniform((), rate[0], rate[1])
  length = tf.shape(x)[1]
  new_size = tf.cast(rate * tf.cast(length, tf.float32), tf.int32)
  new_x = interp1d(x, new_size)
  return new_x

# x [None, n_frames, n_cols]
def temporal_mask(x):
  mask_prob = FLAGS.temporal_mask_prob if not FLAGS.temporal_mask_range else tf.random.uniform((), *FLAGS.temporal_mask_range)
  mask = tf.random.uniform((mt.get_shape(x, 0), mt.get_shape(x, 1), 1))  > mask_prob
  mask = tf.cast(mask, dtype=x.dtype)
  new_x = x * mask
  return new_x

# x [1, n_frames, n_cols//3, 3]
def temporal_seq_mask(x):
  x = x[0]
  mask_value = float('nan')
  l = tf.shape(x)[0]
  mask_size = tf.random.uniform((), *FLAGS.temporal_seq_mask_range)
  mask_size = tf.cast(tf.cast(l, tf.float32) * mask_size, tf.int32)
  mask_offset = tf.random.uniform((), 0, tf.clip_by_value(l-mask_size,1,l), dtype=tf.int32)
  new_x = tf.tensor_scatter_nd_update(x, 
                                      tf.range(mask_offset, mask_offset+mask_size)[...,None],
                                      tf.fill([mask_size, tf.shape(x)[-2], tf.shape(x)[-1]],
                                      mask_value))
  new_x = new_x[None]
  return new_x

# freq mask as in Specaug https://arxiv.org/pdf/1904.08779.pdf
# def spatio_mask(x):
#   mask = tf.random.uniform((mt.get_shape(x, 0), mt.get_shape(x, 1), mt.get_shape(x, 2)))  > FLAGS.spatio_mask_prob
#   mask = tf.cast(mask, dtype=x.dtype)
#   new_x = x * mask
#   return new_x

# x [None, n_frames, n_cols]
def spatio_mask(x):
  mask = tf.random.uniform((mt.get_shape(x, 0), 1, mt.get_shape(x, 2)))  > FLAGS.spatio_mask_prob
  mask = tf.cast(mask, dtype=x.dtype)
  new_x = x * mask
  return new_x

def shift(data):
  range_ = FLAGS.shift_range
  if FLAGS.shift_method == 0:
    shift_ = tf.random.uniform((),*range_)
    data += shift_
  elif FLAGS.shift_method == 1:
    shift_x = tf.random.uniform((),*range_)
    shift_y = tf.random.uniform((),*range_)
    shift_z = tf.random.uniform((),*range_)
    x, y, z = split(data)
    x += shift_x
    y += shift_y
    z += shift_z
    data = concat(x, y, z)
  else:
    raise ValueError('not supported shift method')
  return data

def scale(data):
  scale_ = tf.random.uniform((), *FLAGS.scale_range)
  if FLAGS.scale_method == 0:
    data *= scale_
  elif FLAGS.scale_method == 1:
    data -= 0.5
    data *= scale_
    data += 0.5
  return data

def rotate(data):
  data -= 0.5
  degree = tf.random.uniform((), *FLAGS.rotate_range)
  radian = degree / 180 * np.pi
  c = tf.math.cos(radian)
  s = tf.math.sin(radian)
  rotate_mat = tf.identity([
            [c,s],
            [-s, c],
        ])
  
  x, y, z = split(data)
  xy = tf.concat([x, y], axis=-1)
  # [None, 88, 2]
  # tf.print(xy.shape)
  xy = xy @ rotate_mat
  
  data = tf.concat([xy, z], axis=-1)  
  data += 0.5
  return data
  
def norm_frame(data):  
  x, y, z = split(data)
  x_mean = zero_mean(x, axis=-1, keepdims=True)
  y_mean = zero_mean(y, axis=-1, keepdims=True)
  z_mean = zero_mean(z, axis=-1, keepdims=True)
  x_std = zero_std(x, center=x_mean, axis=-1, keepdims=True)
  y_std = zero_std(y, center=y_mean, axis=-1, keepdims=True)
  z_std = zero_std(z, center=z_mean, axis=-1, keepdims=True)

  x = x - x_mean / x_std
  y = y - y_mean / y_std
  z = z - z_mean / z_std
  data = concat(x, y, z)
  return data

# [1, n_frames, n_cols//3, 3]
def norm_hands(data):  
  # lhand, rhand, lpose, rpose, llip, rlip = splits(data)
  lhand, rhand, lpose, rpose, llip, rlip, leye, reye, lnose, rnose, mid = splits(data)
  lhand = lhand[:,:,1:] - lhand[:,:,0:1]
  rhand = rhand[:,:,1:] - rhand[:,:,0:1]
  data = tf.concat([lhand, rhand], axis=2)
  if FLAGS.norm_hands_size:
    max_val = tf.math.maximum(tf.reduce_max(tf.abs(data)), 0.0001)
    data /= max_val
  return data

def add_motion(data):
  lhand, rhand, lpose, rpose, llip, rlip, leye, reye, lnose, rnose, mid = splits(data)
  hand = tf.concat([lhand, rhand], axis=2)
  hand_motion = tf.concat([tf.zeros_like(hand[:, :1]), hand[:, 1:] - hand[:, :-1]], axis=1)
  hand_motion2 = tf.concat([tf.zeros_like(hand[:, :2]), hand[:, 2:] - hand[:, :-2]], axis=1)
  return tf.concat([hand_motion, hand_motion2], axis=2)
  
def get_pos(n_frames, dtype):
  pos = tf.range(n_frames, dtype=dtype)
  pos /= 1000.
  pos = tf.reshape(pos, [1, n_frames, 1])
  return pos

def cutmix(frames, n_frames, phrase, phrase_len, 
           frames2, n_frames2, phrase2, phrase2_len):
  # TODO 第一版做最简单的cutmix 第一个phrase从开始出发 第二个phrase 从最后开始截断
  frames = frames[:n_frames]
  frames2 = frames2[:n_frames2]
  phrase = phrase[:phrase_len]
  phrase2 = phrase2[:phrase2_len]
  frac = tf.random.uniform((), 0.5, 0.8)
  n_frames = tf.cast(tf.cast(n_frames, frac.dtype) * frac, tf.int32)
  frames = frames[:n_frames]
  n_frames2 = tf.cast(tf.cast(n_frames2, frac.dtype) * frac, tf.int32)
  frames2 = frames2[n_frames2:]
  phrase_len = tf.cast(tf.cast(phrase_len, frac.dtype) * frac, tf.int32)
  phrase2_len = tf.cast(tf.cast(phrase2_len, frac.dtype) * frac, tf.int32)
  phrase = phrase[:phrase_len]
  phrase2 = phrase2[phrase2_len:]
  frames = tf.concat([frames, frames2], axis=0)
  frames = frames[:FLAGS.n_frames]
  n_frames = tf.shape(frames)[0]
  if n_frames < FLAGS.n_frames:
    frames = tf.concat([frames,
                        tf.zeros([FLAGS.n_frames - n_frames, frames.shape[1]], dtype=frames.dtype)], axis=0)
  phrase = tf.concat([phrase, phrase2], axis=0)
  phrase = phrase[:MAX_PHRASE_LEN]
  phrase_len = tf.shape(phrase)[0]
  if phrase_len < MAX_PHRASE_LEN:
    phrase = tf.concat([phrase, tf.zeros([MAX_PHRASE_LEN - phrase_len], dtype=phrase.dtype)], axis=0)
  return frames, n_frames, phrase, phrase_len
