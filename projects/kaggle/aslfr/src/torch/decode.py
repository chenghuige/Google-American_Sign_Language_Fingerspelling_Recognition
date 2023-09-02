#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   decode.py
#        \author   chenghuige  
#          \date   2023-07-13 08:50:20.755764
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
from src.config import *

def simple_decode(pred):
  x = tf.argmax(pred, axis=-1)
  return x

# @tf.function()
# def ctc_decode(pred):
#   x = tf.nn.ctc_greedy_decoder(pred, sequence_length=[FLAGS.max_len] * FLAGS.batch_size, merge_repeated=True)
#   x = x[0][0]
#   x = tf.sparse.to_dense(x)
#   return x

# @tf.function()
# def ctc_beamsearch_decode(pred):
#   x = tf.nn.ctc_beam_search_decoder(pred, sequence_length=[FLAGS.max_len] * FLAGS.batch_size, beam_width=10, top_paths=1)
#   x = x[0][0]
#   x = tf.sparse.to_dense(x)
#   return x

def adjust_pad(pred):
  pred = tf.nn.softmax(pred, axis=-1)
  pred0, pred1 = pred[..., 0:1], pred[..., 1:]
  pred0 *= FLAGS.pad_rate
  pred = tf.concat([pred0, pred1], axis=-1)
  return pred

def ctc_decode(x):
  diff = tf.not_equal(x[:-1], x[1:])
  adjacent_indices = tf.where(diff)[:, 0]
  x = tf.gather(x, adjacent_indices)
  mask = x != PAD_IDX
  x = tf.boolean_mask(x, mask, axis=0)
  return x

@tf.function()
def decode_phrase(pred):
  pred = adjust_pad(pred)
    
  if FLAGS.loss == 'ctc':
    if FLAGS.ctc_decode_method.startswith('tf-'):
      pred = pred[None]
      input_len = np.ones(pred.shape[0]) * pred.shape[1]
      # batch move to dim 1, [max_time, batch_size, num_classes]. The logits.
      pred = tf.transpose(pred, perm=[1, 0, 2])
      pred = tf.concat([pred[:,:,1:], pred[:,:,0:1]], axis=-1)
      if FLAGS.ctc_decode_method.endswith('-greedy'):
        x = tf.nn.ctc_greedy_decoder(pred, sequence_length=input_len, merge_repeated=True)
      else: #like tf-beam-10 or tf-10
        beam_width = int(FLAGS.ctc_decode_method.split('-')[-1])
        x = tf.nn.ctc_beam_search_decoder(pred, sequence_length=input_len, beam_width=beam_width, top_paths=1)
      x = x[0][0]
      x = tf.sparse.to_dense(x)
      x = x[0]
      x = tf.cast(x, tf.int32)
      # before treat vocab-1 as blank, now treat vocab-0 as blank
      x += 1
      mask = int(x != get_vocab_size())
      x *= mask
      # tf.print(x)
      return x
  
  x = tf.argmax(pred, axis=-1)
  if FLAGS.loss != 'ctc':
    return x
 
  # CTC decode 
  if FLAGS.ctc_loss != 'simple':
    x = ctc_decode(x)
  else:
    x = tf.boolean_mask(x, x != PAD_IDX, axis=0)
  return x  
