#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   util.py
#        \author   chenghuige
#          \date   2023-06-20 08:15:48.863688
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import *
import melt as mt
from src.config import *
"""
Attempt to retrieve phrase type
Could be used for pretraining or type specific inference
 *) Phone Number\
 *) URL
 *3) Addres
"""


def get_phrase_type(phrase):
  # Phone Number
  if re.match(r'^[\d+-]+$', phrase):
    return 'phone'
  # url
  elif any([substr in phrase for substr in ['www', '.', '/']
           ]) and ' ' not in phrase:
    return 'url'
  # Address
  else:
    return 'address'


# Custom callback to update weight decay with learning rate
class WeightDecayCallback(tf.keras.callbacks.Callback):

  def __init__(self, model, wd_ratio=0.05):
    self.step_counter = 0
    self.wd_ratio = wd_ratio
    self.model = model

  def on_epoch_begin(self, epoch, logs=None):
    lr = mt.get_lr(self.model.optimizer)
    # ic(lr)
    self.model.optimizer.weight_decay = lr * self.wd_ratio
    # print(
    #     f'learning rate: {model.optimizer.learning_rate.numpy():.2e}, weight decay: {model.optimizer.weight_decay.numpy():.2e}'
    # )


def get_callbacks(model):
  callbacks = []
  if FLAGS.wd_ratio:
    callbacks.append(WeightDecayCallback(model, FLAGS.wd_ratio))
  return callbacks


# TopK accuracy for multi dimensional output
class TopKAccuracy(tf.keras.metrics.Metric):

  def __init__(self, k, **kwargs):
    super(TopKAccuracy, self).__init__(name=f'top{k}acc', **kwargs)
    self.top_k_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=k)

  def update_state(self, y_true, y_pred, sample_weight=None):
    if FLAGS.decode_phrase_type:
      y_true = y_true[:, 1:]
      y_pred = y_pred[:, 1:]
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1, VOCAB_SIZE])
    character_idxs = tf.where(y_true != PAD_TOKEN)
    y_true = tf.gather(y_true, character_idxs, axis=0)
    y_pred = tf.gather(y_pred, character_idxs, axis=0)
    self.top_k_acc.update_state(y_true, y_pred)

  def result(self):
    return self.top_k_acc.result()

  def reset_state(self):
    self.top_k_acc.reset_state()
    
def get_metrics():
  if not FLAGS.obj:
    return [
              TopKAccuracy(1),
              TopKAccuracy(5),
            ]
