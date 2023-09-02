#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   classifier.py
#        \author   chenghuige  
#          \date   2023-07-14 20:20:22.914615
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
import melt as mt
from src.config import *
from src import util
from src.tf.encoder import Encoder

class Model(mt.Model):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.encoder = Encoder()
    self.char_classifer = tf.keras.Sequential(
        [
            mt.layers.Pooling(FLAGS.cls_pooling, name='char_pooling'),
             util.get_cls_dropout(),
            tf.keras.layers.Dense(
                N_CHARS,
                activation=None),
        ],
        name='char_classifier')
    self.type_classifer = tf.keras.Sequential(
        [
            mt.layers.Pooling(FLAGS.cls_pooling, name='type_pooling'),
            util.get_cls_dropout(),
            tf.keras.layers.Dense(
                len(CLASSES),
                activation=None),
        ],
        name='type_classifier')
    self.first_char_classifer = tf.keras.Sequential(
        [
            mt.layers.Pooling(FLAGS.cls_pooling, name='first_char_pooling'),
            util.get_cls_dropout(),
            tf.keras.layers.Dense(
                N_CHARS,
                activation=None),
        ],
        name='first_char_classifier')
    self.last_char_classifer = tf.keras.Sequential(
        [
            mt.layers.Pooling(FLAGS.cls_pooling, name='last_char_pooling'),
            util.get_cls_dropout(),
            tf.keras.layers.Dense(
                N_CHARS,
                activation=None),
        ],
        name='last_char_classifier')
    assert FLAGS.len_cls
    self.len_classifier = tf.keras.Sequential(
        [
            mt.layers.Pooling(FLAGS.cls_pooling, name='len_pooling'),
            util.get_cls_dropout(),
            tf.keras.layers.Dense(MAX_PHRASE_LEN),
        ],
        name='len_regressor')

    
    self.eval_keys = ['phrase_type_', 'first_char', 'last_char',
                      'sequence_id', 'phrase_type', 'phrase_len', 'phrase', 'idx']
    self.out_keys = ['pred', 'type_pred', 'first_char_pred', 'last_char_pred', 'len_pred']
    self.supports_masking = True
    
  def call(self, inputs):
    if FLAGS.work_mode == 'train':
      self.input_ = inputs
    frames = inputs['frames']
    x = self.encoder(frames)
    pred = self.char_classifer(x)
    self.pred = pred
    self.type_pred = self.type_classifer(x)
    self.first_char_pred = self.first_char_classifer(x)
    self.last_char_pred = self.last_char_classifer(x)
    self.len_pred = self.len_classifier(x)
    # res = {
    #   'pred': pred,
    #   'type_pred': self.type_pred,
    #   'first_char_pred': self.first_char_pred,
    #   'last_char_pred': self.last_char_pred,
    # }
    # return res
    return pred
    

  def get_loss_fn(self):

    binary_loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    len_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    def loss_fn(y_true, y_pred, x, model):
      # for char accuracy
      loss = binary_loss_obj(y_true, y_pred)
      # loss = binary_loss_obj(y_true, model.pred)
      loss2 = loss_obj(x['phrase_type_'], model.type_pred)
      loss3 = loss_obj(x['first_char'], model.first_char_pred)
      loss4 = loss_obj(x['last_char'], model.last_char_pred)
      loss5 = len_obj(x['phrase_len'] - 1, model.len_pred)
      # TODO how to track for each loss
      if not FLAGS.cls_loss_weights:
        loss = loss + loss2 + loss3 + loss4 + loss5
      else:
        weights = FLAGS.cls_loss_weights
        loss = loss * weights[0] + loss2 * weights[1] + loss3 * weights[2] + loss4 * weights[3] + loss5 * weights[4]
      loss = mt.reduce_over(loss)
      return loss
    
    loss_fn = self.loss_wrapper(loss_fn)
    return loss_fn
  
  def get_model(self):
    n_cols = get_n_cols()
    inputs = {
      'frames': tf.keras.layers.Input([FLAGS.n_frames, n_cols],
                                       dtype=tf.float32,
                                       name='frames'),
    }
    out = self.call(inputs)
    model = tf.keras.models.Model(inputs, out)
    return model
  
