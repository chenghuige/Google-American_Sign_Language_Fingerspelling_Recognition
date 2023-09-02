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
from src.torch.encoder import Encoder
import lele

class InferModel(nn.Module):
  def __init__(self, model, **kwargs):
    super().__init__(**kwargs)
    self.model = model
  
  def forward(self, frames):
    res = self.model.forward_(frames)
    return res
  
class Model(nn.Module):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.encoder = Encoder()
    self.char_classifer = nn.Sequential(
            lele.layers.Pooling(FLAGS.cls_pooling, FLAGS.encoder_units),
            nn.Dropout(FLAGS.cls_drop),
            nn.Linear(FLAGS.encoder_units, N_CHARS)
            )

    self.type_classifer = nn.Sequential(
            lele.layers.Pooling(FLAGS.cls_pooling, FLAGS.encoder_units),
            nn.Dropout(FLAGS.cls_drop),
            nn.Linear(FLAGS.encoder_units, len(CLASSES))
            )
    self.first_char_classifer = nn.Sequential(
      lele.layers.Pooling(FLAGS.cls_pooling, FLAGS.encoder_units),
      nn.Dropout(FLAGS.cls_drop),
      nn.Linear(FLAGS.encoder_units, N_CHARS)
    )
    self.last_char_classifer = nn.Sequential(
      lele.layers.Pooling(FLAGS.cls_pooling, FLAGS.encoder_units),
      nn.Dropout(FLAGS.cls_drop),
      nn.Linear(FLAGS.encoder_units, N_CHARS)
    )
    if not FLAGS.len_cls:
      if not FLAGS.len_loss == 'bce':
        self.len_classifier = nn.Sequential(
          lele.layers.Pooling(FLAGS.cls_pooling, FLAGS.encoder_units),
          nn.Dropout(FLAGS.cls_drop),
          nn.Linear(FLAGS.encoder_units, 1),
          nn.Sigmoid()
        )
      else:
        self.len_classifier = nn.Sequential(
          lele.layers.Pooling(FLAGS.cls_pooling, FLAGS.encoder_units),
          nn.Dropout(FLAGS.cls_drop),
          nn.Linear(FLAGS.encoder_units, 1)        
        )
    else:
      self.len_classifier = nn.Sequential(
        lele.layers.Pooling(FLAGS.cls_pooling, FLAGS.encoder_units),
        nn.Dropout(FLAGS.cls_drop),
        nn.Linear(FLAGS.encoder_units, MAX_PHRASE_LEN),
      )
      
    if FLAGS.keras_init:
      lele.keras_init(self)
    
    self.eval_keys = ['phrase_type_', 'first_char', 'last_char',
                      'sequence_id', 'phrase_type', 'phrase_len', 'phrase', 'idx']
    self.out_keys = ['pred', 'type_pred', 'first_char_pred', 'last_char_pred']
    
  def forward(self, inputs):
    frames = inputs['frames']
    return self.forward_(frames)
  
  def forward_(self, frames):
    x = self.encoder(frames)
    pred = self.char_classifer(x)
    self.pred = pred
    self.type_pred = self.type_classifer(x)
    self.first_char_pred = self.first_char_classifer(x)
    self.last_char_pred = self.last_char_classifer(x)
    self.len_pred = self.len_classifier(x)
    if not FLAGS.len_cls:
      self.len_pred = self.len_pred.squeeze(-1)
    res = {
      'pred': pred,
      'type_pred': self.type_pred,
      'first_char_pred': self.first_char_pred,
      'last_char_pred': self.last_char_pred,
      'len_pred': self.len_pred,
    }
    return res    

  def get_loss_fn(self):

    binary_loss_obj = nn.BCEWithLogitsLoss()
    loss_obj = nn.CrossEntropyLoss()
    if not FLAGS.len_cls:
      if FLAGS.len_loss == 'l1':
        len_loss_obj = nn.L1Loss()
      elif FLAGS.len_loss == 'l2':
        len_loss_obj = nn.MSELoss()
      elif FLAGS.len_loss == 'smooth_l1':
        len_loss_obj = nn.SmoothL1Loss()
      elif FLAGS.len_loss == 'bce':
        len_loss_obj = nn.BCEWithLogitsLoss()
      else:
        raise ValueError(FLAGS.len_loss)
    else:
      len_loss_obj = nn.CrossEntropyLoss()
      
    def loss_fn(res, y_true, x):
      # for char accuracy
      loss = binary_loss_obj(res['pred'], y_true.float())
      loss2 = loss_obj(res['type_pred'], x['phrase_type_'].long())
      loss3 = loss_obj(res['first_char_pred'], x['first_char'].long())
      loss4 = loss_obj(res['last_char_pred'], x['last_char'].long())
      if not FLAGS.len_cls:
        loss5 = len_loss_obj(res['len_pred'], x['phrase_len'] / MAX_PHRASE_LEN)
      else:
        loss5 = loss_obj(res['len_pred'], x['phrase_len'].long() - 1)
      # TODO how to track for each loss
      if not FLAGS.cls_loss_weights:
        loss = loss + loss2 + loss3 + loss4 + loss5
      else:
        weights = FLAGS.cls_loss_weights
        loss = loss * weights[0] + loss2 * weights[1] + loss3 * weights[2] + loss4 * weights[3] + loss5 * weights[4]
      return loss
    
    return loss_fn
  
  def get_infer_model(self):
    return InferModel(self)
