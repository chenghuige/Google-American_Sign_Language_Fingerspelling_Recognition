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
from src.tf.preprocess import PreprocessLayer
from src.tf.util import *
from src.tf.loss import ctc_loss
from src.tf.decode import decode_phrase
from src import util
from src.tf.encoder import Encoder


class Model(mt.Model):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    assert FLAGS.method == 'encode'
    self.encoder = Encoder()
    layers = [
              util.get_cls_dropout(), 
              tf.keras.layers.Dense(get_vocab_size())
            ]
    if FLAGS.cls_mlp:
      layers = [tf.keras.layers.Dense(FLAGS.encoder_units * 2, activation='relu')] + layers
    self.classifer = tf.keras.Sequential(
        layers,
        name='classifier')
    if FLAGS.center_loss_rate > 0:
      self.center_emb = tf.keras.layers.Embedding(
                          get_vocab_size(), 
                          FLAGS.encoder_units)
    self.supports_masking = True

  # TODO check training flag ok
  def encode(self, frames):
    return self.encoder(frames)

  def forward(self, frames):
    x = self.encode(frames)
    self.feature = x
    x = self.classifer(x)
    return x

  def call(self, inputs):
    if FLAGS.work_mode == 'train':
      self.input_ = inputs
    x = self.forward(inputs['frames'])
    return x
  
  @tf.function()
  def infer(self, frames):
    return self.forward(frames)

  def get_loss_fn(self):
    def loss_fn(labels, preds, x):
      weights = None
      if FLAGS.mix_sup:
        weights = x['weight']
      # tf.print(labels.shape, preds.shape)
      loss = ctc_loss(labels, preds, weights=weights)
      if FLAGS.center_loss_rate > 0:
        labels = tf.argmax(preds, axis=-1)
        label_feats = self.center_emb(labels)
        pred_feats = self.feature
        closs = tf.reduce_mean(tf.square(label_feats - pred_feats), axis=-1)
        closs = mt.reduce_over(closs)
        loss += FLAGS.center_loss_rate * closs
      loss *= FLAGS.loss_scale
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


# TFLite model for submission
class TFLiteModel(tf.keras.Model):

  def __init__(self, model):
    super(TFLiteModel, self).__init__()

    # Load the feature generation and main models
    self.preprocess_layer = PreprocessLayer(FLAGS.n_frames)
    self.model = model

  @tf.function(jit_compile=True)
  def infer(self, frames):
    return self.model.infer(frames)

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None, N_COLS], dtype=tf.float32, name='inputs')
  ])
  def call(self, inputs):
    # Preprocess Data
    frames_inp = self.preprocess_layer(inputs)
  
    # Add Batch Dimension
    frames_inp = tf.expand_dims(frames_inp, axis=0)

    outputs = self.infer(frames_inp)
 
    # Squeeze outputs
    outputs = tf.squeeze(outputs, axis=0)
    outputs = decode_phrase(outputs)
    
    # for 0 is PAD_IDX
    outputs -= 1
    outputs = tf.one_hot(outputs, get_vocab_size())
    if FLAGS.decode_phrase_type:
      ouputs = outputs[1:]

    # Return a dictionary with the output tensor
    return {'outputs': outputs}
