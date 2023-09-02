#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige
#          \date   2023-06-26 15:47:48.601207
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
from src.tf.decode import decode_phrase, adjust_pad
from src import util
from src.tf.encoder import Encoder
from src.tf.decoder import Decoder

def random_category_sampling(x):
  l = tf.split(x, x.shape[1], 1)
  l = [
  tf.random.categorical(
      tf.squeeze(a, 1), 1
  )
  for a in l
  ]
  return tf.concat(l, axis=-1)
  
class Model(mt.Model):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    assert FLAGS.method == 'seq2seq'
    self.encoder = Encoder()
    if FLAGS.add_encoder_loss:
      vocab_size = get_vocab_size() if not FLAGS.decode_phrase_type else get_vocab_size() - N_TYPES
      self.ctc_classifer = tf.keras.Sequential(
          [
              util.get_cls_dropout(),
              tf.keras.layers.Dense(vocab_size),
          ],
          name='classifier')
      # self.out_keys = ['ctc_pred']
    # if FLAGS.work_mode == 'train' or FLAGS.use_decoder_output:
    self.decoder = Decoder()
    self.classifer = tf.keras.Sequential(
        [
            util.get_cls_dropout(),
            tf.keras.layers.Dense(get_vocab_size()),
        ],
        name='seq2seq_classifier')
    
    self.supports_masking = True

  def forward(self, frames, phrase):
    x = self.encode(frames)
    if FLAGS.add_encoder_loss:
      self.ctc_pred = self.ctc_classifer(x)
      if FLAGS.loss == 'ctc':
        # TODO 对比下按概率采样结果 以及有一定概率按概率采样结果
        # 另外可以考虑这里也做一定的mask
        if not FLAGS.random_phrase:
          if FLAGS.pad_rate == 1:
            phrase = tf.argmax(self.ctc_pred, -1)            
          else:
            phrase = tf.argmax(adjust_pad(self.ctc_pred), -1)
        else:
          phrase = random_category_sampling(self.ctc_pred)
    x = self.decode(x, phrase)
    return x

  def encode(self, frames):
    x = self.encoder(frames)
    return x

  def decode(self, x, phrase):
    x = self.decoder(x, phrase)
    x = self.classifer(x)
    return x

  ## TODO why could not use tf.function here ? bad results after training a few epochs
  # @tf.function()
  def call(self, inputs, training=False):
    if FLAGS.work_mode == 'train':
      self.input_ = inputs
    frames = inputs['frames']
    phrase = inputs['phrase_']
    if FLAGS.mask_phrase_prob > 0:
      mask = tf.random.uniform((mt.get_shape(phrase, 0), mt.get_shape(phrase, 1)))  > FLAGS.mask_phrase_prob
      mask = tf.cast(mask, phrase.dtype)
      phrase *= mask

    x = self.forward(frames, phrase)

    return x

  def get_loss_fn(self):
    def ctc_loss_fn(y_true, y_pred, x):
      y_true = tf.cast(y_true, tf.int32)
      if FLAGS.decode_phrase_type:
        y_true = y_true[:, 1:]
      weights = None
      if FLAGS.mix_sup:
        weights = x['weight']
      loss = ctc_loss(y_true, y_pred, weights=weights)
      return loss
    
    # not using SparseCategoricalCrossentropy for it lack of label smoothing support...
    loss_obj = tf.keras.losses.CategoricalCrossentropy(
          from_logits=True,
          label_smoothing=FLAGS.label_smoothing,
          reduction='none')
    
    def s2s_loss_fn(y_true, y_pred, x):
      y_pred = tf.cast(y_pred, tf.float32)
      y_true = x['phrase_']
      # One Hot Encode Sparsely Encoded Target Sign
      y_true = tf.cast(y_true, tf.int32)

      mask = y_true != PAD_IDX
      # mask_ = tf.argmax(y_pred, axis=-1) != PAD_IDX
      # mask = tf.math.logical_or(mask, mask_)

      y_true = tf.one_hot(y_true, get_vocab_size(), axis=2)
      y_true = y_true[:, :FLAGS.max_phrase_len, :]
      # y_pred = y_pred[:, :FLAGS.max_phrase_len, :]
      # Categorical Crossentropy with native label smoothing support
      loss = loss_obj(y_true, y_pred)
      
      if FLAGS.weighted_loss:
        weights = tf.tile(tf.range(FLAGS.max_phrase_len)[None], [tf.shape(y_true)[0], 1])
        weights = FLAGS.max_phrase_len + 1 - weights
        weights = tf.cast(weights, tf.float32)
        if FLAGS.log_weights:
          weights = tf.math.log(weights + 1.)
        loss = loss * weights

      if FLAGS.masked_loss:
        loss = mt.masked_loss(loss, mask, reduction='none')
      
      loss = tf.reduce_mean(loss, axis=-1)  
      if FLAGS.mix_sup:
        weight = x['weight']
        if FLAGS.sup_no_s2s:
          weight *= tf.cast(tf.equal(weight, 1.), weight.dtype)
        loss *= weight
      
      loss = mt.reduce_over(loss)
      return loss
    
    def loss_fn(y_true, y_pred, x):
      loss = 0.
      if FLAGS.decoder_loss_rate > 0:
        if FLAGS.loss != 'ctc':
          decoder_loss_fn = s2s_loss_fn
        else:
          decoder_loss_fn = ctc_loss_fn
        decoder_loss = decoder_loss_fn(y_true, y_pred, x)
        loss += FLAGS.decoder_loss_rate * decoder_loss
        
      if FLAGS.add_encoder_loss and FLAGS.encoder_loss_rate > 0:
        encoder_loss = ctc_loss_fn(y_true, self.ctc_pred, x)
        loss += FLAGS.encoder_loss_rate * encoder_loss
      
      loss *= FLAGS.loss_scale
      return loss
      
    loss_fn = self.loss_wrapper(loss_fn)
    return loss_fn

  @tf.function()
  def infer(self, frames):
    x = self.encode(frames)
    
    if FLAGS.loss != 'ctc':
      phrase = tf.fill([tf.shape(frames)[0], FLAGS.max_phrase_len], SOS_IDX)
      for idx in tf.range(FLAGS.max_phrase_len):
        outputs = self.decode(x, phrase)
        phrase = tf.where(
            tf.range(FLAGS.max_phrase_len) < idx + 1,
            tf.argmax(outputs, axis=2, output_type=tf.int32),
            phrase,
        )
      x = tf.one_hot(phrase, get_vocab_size())
      return x
    else:
      x2 = self.ctc_classifer(x)
      phrase = tf.argmax(x2, axis=-1)
      phrase = self.decode(x, phrase)
      return phrase

  
  def get_model(self):
    n_cols = get_n_cols()
    inputs = {
      'frames': tf.keras.layers.Input([FLAGS.n_frames, n_cols],
                                       dtype=tf.float32,
                                       name='frames'),
      'phrase_': tf.keras.layers.Input([FLAGS.max_phrase_len], 
                                       dtype=tf.int32, 
                                       name='phrase_'),
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
  def encode(self, frames):
    return self.model.encode(frames)

  @tf.function(jit_compile=True)
  def infer(self, frames):
    return self.model.infer(frames)

  @tf.function(jit_compile=True)
  def decode(self, x, phrase):
    return self.model.decode(x, phrase)

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None, N_COLS], dtype=tf.float32, name='inputs')
  ])
  def call(self, inputs):
    # Preprocess Data
    frames_inp = self.preprocess_layer(inputs)
    # Add Batch Dimension
    frames_inp = tf.expand_dims(frames_inp, axis=0)
    
    if FLAGS.loss != 'ctc':
      # Get Encoding
      encoding = self.encode(frames_inp)
      
      # Make Prediction
      phrase = tf.fill([1, FLAGS.max_phrase_len], SOS_IDX)
    
      # Predict One Token At A Time
      # stop = False
      stop = tf.constant(False)
      for idx in tf.range(FLAGS.max_phrase_len):
        # Cast phrase to int8
        # phrase = tf.cast(phrase, tf.int8)
        # If EOS token is predicted, stop predicting
        outputs = tf.cond(
            stop, 
            lambda: tf.one_hot(tf.cast(phrase, tf.int32), get_vocab_size()),
            lambda: self.decode(encoding, phrase))
        # phrase = tf.cast(phrase, tf.int32)
        phrase = tf.where(
            tf.range(FLAGS.max_phrase_len) < idx + 1,
            tf.argmax(outputs, axis=2, output_type=tf.int32),
            phrase,
        )

        stop = tf.cond(stop, 
                        lambda: stop, 
                        lambda: phrase[0, idx] == EOS_IDX)
      # Squeeze outputs
      outputs = tf.squeeze(phrase, axis=0)
      mask = tf.logical_and(outputs != SOS_IDX, outputs != EOS_IDX)
      outputs = tf.boolean_mask(outputs, mask)
    else:
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
  
