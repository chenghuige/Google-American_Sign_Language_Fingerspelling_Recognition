#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   rnn.py
#        \author   chenghuige  
#          \date   2023-07-06 20:51:37.888389
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
import melt as mt
from src.config import *
from src.tf.util import *  
from src.tf.layers import CrossAttention

class Encoder(tf.keras.layers.Layer):
  def __init__(self):
    super(Encoder, self).__init__()
    self.units = FLAGS.encoder_units

    # The RNN layer processes those vectors sequentially.
    self.rnn = tf.keras.Sequential(
        [
          tf.keras.layers.Dropout(FLAGS.rnn_drop),
          tf.keras.layers.Bidirectional(
            merge_mode='sum',
            layer=tf.keras.layers.GRU(self.units,
                                # Return the sequence and state
                                return_sequences=True,
                                recurrent_initializer='glorot_uniform')),
          tf.keras.layers.Bidirectional(
            merge_mode='sum',
            layer=tf.keras.layers.GRU(self.units,
                                # Return the sequence and state
                                return_sequences=True,
                                recurrent_initializer='glorot_uniform')),
          tf.keras.layers.Bidirectional(
            merge_mode='sum',
            layer=tf.keras.layers.GRU(self.units,
                                # Return the sequence and state
                                return_sequences=True,
                                recurrent_initializer='glorot_uniform')),
          tf.keras.layers.Bidirectional(
            merge_mode='sum',
            layer=tf.keras.layers.GRU(self.units,
                                # Return the sequence and state
                                return_sequences=True,
                                recurrent_initializer='glorot_uniform')),
        ])

  def call(self, x, x_inp):
    # shape_checker = ShapeChecker()
    # shape_checker(x, 'batch s')
    
    # 3. The GRU processes the sequence of embeddings.
    x = self.rnn(x)
    # shape_checker(x, 'batch s units')

    # 4. Returns the new sequence of embeddings.
    return x
  
class Decoder(tf.keras.layers.Layer):
  def __init__(self):
    super(Decoder, self).__init__()
    self.start_token = SOS_TOKEN
    self.end_token = EOS_TOKEN
    self.units = FLAGS.decoder_units

    # # 1. The embedding layer converts token IDs to vectors
    # self.embedding = tf.keras.layers.Embedding(self.vocab_size,
    #                                            units, mask_zero=True)
    # Character Embedding
    self.embedding = tf.keras.layers.Embedding(
        VOCAB_SIZE, units, embeddings_initializer=FLAGS.emb_init)

    # 2. The RNN keeps track of what's been generated so far.
    self.rnn = tf.keras.layers.GRU(units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

    # 3. The RNN output will be the query for the attention layer.
    self.attention = CrossAttention(units)

  def call(self,
          context, x,
          state=None,
          return_state=False):  
    shape_checker = ShapeChecker()
    shape_checker(x, 'batch t')
    shape_checker(context, 'batch s units')

    # 1. Lookup the embeddings
    x = self.embedding(x)
    shape_checker(x, 'batch t units')

    # 2. Process the target sequence.
    x, state = self.rnn(x, initial_state=state)
    shape_checker(x, 'batch t units')

    # 3. Use the RNN output as the query for the attention over the context.
    x = self.attention(x, context)
    self.last_attention_weights = self.attention.last_attention_weights
    shape_checker(x, 'batch t units')
    shape_checker(self.last_attention_weights, 'batch t s')

    if return_state:
      return x, state
    else:
      return x
    
  def get_initial_state(self, context):
    batch_size = tf.shape(context)[0]
    start_tokens = tf.fill([batch_size, 1], self.start_token)
    done = tf.zeros([batch_size, 1], dtype=tf.bool)
    embedded = self.embedding(start_tokens)
    return start_tokens, done, self.rnn.get_initial_state(embedded)[0]
  
  def get_next_token(self, context, next_token, done, state, temperature = 0.0):
    logits, state = self(
      context, next_token,
      state = state,
      return_state=True) 
    
    if temperature == 0.0:
      next_token = tf.argmax(logits, axis=-1)
    else:
      logits = logits[:, -1, :]/temperature
      next_token = tf.random.categorical(logits, num_samples=1)

    # If a sequence produces an `end_token`, set it `done`
    done = done | (next_token == self.end_token)
    # Once a sequence is done it only produces 0-padding.
    next_token = tf.where(done, tf.constant(PAD_TOKEN, dtype=tf.int8), next_token)
    
    return next_token, done, state
    
