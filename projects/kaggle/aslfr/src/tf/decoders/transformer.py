#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   decoder.py
#        \author   chenghuige  
#          \date   2023-07-15 11:30:52.858734
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
from src.config import *
from src.tf.layers import *
  
# TODO teacher forcing 
# Decoder based on multiple transformer blocks
class Decoder(tf.keras.Model):

  def __init__(self):
    super().__init__(name='transformer_decoder')
    self.num_blocks = FLAGS.decoder_layers
    self.supports_masking = True

  def build(self, input_shape):
    self.positional_embedding = self.add_weight(
        name=f'positional_embedding',
        shape=[FLAGS.encode_out_feats, FLAGS.decoder_units],
        initializer=FLAGS.emb_init,
    )
    # Character Embedding
    self.char_emb = tf.keras.layers.Embedding(
        get_vocab_size(), FLAGS.decoder_units, embeddings_initializer=FLAGS.emb_init)
    # Positional Encoder MHA
    self.pos_emb_mha = MultiHeadAttention(FLAGS.decoder_units, FLAGS.mhatt_heads, FLAGS.mhatt_drop)
    self.pos_emb_ln = tf.keras.layers.LayerNormalization(
        epsilon=FLAGS.layer_norm_eps)
    # First Layer Normalisation
    self.ln_1s = []
    self.mhas = []
    self.ln_2s = []
    self.mlps = []
    # Make Transformer Blocks
    for i in range(self.num_blocks):
      # First Layer Normalisation
      self.ln_1s.append(
          tf.keras.layers.LayerNormalization(epsilon=FLAGS.layer_norm_eps))
      # Multi Head Attention
      self.mhas.append(MultiHeadAttention(FLAGS.decoder_units, FLAGS.mhatt_heads, FLAGS.mhatt_drop))
      # Second Layer Normalisation
      self.ln_2s.append(
          tf.keras.layers.LayerNormalization(epsilon=FLAGS.layer_norm_eps))
      # Multi Layer Perception
      self.mlps.append(
          tf.keras.Sequential([
              tf.keras.layers.Dense(FLAGS.decoder_units * FLAGS.mlp_ratio,
                                    activation=GELU,
                                    kernel_initializer=INIT_GLOROT_UNIFORM),
              tf.keras.layers.Dropout(FLAGS.mlp_drop),
              tf.keras.layers.Dense(FLAGS.decoder_units,
                                    kernel_initializer=INIT_HE_UNIFORM),
          ]))

  def get_causal_attention_mask(self, B):
    i = tf.range(FLAGS.encode_out_feats)[:, tf.newaxis]
    j = tf.range(FLAGS.encode_out_feats)
    mask = tf.cast(i >= j, dtype=tf.int32)
    mask = tf.reshape(mask, (1, FLAGS.encode_out_feats, FLAGS.encode_out_feats))
    mult = tf.concat(
        [tf.expand_dims(B, -1),
         tf.constant([1, 1], dtype=tf.int32)],
        axis=0,
    )
    mask = tf.tile(mask, mult)
    mask = tf.cast(mask, tf.float32)
    # mask = tf.cast(mask, tf.float32) if not FLAGS.fp16 else tf.cast(mask, tf.float16)
    return mask

  def call(self, encoder_outputs, phrase):
    # Batch Size
    B = tf.shape(encoder_outputs)[0]
    # Cast to INT32
    phrase = tf.cast(phrase, tf.int32)
    if FLAGS.loss != 'ctc':
      phrase = phrase[:, :-1]
      # Prepend SOS Token
      phrase = tf.pad(phrase, [[0, 0], [1, 0]],
                      constant_values=SOS_IDX,
                      name='prepend_sos_token')
      # # Pad With PAD Token
      phrase = tf.pad(phrase,
                      [
                      [0, 0], 
                      [0, FLAGS.encode_out_feats - FLAGS.max_phrase_len]
                      ],
                      constant_values=PAD_IDX,
                      name='append_pad_token')
    # Positional Embedding
    # ic(self.positional_embedding.shape, phrase.shape, self.char_emb(phrase).shape)
    x = self.positional_embedding + self.char_emb(phrase)
    if FLAGS.loss!= 'ctc':
      # Causal Attention
      causal_mask = self.get_causal_attention_mask(B)
      causal_mask = tf.cast(causal_mask, x.dtype)
    else:
      causal_mask = None
    x = self.pos_emb_ln(x +
                        self.pos_emb_mha(x, x, x, attention_mask=causal_mask))
    # Iterate input over transformer blocks
    for ln_1, mha, ln_2, mlp in zip(self.ln_1s, self.mhas, self.ln_2s,
                                    self.mlps):
      x = ln_1(
          x +
          mha(x, encoder_outputs, encoder_outputs, attention_mask=causal_mask))
      x = ln_2(x + mlp(x))
      
    if FLAGS.loss!= 'ctc':
      # Slice 31 Characters
      x = x[:,:FLAGS.max_phrase_len]

    return x
  