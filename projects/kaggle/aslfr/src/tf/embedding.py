#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   embedding.py
#        \author   chenghuige  
#          \date   2023-07-17 09:23:44.686024
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
from src.config import *

# Initiailizers
INIT_HE_UNIFORM = tf.keras.initializers.he_uniform
INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform
# Activations
GELU = tf.keras.activations.gelu

# Embeds a landmark using fully connected layers
class LandmarkEmbedding(tf.keras.Model):

  def __init__(self, name='landmark'):
    super(LandmarkEmbedding, self).__init__(name=f'{name}_embedding')
    self.units = FLAGS.encoder_units
    self.supports_masking = True

  def build(self, input_shape):
    # Embedding for missing landmark in frame, initizlied with zeros
    self.empty_embedding = self.add_weight(
        name=f'{self.name}_empty_embedding',
        shape=[self.units],
        initializer=FLAGS.emb_init,
    )
    layers = [
        tf.keras.layers.Dense(self.units * 2,
                              name=f'{self.name}_dense_1',
                              use_bias=False,
                              kernel_initializer=INIT_GLOROT_UNIFORM,
                              activation=GELU),
        tf.keras.layers.Dense(self.units,
                              name=f'{self.name}_dense_2',
                              use_bias=False,
                              kernel_initializer=INIT_HE_UNIFORM),
    ]
    emb_batchnorm = False if FLAGS.emb_batchnorm is None else FLAGS.emb_batchnorm
    ic(emb_batchnorm)
    if emb_batchnorm:
      layers.append(tf.keras.layers.BatchNormalization(momentum=0.95, name='input_batchnorm'))
    
    self.embedding = tf.keras.Sequential(layers, name=f'{self.name}_dense')

  def call(self, x):
    if FLAGS.dominant_emb:
      return tf.where(
          # Checks whether landmark is missing in frame
          tf.reduce_sum(x, axis=2, keepdims=True) == 0,
          # If so, the empty embedding is used
          self.empty_embedding,
          # Otherwise the landmark data is embedded
          self.embedding(x),
      )
    return self.embedding(x)

# Creates embedding for each frame
class PositionEmbedding(tf.keras.Model):

  def __init__(self):
    super(PositionEmbedding, self).__init__(name='position_embedding')
    self.supports_masking = True

  def build(self, input_shape):
    # Positional embedding for each frame index
    self.positional_embedding = self.add_weight(
        name=f'positional_embedding',
        shape=[FLAGS.n_frames, FLAGS.encoder_units],
        initializer=FLAGS.emb_init,
    )
    # Embedding layer for Landmarks
    self.embedding = LandmarkEmbedding()

  def call(self, x):
    x = self.embedding(x)
    x = x + self.positional_embedding

    return x 
  
class SimpleEmbedding(tf.keras.Model):

  def __init__(self):
    super(SimpleEmbedding, self).__init__(name='simple_embedding')
    self.supports_masking = True

  def build(self, input_shape):
    # Embedding layer for Landmarks
    emb_batchnorm = True if FLAGS.emb_batchnorm is None else FLAGS.emb_batchnorm
    ic(emb_batchnorm)
    if emb_batchnorm:
      self.embedding = tf.keras.Sequential([
        tf.keras.layers.Dense(FLAGS.encoder_units, use_bias=False, name='input_dense'),
        tf.keras.layers.BatchNormalization(momentum=0.95, name='input_batchnorm')
        ], name=f'{self.name}_dense')
    else:
      self.embedding = tf.keras.layers.Dense(FLAGS.encoder_units, use_bias=False, name=f'{self.name}_dense')

  def call(self, x):
    x = self.embedding(x)
    return x
  
class PositionEmbeddingV2(tf.keras.Model):

  def __init__(self):
    super(PositionEmbeddingV2, self).__init__(name='position_embedding_v2')
    self.supports_masking = True

  def build(self, input_shape):
    # Positional embedding for each frame index
    self.positional_embedding = self.add_weight(
        name=f'positional_embedding',
        shape=[FLAGS.n_frames, FLAGS.encoder_units],
        initializer=FLAGS.emb_init,
    )
    # Embedding layer for Landmarks
    self.embedding = SimpleEmbedding()

  def call(self, x):
    x = self.embedding(x)
    x = x + self.positional_embedding

    return x 
  
Embeddings = {
  'landmark': LandmarkEmbedding,
  'positional': PositionEmbedding,
  'simple': SimpleEmbedding,
  'positionalv2': PositionEmbeddingV2,
}

def get_embeddding():
  return Embeddings[FLAGS.embedding]()
