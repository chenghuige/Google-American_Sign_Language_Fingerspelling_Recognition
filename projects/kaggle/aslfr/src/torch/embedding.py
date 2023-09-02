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

from torch import nn
from gezi.common import *
from src.config import *
from src.torch.layers import BatchNorm

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
      layers.append(
          tf.keras.layers.BatchNormalization(momentum=0.95,
                                             name='input_batchnorm'))

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


class SimpleEmbedding(nn.Module):

  def __init__(self):
    super(SimpleEmbedding, self).__init__()

    self.emb_batchnorm = True if FLAGS.emb_batchnorm is None else FLAGS.emb_batchnorm
    self.embedding = nn.Linear(get_n_cols(), FLAGS.encoder_units, bias=False)
    if self.emb_batchnorm:
      self.batch_norm = BatchNorm(FLAGS.encoder_units, momentum=0.05, eps=1e-3)

  def forward(self, x):
    x = self.embedding(x)
    if self.emb_batchnorm:
      x = self.batch_norm(x)
    return x


class LearnedPositionEncoding(nn.Embedding):

  def __init__(self, num_embeddings, embedding_dim):
    super().__init__(num_embeddings, embedding_dim)

  def forward(self, x):
    weight = self.weight.data.unsqueeze(0)
    x = x + weight
    return x


class PositionEmbeddingV2(nn.Module):

  def __init__(self):
    super(PositionEmbeddingV2, self).__init__()
    self.positional_encode = LearnedPositionEncoding(FLAGS.n_frames,
                                                     FLAGS.encoder_units)
    self.positional_encode.weight.data.fill_(0.)
    # Embedding layer for Landmarks
    self.embedding = SimpleEmbedding()

  def forward(self, x):
    x = self.embedding(x)
    x = self.positional_encode(x)
    return x


Embeddings = {
    'landmark': LandmarkEmbedding,
    'positional': PositionEmbedding,
    'simple': SimpleEmbedding,
    'positionalv2': PositionEmbeddingV2,
}


def get_embeddding():
  return Embeddings[FLAGS.embedding]()
