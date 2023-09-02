#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   loss.py
#        \author   chenghuige  
#          \date   2023-07-13 09:32:53.726596
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import *
import melt as mt
from src.config import * 
import tensorflow as tf
from tf_seq2seq_losses import classic_ctc_loss, simplified_ctc_loss

def ctc_loss(labels, logits, weights=None):

  label_length = tf.reduce_sum(tf.cast(labels != PAD_IDX, tf.int32), axis=-1)
  logit_length = tf.ones(tf.shape(logits)[0],
                         dtype=tf.int32) * tf.shape(logits)[1]
  
  ctc_losses = {
      'ori': tf.nn.ctc_loss,
      'classic': classic_ctc_loss,
      'simple': simplified_ctc_loss,
  }
  ctc_loss_fn = ctc_losses[FLAGS.ctc_loss]

  kwargs = {}
  if ctc_loss_fn == tf.nn.ctc_loss:
    kwargs['logits_time_major'] = False 
  loss = ctc_loss_fn(labels=tf.cast(labels, tf.int32),
                     logits=logits,
                     label_length=label_length,
                     logit_length=logit_length,
                     blank_index=PAD_IDX,
                     **kwargs)
  if weights is not None:
    loss *= weights
  loss = mt.reduce_over(loss)
  return loss  

# CTC and Transducer  attention? Transducer TODO
# TODO https://github.com/TeaPoly/CTC-OptimizedLoss/blob/main/ctc_label_smoothing_loss.py
def ctc_label_smoothing_loss(logits, ylens):
    """Compute KL divergence loss for label smoothing of CTC models.

    Args:
        logits (Tensor): `[B, T, vocab]`
        ylens (Tensor): `[B]`
    Returns:
        loss_mean (Tensor): `[1]`

    """
    bs, max_ylens, vocab = logits.shape

    log_uniform = tf.zeros_like(logits) + tf.math.log(1/(vocab-1))
    probs = tf.nn.softmax(logits)
    log_probs = tf.nn.log_softmax(logits)
    loss = tf.math.multiply(probs, log_probs - log_uniform)
    ylens_mask = tf.sequence_mask(
        ylens, maxlen=max_ylens, dtype=logits.dtype)
    loss = tf.math.reduce_sum(loss, axis=-1)
    loss_mean = tf.math.reduce_sum(
        loss*ylens_mask)/tf.cast(tf.math.reduce_sum(ylens), dtype=logits.dtype)
    return loss_mean
