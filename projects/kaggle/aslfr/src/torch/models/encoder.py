#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model2.py
#        \author   chenghuige
#          \date   2023-07-13 00:12:33.561837
#   \Description  TODO should rename encoder.py to ctc.py
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
from gezi.common import *
import melt as mt
import lele
from src.config import *
from src import util
from src.torch.encoder import Encoder
# for tflite inference
from src.tf.preprocess import PreprocessLayer
from src.tf.decode import decode_phrase

class InferModel(nn.Module):
  def __init__(self, model, **kwargs):
    super().__init__(**kwargs)
    self.model = model
  
  def forward(self, frames):
    res = self.model.infer(frames)
    return res
  
  def infer(self, frames):
    return self.forward(frames)

class Model(nn.Module):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    assert FLAGS.method == 'encode'
    self.encoder = Encoder()
    self.classifer = nn.Sequential(
            nn.Dropout(FLAGS.cls_drop),
            nn.Linear(FLAGS.encoder_units, get_vocab_size()),
        )
    if FLAGS.len_loss_weight:
      self.len_classifier = nn.Sequential(
        lele.layers.Pooling(FLAGS.cls_pooling, FLAGS.encoder_units),
        nn.Dropout(FLAGS.cls_drop),
        nn.Linear(FLAGS.encoder_units, MAX_PHRASE_LEN),
      )
    if FLAGS.center_loss_rate > 0:
      self.center_emb = nn.Embedding(
                          get_vocab_size(), 
                          FLAGS.encoder_units)
      self.center_emb.weight.data.copy_(torch.from_numpy(gezi.load(f'{FLAGS.root}/embs.npy')))

      
    if FLAGS.keras_init:
      lele.keras_init(self)
    
  # TODO check training flag ok
  def encode(self, frames):
    return self.encoder(frames)

  def forward_(self, frames):
    x = self.encode(frames)
    # self.feature = x
    if FLAGS.eval_train or FLAGS.center_loss_rate > 0 or FLAGS.rdrop_key == 'feature':
      gezi.set('feature', x)
    if FLAGS.len_loss_weight:
      len_pred = self.len_classifier(x)
      gezi.set('len_pred', len_pred)
    x = self.classifer(x)
    return x

  def forward(self, inputs):
    if FLAGS.work_mode == 'train':
      self.input_ = inputs
    x = self.forward_(inputs['frames'])
    res = {
      'pred': x,
    }
    if FLAGS.inter_ctc:
      if self.training:
        x_ = gezi.get('inter_ctc_out')
        if x_ is not None:
          x_ = self.classifer(x_)
        # gezi.set('inter_ctc_out', x_)
        res['inter_pred'] = x_
        gezi.set('inter_ctc_out', None)
    if FLAGS.len_loss_weight:
      res['len_pred'] = gezi.get('len_pred')
    if FLAGS.center_loss_rate > 0 or FLAGS.rdrop_key == 'feature':
      res['feature'] = gezi.get('feature')
    return res
  
  def infer(self, frames):
    return self.forward_(frames)

  def get_loss_fn(self):  
    class SmoothCTCLoss(nn.Module):

      def __init__(self, num_classes, blank=0, weight=0.01, reduction='none'):
        super().__init__()
        self.weight = weight
        self.num_classes = num_classes

        self.ctc = nn.CTCLoss(reduction=reduction, blank=blank, zero_infinity=True)
        self.kldiv = nn.KLDivLoss(reduction='batchmean')

      def forward(self, log_probs, targets, input_lengths, target_lengths):
        ctc_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)

        kl_inp = log_probs.transpose(0, 1)
        kl_tar = torch.full_like(kl_inp, 1. / self.num_classes)
        kldiv_loss = self.kldiv(kl_inp, kl_tar)

        #print(ctc_loss, kldiv_loss)
        loss = (1. - self.weight) * ctc_loss + self.weight * kldiv_loss
        return loss
      
    class FocalCTCLoss(nn.Module):

      def __init__(self, alpha=0.5, gamma=0.5, blank=0, num_classes=None, weight=0., reduction='none'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

        self.ctc = nn.CTCLoss(reduction=reduction, blank=blank, zero_infinity=True)
        self.kldiv = nn.KLDivLoss(reduction='batchmean')
        self.weight = weight
        self.num_classes = num_classes

      def forward(self, log_probs, targets, input_lengths, target_lengths):
        loss = self.ctc(log_probs, targets, input_lengths, target_lengths)
        
        if self.weight > 0:
          kl_inp = log_probs.transpose(0, 1)
          kl_tar = torch.full_like(kl_inp, 1. / self.num_classes)
          kldiv_loss = self.kldiv(kl_inp, kl_tar)
          loss = (1. - self.weight) * loss + self.weight * kldiv_loss
          
        p = torch.exp(-loss)
        loss = ((self.alpha)*((1-p)**self.gamma)*(loss))
        return loss
    
      
    def ctc_loss(loss_obj, preds, labels, labels_lengths, weights=None):
      preds = F.log_softmax(preds, dim=-1)
      preds_lengths = torch.sum(torch.ones_like(preds[:,:,0]).long(), dim=-1)
      loss = loss_obj(preds.transpose(0, 1), labels, preds_lengths, labels_lengths)
      if weights is not None:
        if not FLAGS.ctc_torch_loss:
          loss = torch.mean(loss * weights * labels_lengths.float())
        else:
          # this is much better
          loss = torch.mean(loss * weights)
      else:
        if not FLAGS.ctc_torch_loss:
          loss /= labels.shape[0]
      return loss
      
    def loss_fn(res, labels, x, step=None, epoch=None, training=None):
      scalars = {}
      weights = None
      reduction = 'sum' if not FLAGS.ctc_torch_loss else 'mean'
      if FLAGS.mix_sup:
        weights = x['weight']
        reduction = 'none'
      if FLAGS.focal_ctc_loss:
        loss_obj = FocalCTCLoss(alpha=FLAGS.focal_ctc_alpha, gamma=FLAGS.focal_ctc_gamma, 
                                blank=PAD_IDX, num_classes=get_vocab_size(), weight=FLAGS.ctc_label_smoothing,
                                reduction=reduction)
      elif FLAGS.ctc_label_smoothing > 0:
        loss_obj = SmoothCTCLoss(get_vocab_size(), blank=PAD_IDX, weight=FLAGS.ctc_label_smoothing, 
                                 reduction=reduction)
      else:
        loss_obj = nn.CTCLoss(zero_infinity=True, reduction=reduction)
      preds = res['pred'].float()
      labels = labels.float()
      labels_lengths = torch.sum((labels != PAD_IDX).long(), dim=-1)
      loss = ctc_loss(loss_obj, preds, labels, labels_lengths, weights)
      scalars['loss/ctc'] = loss.item()
      
      if FLAGS.inter_ctc:
        inter_preds = res['inter_pred'].float()
        inter_loss = ctc_loss(loss_obj, inter_preds, labels, labels_lengths, weights)
        scalars['loss/inter_ctc'] = inter_loss.item()
        loss += FLAGS.inter_ctc_rate * inter_loss
      
      reduction = 'none'
      if weights is None:
        weights = 1.
      if FLAGS.len_loss_weight:
        len_loss_obj = nn.CrossEntropyLoss(reduction=reduction)
        len_loss = len_loss_obj(res['len_pred'], x['phrase_len'].long() - 1)
        len_loss = (len_loss * weights).mean()
        scalars['loss/len'] = len_loss.item()
        loss += FLAGS.len_loss_weight * len_loss
      
      # TODO
      if FLAGS.center_loss_rate > 0:
        labels = torch.argmax(preds, dim=-1)
        label_feats = self.center_emb(labels)
        pred_feats = res['feature']
        closs = torch.square(label_feats - pred_feats).sum(dim=-1).mean(dim=-1)
        closs = (closs * weights).mean()
        scalars['loss/closs'] = closs.item()
        loss += FLAGS.center_loss_rate * closs
      loss *= FLAGS.loss_scale
      # ic(loss)
      if FLAGS.rdrop_rate > 0:       
        def rdrop_loss(p, q, mask=None):
          # TODO pred or feature?
          key = FLAGS.rdrop_key
          rloss = lele.losses.compute_kl_loss(p[key], q[key], mask=mask)
          scalars['loss/rdrop'] = rloss.item()
          return rloss
        # ic(epoch, FLAGS.rdrop_start_epoch)
        if epoch and epoch >= FLAGS.rdrop_start_epoch:
          if not gezi.get('rdrop_loss_fn'):
            ic(epoch, FLAGS.rdrop_start_epoch)
            gezi.set('rdrop_loss_fn', lambda p, q: rdrop_loss(p, q))
          
      lele.update_scalars(scalars, decay=FLAGS.loss_decay, training=training)
      return loss
    return loss_fn
  
  def get_infer_model(self):
    return InferModel(self)
  
# TFLite model for submission
class TFLiteModel(tf.keras.Model):

  def __init__(self, model):
    super(TFLiteModel, self).__init__()

    # Load the feature generation and main models
    self.preprocess_layer = PreprocessLayer(FLAGS.n_frames)
    self.model = model

  @tf.function(jit_compile=True)
  def infer(self, frames):
    return self.model(frames)

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None, N_COLS], dtype=tf.float32, name='inputs')
  ])
  def call(self, inputs):
    # Preprocess Data
    frames_inp = self.preprocess_layer(inputs)
    # x = frames_inp
    # Add Batch Dimension
    frames_inp = tf.expand_dims(frames_inp, axis=0)

    outputs = self.infer(frames_inp)
    # y = outputs
 
    # Squeeze outputs
    outputs = tf.squeeze(outputs, axis=0)
    outputs = decode_phrase(outputs)
    
    # for 0 is PAD_IDX
    outputs -= 1
    # outputs = tf.one_hot(outputs, get_vocab_size())
    # vocab_size = 61 if not FLAGS.no_eos else 60
    vocab_size = get_vocab_size()
    outputs = tf.one_hot(outputs, vocab_size)
    if FLAGS.decode_phrase_type:
      ouputs = outputs[1:]

    # Return a dictionary with the output tensor
    return {'outputs': outputs}
    # return {
    #   'outputs': outputs,
    #   'frames': x,
    #   'intermediate': y,
    # }

  # TODO not work... an intermediate Keras symbolic input/output, to a TF API that does not allow registering custom dispatchers, such as `tf.cond`, `tf.function`, gradient tapes, or `tf.map_fn`. Keras Functional model construction only supports TF API calls that *do* support dispatching, such as `tf.math.add` or `tf.reshape`. Other APIs cannot be called directly on symbolic Kerasinputs/outputs. You can work around this limitation by putting the operation in a custom Keras layer `call` and calling that layer on this symbolic input/output.
  def get_model(self):
    inputs = tf.keras.layers.Input([N_COLS],
                                    dtype=tf.float32,
                                    name='inputs')
    out = self.call(inputs)
    model = tf.keras.models.Model(inputs, out)
    model.summary()
    return model
