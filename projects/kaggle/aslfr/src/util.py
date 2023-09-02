#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   util.py
#        \author   chenghuige
#          \date   2023-06-20 08:15:48.863688
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import *
from tensorflow.keras import backend as K

import melt as mt
import husky
import src
from src.config import *
from src.tf.models import (
    classifier,
    encoder,
    seq2seq,
    torch_encoder,
    torch_classifier,
)
from src.tf.encoders import (
    transformer,
    conv1d,
    rnn,
    conv1d_transformer,
    conv1d_rnn,
    conv1d_transformer_rnn,
    conv1d_transformer_rnn2,
)
from src.tf.decoders import (
    transformer,
)

from src.torch.models import (
  encoder,
  classifier,
)
from src.torch.encoders import (
  rnn,
  conv1d,
  conformer,
  conformer_v2,
  conformer_v3,
  conformer_v4,
  conformer_v5,
  conformer_v5_1,
  conformer_v5_1_1,
  conformer_v5_1_2,
  conformer_v5_1_3,
  conformer_v5_1_4,
  conformer_v5_1_5,
  conformer_v5_1_6,
  conformer_v5_1_7,
  conformer_v5_1_8,
  conformer_v5_1_9,
  conformer_v5_1_10,
  conformer_v5_1_11,
  conformer_v5_1_12,
  conformer_v5_2,
  conformer_v5_3,
  conformer_v5_4,
  conformer_v6,
  conformer_v7,
  conv1d_transformer,
  conv1d_transformer_v2,
  conv1d_transformer_v3,
  conformer2,
  conformer3,
  conformer_openspeech,
  squeezeformer,
  squeezeformer2,
  squeezeformer3,
  squeezeformer4,
)

"""
Attempt to retrieve phrase type
Could be used for pretraining or type specific inference
 *) Phone Number\
 *) URL
 *3) Addres
"""


def get_phrase_type(phrase):
  if FLAGS.obj == 'sup':
    phrase == 'sup'
  # Phone Number
  if re.match(r'^[\d+-]+$', phrase):
    return 'phone'
  # url
  elif any([substr in phrase for substr in ['www', '.', '/']
           ]) and ' ' not in phrase:
    return 'url'
  # Address
  else:
    return 'address'


# Custom callback to update weight decay with learning rate
class WeightDecayCallback(tf.keras.callbacks.Callback):

  def __init__(self, model, wd_ratio=0.05):
    self.step_counter = 0
    self.wd_ratio = wd_ratio
    self.model = model
    ic(self.wd_ratio)

  def on_epoch_begin(self, epoch, logs=None):
    lr = mt.get_lr(self.model.optimizer)
    # ic(lr)
    self.model.optimizer.weight_decay = lr * self.wd_ratio
    # print(
    #     f'learning rate: {model.optimizer.learning_rate.numpy():.2e}, weight decay: {model.optimizer.weight_decay.numpy():.2e}'
    # )
    
class FreezeEncoderCallback(tf.keras.callbacks.Callback):
  def __init__(self, model):
    self.model = model
    
  def on_train_begin(self, logs=None):
    self.model.encoder.trainable = False

def get_callbacks(model):
  callbacks = []
  if not FLAGS.torch and FLAGS.keras:
    if FLAGS.wd_ratio:
      callbacks.append(WeightDecayCallback(model, FLAGS.wd_ratio))
    if FLAGS.sie2:
      callbacks.append(husky.callbacks.SaveIntervalModelsCallback(FLAGS.sie2))
    if FLAGS.freeze_encoder:
      callbacks.append(FreezeEncoderCallback(model))  
  return callbacks

def get_cls_dropout():
  if FLAGS.cls_late_drop == 0:
    return tf.keras.layers.Dropout(FLAGS.cls_drop)
  else:
    return mt.layers.LateDropout(late_rate=FLAGS.cls_late_drop,
                                 early_rate=FLAGS.cls_drop,
                                 start_epoch=FLAGS.latedrop_start_epoch)


# TopK accuracy for multi dimensional output
class TopKAccuracy(tf.keras.metrics.Metric):

  def __init__(self, k, **kwargs):
    super(TopKAccuracy, self).__init__(name=f'top{k}acc', **kwargs)
    self.top_k_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=k)

  def update_state(self, y_true, y_pred, sample_weight=None):
    # if K.learning_phase():
    #   return
    if FLAGS.decode_phrase_type:
      y_true = y_true[:, 1:]
      y_pred = y_pred[:, 1:]
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1, get_vocab_size()])
    character_idxs = tf.where(y_true != PAD_IDX)
    y_true = tf.gather(y_true, character_idxs, axis=0)
    y_pred = tf.gather(y_pred, character_idxs, axis=0)
    self.top_k_acc.update_state(y_true, y_pred)

  def result(self):
    return self.top_k_acc.result()

  def reset_state(self):
    self.top_k_acc.reset_state()


# TODO topk acc and edit distance metric not very suitable for ctc loss as like adjacent aa -> a
class EditDistance(tf.keras.metrics.Metric):

  def __init__(self, name='distance', dtype=None):
    super(EditDistance, self).__init__(name, dtype=dtype)
    self.acc_metric = tf.keras.metrics.Mean(name="distance")

  def update_state(self, y_true, y_pred, sample_weight=None):
    # if K.learning_phase():
    #   return
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)

    def mask_fn(x):
      # mask = tf.cast(x == 0, tf.int32)
      # # turn 0 to -1
      # x = x * (1 - mask) + mask * (-1)
      # turn non ori vocab chars like EOS PHRASE_TYPES to 0
      # mask = tf.cast(x >= N_CHARS, tf.int32)
      ## 0 PAD, 1 - N_CHARS (original chars) N_CHARS + 1 for EOS, N_CHARS + 2, 3, 4 for PHRASE_TYPES
      mask = tf.cast(x > N_CHARS, tf.int32)
      x = x * (1 - mask)
      return x

    y_true, y_pred = mask_fn(y_true), mask_fn(y_pred)
    edit_dist = tf.edit_distance(tf.sparse.from_dense(y_pred),
                                 tf.sparse.from_dense(y_true),
                                 normalize=False)
    edit_dist = tf.reduce_mean(edit_dist)
    self.acc_metric.update_state(edit_dist)

  def result(self):
    return self.acc_metric.result()

  def reset_state(self):
    self.acc_metric.reset_state()


def get_metrics():
  if FLAGS.task == 'seq':
    return [
        TopKAccuracy(1),
        TopKAccuracy(5),
        EditDistance(),
    ]


def get_model():
  if not (FLAGS.torch or FLAGS.torch2tf):
    Model = getattr(src.tf.models, FLAGS.model).Model
  else:
    Model = getattr(src.torch.models, FLAGS.model).Model
  model = Model()
  if FLAGS.torch2tf:
    from src.torch.util import torch2keras
    model = torch2keras(model) 
    # TODO
    if FLAGS.task == 'seq':
      Model = getattr(src.tf.models, 'torch_encoder').Model 
    else:
      Model = getattr(src.tf.models, 'torch_classifier').Model 
    model = Model(model)
  return model


def get_encoder():
  if not (FLAGS.torch or FLAGS.torch2tf):
    Encoder = getattr(src.tf.encoders, FLAGS.encoder).Encoder
  else:
    Encoder = getattr(src.torch.encoders, FLAGS.encoder).Encoder
  return Encoder()


def get_decoder():
  if not (FLAGS.torch or FLAGS.torch2tf):
    Decoder = getattr(src.tf.decoders, FLAGS.decoder).Decoder
  else:
    Decoder = getattr(src.torch.decoders, FLAGS.decoder).Decoder
  return Decoder()

def prepare_tflite(model):
  if FLAGS.torch:
    from src.torch.util import torch2keras
    model = torch2keras(model)  
  model.save_weights(f'{FLAGS.model_dir}/tflite.h5')
  return model

def get_tflite_model(model):
  model = prepare_tflite(model)
  if not FLAGS.torch:
    TFLiteModel = getattr(src.tf.models, FLAGS.model).TFLiteModel
  else:
    TFLiteModel = getattr(src.torch.models, FLAGS.model).TFLiteModel
  tflite_keras_model = TFLiteModel(model)
  return tflite_keras_model

def to_tflite_model(model):
  tflite_keras_model = get_tflite_model(model)
  # tflite_keras_model = tflite_keras_model.get_model()
  # plot_model(tflite_func_model, to_file=f'{FLAGS.model_dir}/tflite.png', show_shapes=True, show_layer_names=True)
  ic(tflite_keras_model)
  # Create Model Converter
  converter = tf.lite.TFLiteConverter.from_keras_model(tflite_keras_model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_types = [tf.float16]
  #  1/4 size but seems hurt acc a lot like 0.8 to 0.7
  # converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
  #                                        tf.lite.OpsSet.TFLITE_BUILTINS]
  converter._experimental_default_to_single_batch_in_tensor_list_ops = True
  
  # converter.post_training_quantize = True
  ic(converter)
  ## converter.experimental_new_converter = True
  # Convert Model
  tflite_model = converter.convert()
  # ic(tflite_model)
  # Write Model
  with open(f'{FLAGS.model_dir}/model.tflite', 'wb') as f:
    f.write(tflite_model)

  gezi.system(f'du -h {FLAGS.model_dir}/model.tflite')
  
  # Add selected_columns json to only select specific columns from input frames
  gezi.system(f'cp {FLAGS.root}/inference_args.json {FLAGS.model_dir}')
  # gezi.system(f'cd {FLAGS.model_dir};zip submission.zip model.tflite inference_args.json')
  gezi.system(f'cd {FLAGS.model_dir};mkdir -p ./ckpt;cp model.tflite inference_args.json metrics.csv dataset-metadata.json ./ckpt')
  
  gezi.system(f'./infer.py {FLAGS.model_dir} {int(FLAGS.group_fold)} {FLAGS.fold} {FLAGS.n_infers}')
  tflite_score = gezi.read_float_from(f'{FLAGS.model_dir}/tflite_score.txt')
  head_score = gezi.get('score/head')
  diff = tflite_score - head_score
  abs_diff = abs(diff)
  ic(tflite_score, head_score, diff, abs_diff)
  assert abs_diff < 0.001

def check_masking(model):
  #check supports_masking
  for x in model.layers:
    ic(x.name, x.supports_masking)
    assert x.supports_masking

def verify_correct_training_flag(model, batch):
  # Verify static output for inference
  pred = model(batch, training=False)
  for _ in tqdm(range(10)):
    assert tf.reduce_min(
        tf.cast(pred == model(batch, training=False), tf.int8)) == 1

  # Verify at least 99% varying output due to dropout during training
  for _ in tqdm(range(10)):
    assert tf.reduce_mean(
        tf.cast(pred != model(batch, training=True), tf.float32)) > 0.99

# from NEMO
from typing import List

def compute_stochastic_depth_drop_probs(
    num_layers: int,
    stochastic_depth_drop_prob: float = 0.0,
    stochastic_depth_mode: str = "linear",
    stochastic_depth_start_layer: int = 1,
) -> List[float]:
    """Computes drop probabilities for stochastic depth regularization technique.
    The first layer is never dropped and the starting layer needs to be greater
    or equal to 1.

    Args:
        num_layers (int): number of layers in the network.
        stochastic_depth_drop_prob (float): if non-zero, will randomly drop
            layers during training. The higher this value, the more often layers
            are dropped. Defaults to 0.0.
        stochastic_depth_mode (str): can be either "linear" or "uniform". If
            set to "uniform", all layers have the same probability of drop. If
            set to "linear", the drop probability grows linearly from 0 for the
            first layer to the desired value for the final layer. Defaults to
            "linear".
        stochastic_depth_start_layer (int): starting layer for stochastic depth.
            All layers before this will never be dropped. Note that drop
            probability will be adjusted accordingly if mode is "linear" when
            start layer is > 1. Defaults to 1.
    Returns:
        List[float]: list of drop probabilities for all layers
    """
    if not (0 <= stochastic_depth_drop_prob < 1.0):
        raise ValueError("stochastic_depth_drop_prob has to be in [0, 1).")
    if not (1 <= stochastic_depth_start_layer <= num_layers):
        raise ValueError("stochastic_depth_start_layer has to be in [1, num layers].")

    # Layers before `stochastic_depth_start_layer` are never dropped
    layer_drop_probs = [0.0] * stochastic_depth_start_layer

    # Layers starting with `stochastic_depth_start_layer` may be dropped
    if (L := num_layers - stochastic_depth_start_layer) > 0:
        if stochastic_depth_mode == "linear":
            # we start with 1/L * drop_prob and and end with the desired drop probability.
            layer_drop_probs += [l / L * stochastic_depth_drop_prob for l in range(1, L + 1)]
        elif stochastic_depth_mode == "uniform":
            layer_drop_probs += [stochastic_depth_drop_prob] * L
        else:
            raise ValueError(
                f'stochastic_depth_mode has to be one of ["linear", "uniform"]. Current value: {stochastic_depth_mode}'
            )
    return layer_drop_probs