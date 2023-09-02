#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   util.py
#        \author   chenghuige
#          \date   2023-07-06 20:49:35.199220
#   \Description
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import *

import tensorflow as tf
import torch
from src.config import *

# import onnx
# import onnx2tf

def torch2onnx(model):
  dummy_input = torch.randn(1, FLAGS.n_frames, get_n_cols()).to('cuda')
  ic(dummy_input.shape)
  input_names = ['input']
  output_names = ['output']
  model = model.get_infer_model()
  torch.onnx.export(model,
                    dummy_input,
                    f'{FLAGS.model_dir}/model.onnx',
                    verbose=False,
                    input_names=input_names,
                    output_names=output_names)

# TODO this is to tflite..
def onnx2tf_():
  pass
  # gezi.system(f'cd {FLAGS.model_dir}; onnx2tf -i model.onnx -osd --disable_group_convolution')

  # tf_model_path = f'{FLAGS.model_dir}/saved_model'
  # # this will get directly tflite model tf_model/model_float16.tflite  model_float32.tflite
  # onnx2tf.convert(
  #   input_onnx_file_path=f'{FLAGS.model_dir}/model.onnx',
  #   output_folder_path=tf_model_path,
  #   copy_onnx_input_output_names_to_tflite=True,
  #   # output_signaturedefs=True,
  #   output_h5=True,
  #   non_verbose=True,
  # )

def onnx2keras_():
  from onnx2keras import onnx_to_keras

  # Load ONNX model
  onnx_model = onnx.load(f'{FLAGS.model_dir}/model.onnx')

  # Call the converter (input - is the main model input name, can be different for your model)
  k_model = onnx_to_keras(onnx_model, ['input'])
  ic(dir(k_model))
  return k_model

# def torch2keras(model):
#   torch2onnx(model)
#   k_model = onnx2keras_()
#   return k_model

def torch2tf(model):
  onnx2tf_()
  # interpreter = tf.lite.Interpreter(model_path=f'{FLAGS.model_dir}/tf_model/model_float32.tflite')
  # tf_lite_model = interpreter.get_signature_runner()
  # interpreter.allocate_tensors()
  # return tf_lite_model
  model = tf.saved_model.load(f'{FLAGS.model_dir}/saved_model')
  logger.info('saved model load done')
  return model

def torch2keras(model):
  import nobuco
  from nobuco import ChannelOrder, ChannelOrderingStrategy
  from nobuco.layers.weight import WeightLayer
  input_shape = [1, FLAGS.n_frames, get_n_cols()]
  ic(input_shape)
  dummy_input = torch.randn(input_shape)
  model = model.get_infer_model()
  # # https://www.kaggle.com/competitions/asl-signs/discussion/406411
  # model = model.half().float()
  model = model.to('cpu')
  model = model.eval()
  
  ## TODO help merge to nobuco codebase
  # for this you could use x = F.avg_pool1d(x, FLAGS.encode_pool_size)
  @nobuco.converter(F.avg_pool1d, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER)
  def converter_avg_pool1d(input: torch.Tensor, input2: int, inplace: bool = False):
    return lambda input, input2, inplace=False: tf.keras.layers.AveragePooling1D(input2)(input)
  
  @nobuco.converter(nn.AvgPool1d, channel_ordering_strategy=ChannelOrderingStrategy.FORCE_TENSORFLOW_ORDER)
  def converter_AvgPool1d(self, input: torch.Tensor):
    return tf.keras.layers.AveragePooling1D(self.kernel_size)
    
  @nobuco.converter(torch.ones_like, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
  def converter_ones_like(input, *args, **kwargs):
      def func(input, *args, **kwargs):
          return tf.ones_like(input)
      return func
  
  @nobuco.converter(torch.Tensor.bmm, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS_OR_PYTORCH, autocast=True)
  def converter_bmm(self, value, *args, **kwargs):
    def func(self, value, *args, **kwargs):
        return tf.matmul(self, value)
    return func
  
  @nobuco.converter(torch.flip, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
  def converter_flip(input, input2, *args, **kwargs):
      def func(input, input2, *args, **kwargs):
          return tf.reverse(input, input2)
      return func
  
  # @nobuco.converter(torch.outer, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
  # def converter_outer(input, input2, *args, **kwargs):
  #     def func(input, input2, *args, **kwargs):
  #         return tf.tensordot(input, input2, axes=[0,1])
  #     return func
    
  # @nobuco.converter(torch.neg, channel_ordering_strategy=ChannelOrderingStrategy.MINIMUM_TRANSPOSITIONS)
  # def converter_neg(input, *args, **kwargs):
  #     def func(input, *args, **kwargs):
  #         return tf.math.negative(input)
  #     return func
  
  input_shape[0] = None
  keras_model = nobuco.pytorch_to_keras(
      model,
      args=[dummy_input], 
      input_shapes={dummy_input: input_shape},
      inputs_channel_order=ChannelOrder.PYTORCH,
      outputs_channel_order=ChannelOrder.PYTORCH,
      trace_shape=True, 
      debug_traces=FLAGS.convert_trace,
    )
  keras_model.summary()
  return keras_model
  