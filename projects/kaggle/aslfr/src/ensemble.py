#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   ensemble.py
#        \author   chenghuige  
#          \date   2023-07-23 08:21:18.961942
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# os.environ["OMP_NUM_THREADS"] = '8'
import sys
sys.path.append('/work/pikachu/utils/')
sys.path.append('/work/pikachu/third/')
sys.path.append('..')
from gezi.common import *
import gezi
from src import config
from src.config import *
from src import util
from src.dataset import Dataset
from src.eval import eval_seq
gezi.init_flags()

V = RUN_VERSION
fold = 0
mode = 'offline'
# mode = 'online'
model_names = [
    'conformer-fat.encoder_layers-15.time_reduce.time_reduce_idx-7.relpos_combine_mode-2.fixreduce.eval',
    'conformer-fat.encoder_layers-15.encoder-conformer_v5_1_11.encode_pool_size-1.reduce_idxes-8,13.eval',
  ]
# weights = [5, 1]
weights = []
root = f'../working/{mode}/{V}/{fold}'
model_paths = [f'{root}/{model_name}' for model_name in model_names]
gezi.restore_configs(model_paths[0].replace('.eval', ''))

FLAGS.mn = 'ensemble'
config.init()
mt.init()
config.show()

model_dir = FLAGS.model_dir

os.system(f'rsync -aq ../src {FLAGS.model_dir}')
os.system(f'rsync ../dataset-metadata.json {FLAGS.model_dir}')

# record_dir = f'{FLAGS.root}/tfrecords/{FLAGS.records_version}'
record_dir = f'{FLAGS.root}/tfrecords/0.1'
records_pattern = f'{record_dir}/*.tfrec'
files = gezi.list_files(records_pattern) 
FLAGS.valid_files = [x for x in files if int(os.path.basename(x).split('.')[0]) % FLAGS.folds == FLAGS.fold]
dataset = Dataset('valid', files=FLAGS.valid_files)
# means = gezi.load(f'{record_dir}/means.npy')
# stds = gezi.load(f'{record_dir}/stds.npy')
# STATS['means'] = means
# STATS['stds'] = stds
datas = dataset.make_batch(FLAGS.batch_size)

class EnsembleModel(mt.Model):

  def __init__(self,
               models,
               weights=[],
               activation=None,
               cpu_merge=False,
               **kwargs):
    super(EnsembleModel, self).__init__(**kwargs)
    assert models
    self.models = models
    for i in range(len(models)):
      self.models[i]._name = self.models[i].name + '_' + str(i)
    self.models_weights = list(map(float, weights))
    self.activation = tf.keras.activations.get(activation)
    self.cpu_merge = cpu_merge
    self.name_ = f'Ensemble_{len(models)}'

  def call(self, x):
    return self.infer(x)
  
  def infer(self, x):
    if not self.cpu_merge:
      xs = [model.infer(x) if hasattr(model, 'infer') else model(x) for model in self.models]
    else:
      xs = []
      for model in self.models:
        res = model(x)
        with tf.device('/cpu:0'):
          res = tf.identity(res)
          xs.append(res)

    device_ = 'gpu' if not self.cpu_merge else 'cpu'
    with mt.device(device_):
      reduce_fn = tf.reduce_mean

      if self.models_weights:
        reduce_fn = tf.reduce_sum
        assert len(self.models_weights) == len(xs)
        xs = [
            self.activation(xs[i]) * self.models_weights[i]
            for i in range(len(xs))
        ]
      else:
        xs = [self.activation(xs[i]) for i in range(len(xs))]

      x = reduce_fn(tf.stack(xs, axis=1), axis=1)

    return x

  def get_model(self):
    try:
      inp = self.models[0].input
      out = self.call(inp)
      return tf.keras.Model(inp, out, name=self.name_)
    except Exception as e:
      print(e)
      return self

strategy = mt.distributed.get_strategy()
with strategy.scope():    
  def get_model(model_path):
    model_path_ = model_path.replace('.eval', '')
    ic(model_path_)
    gezi.restore_configs(model_path_)
    ic(FLAGS.encoder_layers, FLAGS.encoder_units, FLAGS.n_frames)
    model = util.get_model()
    model = util.prepare_tflite(model)
    model(iter(datas).next()[0]['frames'])
    model.load_weights(f'{model_path}/tflite.h5')
    # model.get_model().summary()
    ic(pd.read_csv(f'{model_path}/metrics.csv').tail(1)['score'].values[0])
    return model

  models = [get_model(model_path) for model_path in tqdm(model_paths)]
  model = EnsembleModel(models, weights=weights, activation='softmax')
  model(iter(datas).next()[0]['frames'])
  model.get_model().summary()

  FLAGS.torch = False
  num_steps = dataset.num_steps
  metrics = eval_seq(datas, model, num_steps)
  ic(metrics)
  ic(metrics['score'], metrics['score/head'])

FLAGS.model_dir = model_dir
df = pd.DataFrame([metrics])
df.to_csv(f'{FLAGS.model_dir}/metrics.csv', index=False)
util.to_tflite_model(model)
  
