#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   main.py
#        \author   chenghuige  
#          \date   2023-06-26 17:04:46.311378
#   \Description  
# ==============================================================================

  
import sys, os
sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_DEBUG"] = 'WARNING'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from gezi.common import *

import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp

import src
from src import config
from src.config import *
from src import util
from src.preprocess import *
from src import eval as ev

def main(_):
  timer = gezi.Timer()
  fit = mt.fit  
  
  rank = gezi.get('RANK', 0)
  if rank != 0:
    ic.disable()
    
  config.init()
  mt.init()
  
  if FLAGS.ensemble_metrics:
    args = ' '.join(sys.argv[1:])
    gezi.system(f'./ensemble.py {args} --mn={FLAGS.mn}')
    exit(0)
  
  # NOTICE will exit after prepare so if DDP will fail for other process but that's fine
  if rank == 0:
    os.system(f'rsync -aq ../src {FLAGS.model_dir}')
    os.system(f'rsync ../dataset-metadata.json {FLAGS.model_dir}')
    if FLAGS.kaggle_prepare or os.path.exists(f'{FLAGS.model_dir}/done.txt'):
      gezi.prepare_kaggle_dataset(f'{MODEL_NAME}-model')
  
  ic(FLAGS.torch, FLAGS.tf_dataset)
  FLAGS.torch = False
  FLAGS.tf_dataset = True
  
  if not FLAGS.torch:
    from src.tf.model import Model, TFLiteModel
  else:
    from src.torch.model import Model
  
  strategy = mt.distributed.get_strategy()
  with strategy.scope():    
    model = Model()

    # train from tfrecords input pre gen
    if FLAGS.tf_dataset:
      from src.dataset import Dataset
      fit(model,  
          Dataset=Dataset,
          eval_fn=ev.evaluate,
          metrics=util.get_metrics(),
          callbacks=util.get_callbacks(model),
          ) 
    else:
      assert FLAGS.tf_dataset
      
    if rank == 0:
      if FLAGS.work_mode == 'train' or FLAGS.force_save or FLAGS.save_final:
        gezi.save_model(model, FLAGS.model_dir, fp16=False)
        
      tflite_keras_model = TFLiteModel(model)
      # Create Model Converter
      converter = tf.lite.TFLiteConverter.from_keras_model(tflite_keras_model)  
      ## float16 seems slower 55 to 60, but it will make model half size if needed
      # converter.optimizations = [tf.lite.Optimize.DEFAULT]
      # converter.target_spec.supported_types = [tf.float16]
      ## converter.experimental_new_converter = True
      # Convert Model
      tflite_model = converter.convert()
      # Write Model
      with open(f'{FLAGS.model_dir}/model.tflite', 'wb') as f:
        f.write(tflite_model)
      # Add selected_columns json to only select specific columns from input frames
      gezi.system(f'cp {FLAGS.root}/inference_args.json {FLAGS.model_dir}')
      # gezi.system(f'cd {FLAGS.model_dir};zip submission.zip model.tflite inference_args.json')
      gezi.system(f'cd {FLAGS.model_dir};cp model.tflite inference_args.json dataset-metadata.json ./ckpt')
      
      if FLAGS.work_mode == 'train':
        gezi.system(f'sh ./infer.sh {FLAGS.model_dir} {FLAGS.fold} &')
      else: 
        gezi.system(f'./infer.py {FLAGS.model_dir} {FLAGS.fold} &')
      
      df = gezi.get('eval_df')
      if df is not None:
        df.to_csv(f'{FLAGS.model_dir}/eval.csv', index=False) 
      
      if not FLAGS.online:
        gezi.folds_metrics_wandb(FLAGS.model_dir, FLAGS.folds)
      
if __name__ == '__main__':
  app.run(main)  
