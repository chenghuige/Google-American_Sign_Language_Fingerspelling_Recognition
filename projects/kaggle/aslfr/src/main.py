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
os.environ["NCCL_DEBUG"] = 'WARNING'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["OMP_NUM_THREADS"] = '8'
  
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
  config.show()
    
  # NOTICE will exit after prepare so if DDP will fail for other process but that's fine
  if rank == 0:
    os.system(f'rsync -aq ../src {FLAGS.model_dir}')
    os.system(f'rsync ../dataset-metadata.json {FLAGS.model_dir}')
    if FLAGS.kaggle_prepare or os.path.exists(f'{FLAGS.model_dir}/done.txt'):
      gezi.prepare_kaggle_dataset(f'{MODEL_NAME}-model')
  
  ic(FLAGS.torch, FLAGS.tf_dataset, FLAGS.tfrecords, FLAGS.torch2tf_convert, FLAGS.no_convert, FLAGS.force_convert)
  # FLAGS.torch = False
  # FLAGS.tf_dataset = True
  
  strategy = mt.distributed.get_strategy()
  ic(strategy)
  with strategy.scope():    
    model = util.get_model()
    if not FLAGS.torch:
      func_model = model.get_model()
      mt.print_model(func_model, depth=0, print_fn=print)
      from tensorflow.keras.utils import plot_model
      plot_model(func_model, to_file=f'{FLAGS.model_dir}/model.png', show_shapes=True, show_layer_names=True)
      try:
        func_model.save(f'{FLAGS.model_dir}/func_model.h5')
      except Exception as e:
        ic(e)
      # except Exception:
      #   pass
    else:
      ic(model)
      
      from torchinfo import summary
      input_shape = (1, FLAGS.n_frames, get_n_cols())
      summary(model.get_infer_model(), input_shape)
      # pass
      
    if FLAGS.convert_only:
      util.to_tflite_model(model)
      exit(0)
    
    if FLAGS.tf_dataset:
      # train from tfrecords input pre gen
      if FLAGS.tfrecords:
        # 如果torch 目前DDP 读取tf shard模式 不知道为何效果会差一些 TODO
        from src.dataset import Dataset
        fit(model,  
            Dataset=Dataset,
            eval_fn=ev.get_eval_fn(),
            metrics=util.get_metrics(),
            callbacks=util.get_callbacks(model),
            ) 
      else:
        from src.dataset import get_datasets
        train_ds, eval_ds = get_datasets()
        fit(model,  
            dataset=train_ds,
            eval_dataset=eval_ds,
            eval_fn=ev.get_eval_fn(),
            metrics=util.get_metrics(),
            callbacks=util.get_callbacks(model),
            ) 
    else:
      from src.torch.dataset import get_dataloaders
      train_loader, eval_loader = get_dataloaders()
      fit(model,  
          dataset=train_loader,
          eval_dataset=eval_loader,
          eval_fn=ev.get_eval_fn(),
          callbacks=util.get_callbacks(model),
          ) 
       
    if rank == 0:
      if FLAGS.task == 'seq' and (not FLAGS.no_convert):
        if FLAGS.work_mode != 'train' or FLAGS.force_convert:
          if FLAGS.force_convert or (not FLAGS.torch) or FLAGS.torch2tf_convert or gezi.get('torch2tf'):
            util.to_tflite_model(model)
            
        df = gezi.get('eval_df')
        if df is not None and (not os.path.exists(f'{FLAGS.model_dir}/eval.csv')):
          df.to_csv(f'{FLAGS.model_dir}/eval.csv', index=False) 
                      
        if (not FLAGS.online) and FLAGS.wandb:
          gezi.folds_metrics_wandb(FLAGS.model_dir, FLAGS.folds)
      
if __name__ == '__main__':
  app.run(main)  
