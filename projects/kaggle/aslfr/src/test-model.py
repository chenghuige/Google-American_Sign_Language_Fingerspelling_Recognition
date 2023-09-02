#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   main.py
#        \author   chenghuige  
#          \date   2023-06-26 17:04:46.311378
#   \Description   py ./test-model.py --flagfile=flags/encode
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
from src.dataset import Dataset
from src import eval as ev

def main(_):
  timer = gezi.Timer()
  fit = mt.fit  
  
  rank = gezi.get('RANK', 0)
  if rank != 0:
    ic.disable()
  
  FLAGS.wandb =False
  config.init()
  mt.init()
  config.show()
  
  ic(FLAGS.torch, FLAGS.tf_dataset)
  FLAGS.torch = False
  FLAGS.tf_dataset = True
  FLAGS.use_masking = True
  
  strategy = mt.distributed.get_strategy()
  with strategy.scope():    
    model = util.get_model()
    
  util.check_masking(model)

  FLAGS.batch_parse = False
  record_dir = f'{FLAGS.root}/tfrecords/{FLAGS.records_version}'
  record_files = gezi.list_files(f'{record_dir}/*.tfrec')
  ic(record_files[:2])
  dataset = Dataset('valid', 
                     files=record_files)
  datas = dataset.make_batch(256, return_numpy=True)

  ic(dataset.num_instances)
  num_steps = dataset.num_steps
  ic(num_steps)
  
  l = []
  for x, y in tqdm(datas, total=num_steps, desc='Loop-dataset'): 
    util.verify_correct_training_flag(model, x)
    exit(0)


if __name__ == '__main__':
  app.run(main)  
