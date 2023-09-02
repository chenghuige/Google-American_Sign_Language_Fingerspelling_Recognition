#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-mean.py
#        \author   chenghuige  
#          \date   2023-06-28 17:08:49.347831
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
import os
import gezi
from gezi.common import *
import melt as mt

from src import config
from src.config import *
from src.preprocess import *
from src.dataset import Dataset
from src import util

def main(_):
  # config.init()
  FLAGS.batch_parse = False
  record_dir = f'{FLAGS.root}/tfrecords/0.1'
  record_files = gezi.list_files(f'{record_dir}/*.tfrec')
  record_files = [x for x in record_files if int(os.path.basename(x).split('.')[0]) % 4 == 0]
  
  ic(record_files[:2])
  dataset = Dataset('valid', 
                     files=record_files)
  # datas = dataset.make_batch(128, repeat=True, return_numpy=True)
  
  datas = dataset.make_batch(128, return_numpy=False)

  ic(dataset.num_instances)
  num_steps = dataset.num_steps
  ic(num_steps)
  
  # str_keys=['phrase', 'phrase_type']
  # l = []
  # step = 0
  # di = iter(datas)
  # for _ in tqdm(range(num_steps), total=num_steps, desc='Loop-dataset'):
  #   x, y = next(di)
  #   if step == 0:
  #      ic(step, x['idx'])
  #   step += 1
  #   if step == num_steps:
  #     ic(step, len(x['idx']), x['idx'])
      
  # step = 0
  # for _ in tqdm(range(num_steps), total=num_steps, desc='Loop-dataset'):
  #   x, y = next(di)
  #   if step == 0:
  #      ic(step, x['idx'])
  #   step += 1
  #   if step == num_steps:
  #     ic(step, len(x['idx']), x['idx'])
  
  l = []
  step = 0
  di = iter(datas)
  for x, y in tqdm(datas, total=num_steps, desc='Loop-dataset'):
    if step == 0:
       ic(step, x['idx'])
    step += 1
    if step == num_steps:
      ic(step, len(x['idx']), x['idx'])
      
  step = 0
  for x, y in tqdm(datas, total=num_steps, desc='Loop-dataset'):
    if step == 0:
       ic(step, x['idx'])
    step += 1
    if step == num_steps:
      ic(step, len(x['idx']), x['idx'])
    
        
if __name__ == '__main__':
  app.run(main)
  
