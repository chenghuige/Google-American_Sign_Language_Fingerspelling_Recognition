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
  record_dir = f'{FLAGS.root}/tfrecords/{FLAGS.records_version}'
  record_files = gezi.list_files(f'{record_dir}/*.tfrec')
  ic(record_files[:2])
  dataset = Dataset('valid', 
                     files=record_files)
  datas = dataset.make_batch(256, return_numpy=True)

  ic(dataset.num_instances)
  num_steps = dataset.num_steps
  ic(num_steps)
  
  # str_keys=['phrase', 'phrase_type']
  l = []
  for x, y in tqdm(datas, total=num_steps, desc='Loop-dataset'):
    for key in x:
      x[key] = gezi.decode(x[key])
    ic(x['frames'].shape, y.shape)
    xs = gezi.batch2list(x)
    for x in xs:
      ic(x['frames'])
      ic(x['frames'].shape)
      exit(0)
        
if __name__ == '__main__':
  app.run(main)
  