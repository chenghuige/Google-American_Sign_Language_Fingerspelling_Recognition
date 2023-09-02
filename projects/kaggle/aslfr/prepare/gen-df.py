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
from src import util

def main(_):
  record_dir = f'{FLAGS.root}/tfrecords/{FLAGS.records_version}'
  record_files = gezi.list_files(f'{record_dir}/*.tfrec')
  ic(record_files[:2])
  dataset = mt.Dataset('valid', 
                       files=record_files, 
                       varlen_keys=['frames'])
  datas = dataset.make_batch(256, return_numpy=True)

  ic(dataset.num_instances)
  num_steps = dataset.num_steps
  ic(num_steps)
  
  # str_keys=['phrase', 'phrase_type']
  l = []
  for x in tqdm(datas, total=num_steps, desc='Loop-dataset'):
    for key in x:
      x[key] = gezi.decode(x[key])
    xs = gezi.batch2list(x)
    for x in xs:
      frames = x['frames']
      frames = frames.reshape(-1, N_COLS)
      mask = frames.sum(axis=-1) != 0
      frames = frames[mask]
      assert frames.shape[0] == x['n_frames']
      # TODO seems could not save 2d array to feather
      # pyarrow.lib.ArrowInvalid: ('Can only convert 1-dimensional array values', 'Conversion failed for column frames with type object')
      # x['frames'] = frames.reshape(-1)
      x['frames'] = list(frames)
      l.append(x)
      
  df = pd.DataFrame(l)
  ic(df)
  if FLAGS.obj != 'sup':
    df.reset_index().to_feather(f'{FLAGS.root}/train.fea')
  else:
    df.reset_index().to_feather(f'{FLAGS.root}/sup.fea')

if __name__ == '__main__':
  app.run(main)
  