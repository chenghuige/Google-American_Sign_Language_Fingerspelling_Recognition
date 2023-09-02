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
                       incl_keys=['frames', 'n_frames'],
                       varlen_keys=['frames'])
  datas = dataset.make_batch(256, return_numpy=True)

  ic(dataset.num_instances)
  num_steps = dataset.num_steps
  ic(num_steps)

  l = []
  # N_COLS = int(iter(datas).next()['frames'].shape[-1] / FLAGS.n_frames)
  ic(N_COLS)
  means = np.zeros([N_COLS], dtype=np.float32)
  stds = np.zeros([N_COLS], dtype=np.float32)
  frames_list = []
  for x in tqdm(datas, total=num_steps, desc='Loop-dataset'):
    frames = x['frames']
    n_frames = x['n_frames']
    frames = frames.reshape(frames.shape[0], -1, N_COLS)
    for frame, n_frame in zip(frames, n_frames):
      frame = frame[:n_frame]
      frames_list.append(frame)

  frames = np.concatenate(frames_list, axis=0)
  for col, v in tqdm(enumerate(frames.reshape([-1, N_COLS]).T), total=N_COLS, desc='N_COLS'):
    v = v[~np.isnan(v)]
    # v = v[np.nonzero(v)]
    # Remove zero values as they are NaN values
    means[col] = v.astype(np.float32).mean()
    stds[col] = v.astype(np.float32).std()
    mean = means[col]
    std = stds[col]
    ic(col, 
       mean, 
       std,
       v.astype(np.float32).max(),
       v.astype(np.float32).min(),
       )
    
    # very important, other wise keras and tflite results diff...
    if std < 1e-6:
      ic('bad std', col, mean, std)
      # means[col] = 0
      stds[col] = 1.

  ic(means, means.mean(), means.max(), means.min(), np.absolute(means).mean(), means.shape)
  ic(stds, stds.mean(), stds.max(), stds.min())
  gezi.save(means, f'{FLAGS.root}/means.npy')
  gezi.save(stds, f'{FLAGS.root}/stds.npy')
  
if __name__ == '__main__':
  app.run(main)
  