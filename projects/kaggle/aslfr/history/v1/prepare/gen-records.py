#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   gen-records.py
#        \author   chenghuige  
#          \date   2023-06-25 15:25:12.934341
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

flags.DEFINE_integer('buf_size', 1000, '')
flags.DEFINE_float('min_frames_per_char', 0, '')

train = {}
file_paths = []
records_dir = None

def deal(index):
  file_path = file_paths[index]
  start_idx = index * FLAGS.folds
  ofiles = [f'{records_dir}/{start_idx + idx}.tfrec' for idx in range(FLAGS.folds)]
  writers = [mt.tfrecords.Writer(ofile, buffer_size=FLAGS.buf_size, shuffle=True, seed=1024) for ofile in ofiles]
  for sequence_id, frame, n_frame in preprocess_parquet(file_path, save=(index==0)):
    row = train[sequence_id]
    fe = {}
    for key in ['sequence_id', 'file_id', 'participant_id', 'phrase', 'fold',
                'phrase_len', 'phrase_type', 'phrase_dup', 'idx']:
      fe[key] = row[key]
    fe['frames'] = frame
    fe['n_frames'] = n_frame
    fe['n_frames_per_char'] = n_frame / row['phrase_len']
    if FLAGS.min_frames_per_char and fe['n_frames_per_char'] < FLAGS.min_frames_per_char:
      continue
    phrase = [CHAR2IDX[c] for c in row['phrase']]
    phrase.append(EOS_TOKEN)
    phrase = gezi.pad(phrase, FLAGS.max_phrase_len, PAD_TOKEN)
    fe['phrase'] = phrase
    fe['phrase_type_'] = PHRASE_TYPES[row['phrase_type']]
    fe['weight'] = 1.0 if FLAGS.mark == 'train' else FLAGS.sup_weight
    writers[row['fold']].write(fe)
    
  for writer in writers:
    writer.close()
      
def main(_):    
  global train, records_dir 
  config.init(gen_records=True)
  init_dfs(mark=FLAGS.mark)
  records_dir = f'{FLAGS.root}/tfrecords/{FLAGS.records_version}'
  ic(records_dir)
  command = f'rm -rf {records_dir}'
  gezi.system(command)    
  gezi.try_mkdir(records_dir)
  df = dfs['train']
  for row in tqdm(df.itertuples(), total=len(df), desc='train'):
    row = row._asdict()
    train[row['sequence_id']] = row
    
  file_paths.extend(df.file_path.unique())
  num_records = len(file_paths)
  ic(num_records)
  # deal(0)
  gezi.prun_loop(deal, range(num_records), num_records)
  
  # ic(FLAGS.mark, records_dir, mt.get_num_records_from_dir(records_dir))

if __name__ == '__main__':
  app.run(main)

