#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   preprocess.py
#        \author   chenghuige
#          \date   2023-06-25 15:48:54.622358
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import *
from src import util
from src.config import *
from src.tf.preprocess import PreprocessLayer

# https://www.kaggle.com/code/markwijkhuizen/aslfr-eda-preprocessing-dataset
def get_idxs(df, words_pos, words_neg=[], ret_names=True, idxs_pos=None):
  idxs = []
  names = []
  for w in words_pos:
    for col_idx, col in enumerate(df.columns):
      # Exclude Non Landmark Columns
      if col in ['frame']:
        continue

      col_idx = int(col.split('_')[-1])
      # Check if column name contains all words
      if (w in col) and (idxs_pos is None or col_idx in idxs_pos) and all(
          [w not in col for w in words_neg]):
        idxs.append(col_idx)
        names.append(col)
  # Convert to Numpy arrays
  idxs = np.array(idxs)
  names = np.array(names)
  # Returns either both column indices and names
  if ret_names:
    return idxs, names
  # Or only columns indices
  else:
    return idxs

dfs = {}
n_frames = {}
frames = {}

def init_folds_(train):
  if FLAGS.group_fold:
    gezi.set_fold(train, 
                  FLAGS.folds, 
                  group_key='participant_id', 
                  stratify_key='phrase_type',
                  seed=FLAGS.fold_seed)
  else:
    gezi.set_fold(train, 
                  FLAGS.folds,
                  stratify_key='phrase_type',
                  seed=FLAGS.fold_seed)

def check_phrase_dup_(train):
  counter = Counter()
  for row in train.itertuples():
    row = row._asdict()
    phrase = row['phrase']
    fold = row['fold']
    counter[phrase] += 1
    counter[f'{fold}^{phrase}'] += 1

  l = []
  for row in train.itertuples():
    dup = 0
    row = row._asdict()
    phrase = row['phrase']
    fold = row['fold']
    if counter[f'{fold}^{phrase}'] < counter[phrase]:
      dup = 1
    l.append(dup)

  train['phrase_dup'] = l
  
def preprocess_parquet(file_path, save=False):
  if save:
    with open(f'{FLAGS.root}/inference_args.json', 'w') as f:
      json.dump({ 'selected_columns': SEL_COLS }, f)
  
  df = pd.read_parquet(file_path, columns=SEL_COLS)
  seq_ids = df.index.unique()
  for seq_id in tqdm(seq_ids, total=len(seq_ids), desc='per_seq'):
    frame = df[df.index == seq_id].values
    assert frame.ndim == 2
    assert frame.shape[-1] == N_COLS    
    n_frame = frame.shape[0]
    frame = list(frame.reshape(-1))
    # for i in range(len(frame)):
    #   if abs(frame[i]) < 2**(-14):
    #     frame[i] = 0
    yield seq_id, frame, n_frame

def preprocss_(train):
  train['phrase_len'] = train['phrase'].apply(len)
  train['phrase_type'] = train['phrase'].apply(util.get_phrase_type)

  # Get complete file path to file
  def get_file_path(path):
    return f'{FLAGS.root}/{path}'

  train['file_path'] = train['path'].apply(get_file_path)

  
def set_idx_(train):
  idxes = [0] * FLAGS.folds
  l = []
  for row in train.itertuples():
    l.append(idxes[row.fold])
    idxes[row.fold] += 1
  train['idx'] = l  

def init_dfs(obj='train'):
  file_name = 'train' if obj == 'train' else 'supplemental_metadata'
  train = pd.read_csv(f'{FLAGS.root}/{file_name}.csv')
  preprocss_(train)
  init_folds_(train)
  check_phrase_dup_(train)
  set_idx_(train)
  
  if obj == 'train':
    if gezi.get('RANK', 0) == 0:
      gezi.try_mkdir(f'{FLAGS.root}/tfrecords/0')
      train.to_csv(f'{FLAGS.root}/tfrecords/0/train.csv', index=False)
  dfs['train'] = train
