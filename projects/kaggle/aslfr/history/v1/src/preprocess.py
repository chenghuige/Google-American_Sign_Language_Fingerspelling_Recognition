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
from src.tf.preprocess import PreprocessLayer, PreprocessLayerNonNaN

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
  gezi.set_fold(train, 
                FLAGS.folds, 
                # group_key='participant_id', 
                # stratify_key='phrase_type',
                seed=FLAGS.fold_seed)
  # rng = np.random.default_rng(FLAGS.fold_seed)
  # train['fold_woker'] = [
  #     rng.integers(FLAGS.fold_workers) for _ in range(len(train))
  # ]
  # train['worker'] = train['fold'] * FLAGS.fold_workers + train['fold_woker']


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
  df = pd.read_parquet(file_path)
  # # Landmark Indices for Left/Right hand without z axis in raw data
  LEFT_HAND_IDXS0, LEFT_HAND_NAMES0 = get_idxs(df, ['left_hand'], ['z'])
  RIGHT_HAND_IDXS0, RIGHT_HAND_NAMES0 = get_idxs(df, ['right_hand'], ['z'])
  LIPS_IDXS0, LIPS_NAMES0 = get_idxs(df, ['face'], ['z'],
                                     idxs_pos=LIPS_LANDMARK_IDXS)
  # COLUMNS0 = np.concatenate((LEFT_HAND_NAMES0, RIGHT_HAND_NAMES0, LIPS_NAMES0))
  ## v3
  POSE_IDXS0, POSE_NAMES0 = get_idxs(df, ['pose'], ['z'], idxs_pos=POSE)
  
  # COLUMNS0 = np.concatenate((LEFT_HAND_NAMES0, RIGHT_HAND_NAMES0, LIPS_NAMES0, POSE_NAMES0))
  
  # # Landmark Indices for Left/Right hand without z axis in raw data
  # LEFT_HAND_IDXS0, LEFT_HAND_NAMES0 = get_idxs(df, ['left_hand'])
  # RIGHT_HAND_IDXS0, RIGHT_HAND_NAMES0 = get_idxs(df, ['right_hand'])
  # LIPS_IDXS0, LIPS_NAMES0 = get_idxs(df, ['face'],
  #                                    idxs_pos=LIPS_LANDMARK_IDXS)
  # POSE_IDXS0, POSE_NAMES0 = get_idxs(df, ['pose'], idxs_pos=POSE)
  # # COLUMNS0 = np.concatenate((LEFT_HAND_NAMES0, RIGHT_HAND_NAMES0, LIPS_NAMES0))
  # ## v3
  # POSE_IDXS0, POSE_NAMES0 = get_idxs(df, ['pose'], idxs_pos=POSE)
  COLUMNS0 = np.concatenate((LEFT_HAND_NAMES0, RIGHT_HAND_NAMES0, LIPS_NAMES0, POSE_NAMES0))
  
  if save:
    with open(f'{FLAGS.root}/inference_args.json', 'w') as f:
        json.dump({ 'selected_columns': COLUMNS0.tolist() }, f)
  
  N_COLS0 = len(COLUMNS0)
  ic(N_COLS0)
  # Only X/Y axes are used
  N_DIMS0 = 2

  # Landmark Indices in subset of dataframe with only COLUMNS selected
  LEFT_HAND_IDXS = np.argwhere(np.isin(COLUMNS0, LEFT_HAND_NAMES0)).squeeze()
  RIGHT_HAND_IDXS = np.argwhere(np.isin(COLUMNS0, RIGHT_HAND_NAMES0)).squeeze()
  LIPS_IDXS = np.argwhere(np.isin(COLUMNS0, LIPS_NAMES0)).squeeze()
  N_COLS = N_COLS0
  ic(N_COLS)
  # configs['N_COLS'] = N_COLS
  # Only X/Y axes are used
  N_DIMS = 2

  # Indices in processed data by axes with only dominant hand
  HAND_X_IDXS = np.array([
      idx for idx, name in enumerate(LEFT_HAND_NAMES0) if 'x' in name
  ]).squeeze()
  HAND_Y_IDXS = np.array([
      idx for idx, name in enumerate(LEFT_HAND_NAMES0) if 'y' in name
  ]).squeeze()
  # Names in processed data by axes
  HAND_X_NAMES = LEFT_HAND_NAMES0[HAND_X_IDXS]
  HAND_Y_NAMES = LEFT_HAND_NAMES0[HAND_Y_IDXS]

  preprocess_layer = PreprocessLayer(FLAGS.n_frames)
  # Iterate Over Samples
  for group, group_df in tqdm(df.groupby('sequence_id'), total=len(df.index.unique()), desc='per_seq'):
    # Number of Frames Per Character
    n_frame = len(group_df[COLUMNS0].values)
    # Get Processed Frames and non empty frame indices
    # ic(group_df[COLUMNS0].values.shape)
    # (n_frames, n_cols)
    frame = preprocess_layer(group_df[COLUMNS0].values)
    assert frame.ndim == 2
    frame = list(frame.numpy().reshape(-1))
    yield group, frame, n_frame

  # # Iterate Over Samples
  # for group, group_df in tqdm(df.groupby('sequence_id'), total=len(df.index.unique()), desc='per_seq'):
  #   # Number of Frames Per Character
  #   n_frame = len(group_df[COLUMNS0].values)
  #   frame = group_df[COLUMNS0].values
  #   assert frame.ndim == 2
  #   frame = list(frame.reshape(-1))
  #   yield group, frame, n_frame

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

def init_dfs(mark='train'):
  file_name = 'train' if mark == 'train' else 'supplemental_metadata'
  train = pd.read_csv(f'{FLAGS.root}/{file_name}.csv')
  preprocss_(train)
  init_folds_(train)
  check_phrase_dup_(train)
  set_idx_(train)
  
  if mark == 'train':
    if gezi.get('RANK', 0) == 0:
      train.to_csv(f'{FLAGS.root}/train2.csv', index=False)
  dfs['train'] = train
