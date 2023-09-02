#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   infer.py
#        \author   chenghuige
#          \date   2023-06-27 20:34:34.290044
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os

sys.path.append('..')
sys.path.append('../../../../utils')
sys.path.append('../../../../third')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_DEBUG"] = 'WARNING'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from icecream import ic
import tflite_runtime.interpreter as tflite
import tflite_runtime
ic(tflite_runtime.__version__)

# from gezi.common import *
# from src import config
# gezi.init_flags()
# config.init()
# from src.preprocess import init_folds_, preprocss_

import time
import json
import pandas as pd
from tqdm.auto import tqdm
from leven import levenshtein
# import tensorflow as tf
import numpy as np

model_dir = sys.argv[1]
fold = 0
if len(sys.argv) > 2:
  fold = int(sys.argv[2])

total = 20
if len(sys.argv) > 3:
  total = int(sys.argv[3])

SEL_FEATURES = json.load(
    open(f'{model_dir}/inference_args.json'))['selected_columns']
# ic(SEL_FEATURES)

def load_relevant_data_subset(pq_path):
  return pd.read_parquet(pq_path, columns=SEL_FEATURES)  #selected_columns)

with open("/kaggle/input/asl-fingerspelling/character_to_prediction_index.json",
          "r") as f:
  character_map = json.load(f)
rev_character_map = {j: i for i, j in character_map.items()}

df = pd.read_csv('/kaggle/input/asl-fingerspelling/train2.csv')
df = df[df.fold == fold]
df = df.reset_index(drop=True)

idx = 0
sample = df.loc[idx]
loaded = load_relevant_data_subset('/kaggle/input/asl-fingerspelling/' +
                                   sample['path'])
loaded = loaded[loaded.index == sample['sequence_id']].values
print(loaded.shape)
frames = loaded

interpreter = tflite.Interpreter(f'{model_dir}/model.tflite')
# interpreter = tf.lite.Interpreter(f'{model_dir}/model.tflite')
found_signatures = list(interpreter.get_signature_list().keys())

REQUIRED_SIGNATURE = 'serving_default'
REQUIRED_OUTPUT = 'outputs'
if REQUIRED_SIGNATURE not in found_signatures:
  raise KernelEvalException('Required input signature not found.')

prediction_fn = interpreter.get_signature_runner("serving_default")
output_lite = prediction_fn(inputs=frames)
prediction_str = "".join([
    rev_character_map.get(s, "")
    for s in np.argmax(output_lite[REQUIRED_OUTPUT], axis=1)
])
ic(sample['phrase'], prediction_str)

st = time.time()
cnt = 0
model_time = 0
ic(total, len(df))
total = min(len(df), total)

l = []
for i in tqdm(range(total)):
  sample = df.loc[i]
  loaded = load_relevant_data_subset('/kaggle/input/asl-fingerspelling/' +
                                     sample['path'])
  loaded = loaded[loaded.index == sample['sequence_id']].values
  # ic(loaded)
  md_st = time.time()
  output_ = prediction_fn(inputs=loaded)
  model_time += time.time() - md_st

  prediction_str = "".join([
      rev_character_map.get(s, "")
      for s in np.argmax(output_[REQUIRED_OUTPUT], axis=1)
  ])
  distance = levenshtein(sample['phrase'], prediction_str)
  phrase_true = sample['phrase']
  phrase_pred = prediction_str
  l.append({
    'sequence_id': sample['sequence_id'],
    'phrase_true': phrase_true,
    'phrase_pred': phrase_pred,
    'phrase_len_true': len(phrase_true),
    'phrase_len_pred': len(phrase_pred),
    'distance': distance, 
    'score': max(len(phrase_true) - distance, 0.) / len(phrase_true),
  })
  
df = pd.DataFrame(l)
score = max(df['phrase_len_true'].sum() - df['distance'].sum(), 0.) / df['phrase_len_true'].sum()
ic(df.head(20), score)
mean_time = model_time / total
ic(mean_time)
