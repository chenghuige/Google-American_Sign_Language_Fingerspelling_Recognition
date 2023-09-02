#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   eval.py
#        \author   chenghuige  
#          \date   2023-06-26 20:23:08.127916
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
from src.config import *
from src.tf.model import *
from src import util

from leven import levenshtein
# from mbleven import compare
# import editdistance
# from polyleven import levenshtein

def decode(phrase):
  phrase = ''.join([IDX2CHAR[idx] for idx in phrase]).split('<EOS>')[0]
  # ic(phrase)
  if '>' in phrase:
    phrase = phrase.split('>')[1]
  # ic(phrase)
  return phrase

def calc_score(phrase_true, phrase_pred):
  return levenshtein(phrase_true, phrase_pred)

def evaluate(dataset, model, eval_step, steps, step, is_last, num_examples, loss_fn, outdir):
  l = []
  t = tqdm(enumerate(dataset), total=steps, ascii=True, desc= 'eval_loop')
  for step_, (x, y) in t:
    if step_ == steps:
      break
    # frames = x['frames']
    frames = model.preprocess(x['frames'])
    # y_ = infer_model(frames)
    y_ = model.infer(frames)
    y, y_ = y.numpy(), y_.numpy()
    y_ = np.argmax(y_, axis=-1)
    sequence_ids = x['sequence_id'].numpy().squeeze(-1)
    phrase_types = x['phrase_type'].numpy().squeeze(-1)
    phrase_dups = x['phrase_dup'].numpy().squeeze(-1)
    n_frames = x['n_frames'].numpy().squeeze(-1)
    idxes = x['idx'].numpy().squeeze(-1)
    for sequence_id, phrase_true, phrase_pred, phrase_type, phrase_dup, n_frame, idx \
      in zip(sequence_ids, y, y_, phrase_types, phrase_dups, n_frames, idxes):
      # ic(phrase_true)
      phrase_true = decode(phrase_true)
      # ic(phrase_pred)
      phrase_pred = decode(phrase_pred)
      # ic(phrase_true, phrase_pred)
      phrase_type = phrase_type.decode('utf-8')
      phrase_type_pred = util.get_phrase_type(phrase_pred)
      distance = calc_score(phrase_true, phrase_pred)
      l.append({
        'sequence_id': sequence_id,
        'phrase_type': phrase_type,
        'phrase_dup': phrase_dup,
        'phrase_true': phrase_true,
        'phrase_pred': phrase_pred,
        'n_frame': n_frame,
        'phrase_len_true': len(phrase_true),
        'phrase_len_pred': len(phrase_pred),
        'idx': idx,
        'distance': distance,
        'type_acc': phrase_type == phrase_type_pred,
        'score': max(len(phrase_true) - distance, 0.) / len(phrase_true),
      })
      
      
  df = pd.DataFrame(l)
  def get_metrics(df):
    metrics = {
      'phrase_len_true': df['phrase_len_true'].mean(),
      'phrase_len_pred': df['phrase_len_pred'].mean(),
      'distance': df['distance'].mean(),
      'type_acc': df['type_acc'].mean(),
      'score': max(df['phrase_len_true'].sum() - df['distance'].sum(), 0.) / df['phrase_len_true'].sum(),
    }
    return metrics
  
  metrics = get_metrics(df)
  metrics.update(gezi.dict_suffix(get_metrics(df[df['phrase_type'] == 'phone']), '/phone'))
  metrics.update(gezi.dict_suffix(get_metrics(df[df['phrase_type'] == 'url']), '/url'))
  metrics.update(gezi.dict_suffix(get_metrics(df[df['phrase_type'] == 'address']), '/address'))
  metrics.update(gezi.dict_suffix(get_metrics(df[df['phrase_dup'] == 1]), '/dup'))
  metrics.update(gezi.dict_suffix(get_metrics(df[df['phrase_dup'] == 0]), '/new'))
  metrics.update(gezi.dict_suffix(get_metrics(df[df['n_frame'] > FLAGS.n_frames]), '/long'))
  metrics.update(gezi.dict_suffix(get_metrics(df[df['n_frame'] <= FLAGS.n_frames]), '/short'))
  
  df = df[df.idx < 20]
  df = df.sort_values(by=['idx'], ascending=True)
  ic(df.head(20))
  metrics['score/head'] = max(df['phrase_len_true'].sum() - df['distance'].sum(), 0.) / df['phrase_len_true'].sum()
  gezi.set('eval_df', df)
  # # TODO not sure why but evauate will slow down training, return {} will not slow down....
  # ic(metrics)
  # return {}
  return metrics
