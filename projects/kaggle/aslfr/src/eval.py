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
from src.tf.decode import decode_phrase, simple_decode
from src import util
import lele

from leven import levenshtein
from gezi.metrics import fast_auc
from sklearn.metrics import log_loss
# from mbleven import compare
# import editdistance
# from polyleven import levenshtein

def is_normal_char(idx):
  # return idx > 0 and idx <= N_CHARS
  return not IDX2CHAR[idx].startswith('<')

def to_original_phrase(phrase_pred):
  phrase = simple_decode(phrase_pred).numpy()
  return phrase

def to_str(phrase):
  phrase = ''.join([IDX2CHAR[idx] for idx in phrase if is_normal_char(idx)])
  # if EOS_TOKEN in phrase:
  #   phrase = phrase.split(EOS_TOKEN)[0]
  # else:
  #   phrase = phrase.split(PAD_TOKEN, 1)[0]
  # if '>' in phrase:
  #   phrase = phrase.split('>')[1]
  # ic(phrase)
  return phrase

def max_char_idx(phrase):
  idxs = [i + 1 for i, idx in enumerate(phrase) if is_normal_char(idx)]
  return max(idxs) if idxs else 0

def calc_score(phrase_true, phrase_pred):
  return levenshtein(phrase_true, phrase_pred)

best_score = -1.
bad_count = 0
# To force same as tflite infer might need FLAGS.eval_bs=1
def evaluate(dataset, model, eval_step, steps, step, is_last, num_examples, loss_fn, outdir):
  return eval_seq(dataset, model, steps)

def eval_seq(dataset, model, steps):
  metrics = {}
  rank = gezi.get('RANK', 0)
  if rank == 0:
    l = []
    
    # t = tqdm(enumerate(dataset), total=steps, ascii=True, desc= 'data_loop')
    ## for test perf of loop dataset
    # li = []
    # for step_, (x, y) in t:
    #   li.append((x, y))
    #   if step_ == steps:
    #     break
    
    outputs = []
    feats = []
        
    # t = tqdm(enumerate(li), total=steps, ascii=True, desc= 'eval_loop')
    t = tqdm(enumerate(dataset), total=steps, ascii=True, desc= 'eval_loop')
    for step_, (x, y) in t:
      if step_ == steps:
        break
      
      if FLAGS.torch:
        if not torch.is_tensor(x['frames']):
          x = lele.to_torch(x)  
        else:
          x['frames'] = x['frames'].cuda()
          
      frames = x['frames']
      
      # NOTICE infer only using frames but for seq training also use phrase_true
      if not FLAGS.torch:
        y_ = model.infer(frames)
      else:
        with torch.no_grad():
          y_ = model.infer(frames)
      
      if FLAGS.torch:
        for key in x:
          if hasattr(x[key], 'cpu'):
            x[key] = x[key].cpu().detach()
        y_ = y_.cpu().detach().numpy()
        if FLAGS.len_loss_weight:
          y_len_ = gezi.get('len_pred')
          y_len_ = y_len_.cpu().detach().numpy()
          y_len2_ = np.argmax(y_len_, axis=-1) + 1
          y_len_ = ((gezi.softmax(y_len_) * np.arange(MAX_PHRASE_LEN)).sum(-1) + 0.5 + 1).astype(int)
          
      outputs.append(y_)
      if FLAGS.eval_train:
        feat = gezi.get('feature').cpu().detach().numpy()
        feats.append(feat)
      
      y = y.numpy()
      sequence_ids = gezi.try_numpy(x['sequence_id'])
      phrase_types = gezi.try_numpy(x['phrase_type'])
      phrase_dups = gezi.try_numpy(x['phrase_dup'])
      n_frames = gezi.try_numpy(x['n_frames'])
      frame_means = gezi.try_numpy(x['frame_mean'])
      idxes = gezi.try_numpy(x['idx'])
      cls_labels = gezi.try_numpy(x['cls_label'])
      for i, (sequence_id, phrase_true, phrase_pred, phrase_type, phrase_dup, n_frame, frame_mean, cls_label, idx) \
        in enumerate(zip(sequence_ids, y, y_, phrase_types, phrase_dups, n_frames, frame_means, cls_labels, idxes)):  
        phrase_ori = to_original_phrase(phrase_pred)
        char_max_idx = max_char_idx(phrase_ori)
        char_ori_rate = char_max_idx  / len(phrase_ori)
        phrase_ori = ''.join([IDX2CHAR[idx] for idx in phrase_ori])
        phrase_pred = decode_phrase(phrase_pred).numpy()
        phrase_true = to_str(phrase_true)
        phrase_pred = to_str(phrase_pred)
        # try:
        char_true_rate = 0. if not len(phrase_true) else char_max_idx / len(phrase_true)
        char_pred_rate = 0. if not len(phrase_pred) else char_max_idx / len(phrase_pred)
        # except Exception:
        #   ic(phrase_true, phrase_pred, phrase_ori, char_max_idx, len(phrase_true), len(phrase_pred))
        #   exit(0)
        
        cls_pred = np.zeros_like(cls_label)
        for ch in phrase_pred:
          cidx = CHAR2IDX[ch] - 1
          cls_pred[cidx] = 1
          
        # ic(phrase_true, phrase_pred)
        phrase_type = phrase_type.decode('utf-8')
        phrase_type_pred = util.get_phrase_type(phrase_pred)
        distance = calc_score(phrase_true, phrase_pred)
        # phrase_ori = phrase_ori.replace(PAD_TOKEN, 'P').replace(EOS_TOKEN, 'E') \
        #                       .replace(PHONE_TOKEN, 'D').replace(ADDRESS_TOKEN, 'A') \
        #                       .replace(URL_TOKEN, 'U').replace(SUP_TOKEN, 'S') 
        m = {
          'sequence_id': sequence_id,
          'phrase_type': phrase_type,
          'phrase_dup': phrase_dup,
          'phrase_true': phrase_true,
          'phrase_pred': phrase_pred,
          'phrase_ori': phrase_ori,
          'char/max_idx': char_max_idx,
          'char/ori_rate': char_ori_rate,
          'char/true_rate': char_true_rate,
          'char/pred_rate': char_pred_rate,
          'phrase_len_true': len(phrase_true),
          'phrase_len_pred': len(phrase_pred),
          'phrase_len_pred_': y_len_[i] if FLAGS.len_loss_weight else -1,
          'phrase_len_pred2_': y_len2_[i] if FLAGS.len_loss_weight else -1,
          'phrase_len_rate': len(phrase_pred) / len(phrase_true),
          'idx': idx,
          'distance': distance,
          'acc/char': (cls_label == cls_pred).mean(),
          'acc/type': phrase_type == phrase_type_pred,
          'acc/first': phrase_true[0] == phrase_pred[0] if phrase_pred else False,
          'acc/last': phrase_true[-1] == phrase_pred[-1] if phrase_pred else False,
          'n_frame': n_frame,
          'frame_mean': frame_mean,
          'score': max(len(phrase_true) - distance, 0.) / len(phrase_true),
        }
        l.append(m)
        
    outputs = np.concatenate(outputs, axis=0)
    gezi.save(outputs, f'{FLAGS.model_dir}/outputs.npy')
    if feats:
      feats = np.concatenate(feats, axis=0)
      gezi.save(feats, f'{FLAGS.model_dir}/feats.npy')  
      exit(0)
      
    df = pd.DataFrame(l)
    ic(len(df), len(df.idx.unique()))
    ic(df.head(2))
    assert len(df) == len(df.idx.unique())
    def get_metrics(df):
      metrics = {
        'phrase_len_true': df['phrase_len_true'].mean(),
        'phrase_len_pred': df['phrase_len_pred'].mean(),
        'phrase_len_rate': df['phrase_len_rate'].mean(),
        'len/l1': (df['phrase_len_true'] - df['phrase_len_pred']).abs().mean(),
        'len/l1_': (df['phrase_len_true'] - df['phrase_len_pred_']).abs().mean() if FLAGS.len_loss_weight else -1,
        'len/l1__': (df['phrase_len_true'] - df['phrase_len_pred2_']).abs().mean() if FLAGS.len_loss_weight else -1,
        'char/max_idx_max': df['char/max_idx'].max(),
        'char/max_idx': df['char/max_idx'].mean(),
        'char/ori_rate': df['char/ori_rate'].mean(),
        'char/true_rate': df['char/true_rate'].mean(),
        'char/pred_rate': df['char/pred_rate'].mean(),
        'distance': df['distance'].mean(),
        'acc/char': df['acc/char'].mean(),
        'acc/type': df['acc/type'].mean(),
        'acc/first': df['acc/first'].mean(),
        'acc/last': df['acc/last'].mean(),
        'score': max(df['phrase_len_true'].sum() - df['distance'].sum(), 0.) / df['phrase_len_true'].sum(),
      }
      return metrics
    
    metrics = get_metrics(df)
    metrics.update(gezi.dict_suffix(get_metrics(df[df['phrase_type'] == 'phone']), '/phone'))
    metrics.update(gezi.dict_suffix(get_metrics(df[df['phrase_type'] == 'url']), '/url'))
    metrics.update(gezi.dict_suffix(get_metrics(df[df['phrase_type'] == 'address']), '/address'))
    metrics.update(gezi.dict_suffix(get_metrics(df[df['phrase_dup'] == 1]), '/dup'))
    metrics.update(gezi.dict_suffix(get_metrics(df[df['phrase_dup'] == 0]), '/new'))
    n_frames = 256
    metrics.update(gezi.dict_suffix(get_metrics(df[df['n_frame'] > n_frames]), '/long'))
    metrics.update(gezi.dict_suffix(get_metrics(df[df['n_frame'] <= n_frames]), '/short'))
    
    df.to_csv(f'{FLAGS.model_dir}/eval.csv', index=False)
    df_ = df
    df = df[df.idx < FLAGS.n_infers]
    df = df.sort_values(by=['idx'], ascending=True)
    df2 = df[['sequence_id', 'phrase_type', 'phrase_true', 'phrase_pred', 'phrase_ori', 'phrase_len_true', 'phrase_len_pred', 'phrase_len_pred_', 'distance', 'score']].head(20)
    ic(df2)
    metrics['score/head'] = max(df['phrase_len_true'].sum() - df['distance'].sum(), 0.) / df['phrase_len_true'].sum()
    gezi.set('score/head', metrics['score/head'])
    gezi.set('eval_df', df_)
    # # TODO not sure why but evauate will slow down training, return {} will not slow down....
    # ic(metrics)
    # return {}
    global best_score, bad_count
    if metrics['score'] > best_score:
      best_score = metrics['score']
      bad_count = 0
    else:
      bad_count += 1
      ic(bad_count, best_score, metrics['score'], best_score - metrics['score'])
      # if bad_count >= 10: 
      #   exit(0)
    # ic(metrics)
  if FLAGS.torch and FLAGS.distributed:
    torch.distributed.barrier()
  return metrics

def eval_cls(y_true, y_pred, x, others, is_last=False):       
  metrics = {}
  x.update(others)
  metrics['acc/char'] = np.mean(y_true == (y_pred > 0.))
  metrics['acc/type'] = np.mean(x['phrase_type_'] == np.argmax(x['type_pred'], axis=-1))
  metrics['acc/first'] = np.mean(x['first_char'] == np.argmax(x['first_char_pred'], axis=-1))
  metrics['acc/last'] = np.mean(x['last_char'] == np.argmax(x['last_char_pred'], axis=-1))
  if not FLAGS.len_cls:
    if FLAGS.len_loss == 'bce':
      x['len_pred'] = gezi.sigmoid(x['len_pred'])
    x['pred_len'] = (x['len_pred'] * MAX_PHRASE_LEN + 0.5).astype(int)
  else:
    # x['pred_len'] = np.argmax(x['len_pred'], axis=-1) + 1
    x['pred_len'] = ((gezi.softmax(x['len_pred']) * np.arange(MAX_PHRASE_LEN)).sum(-1) + 0.5 + 1).astype(int)
  metrics['len/l2'] = np.mean((x['phrase_len'] - x['pred_len']) ** 2)
  metrics['len/l1'] = np.mean(abs(x['phrase_len'] - x['pred_len']))
  metrics['len/acc'] = np.mean(x['pred_len'] == x['phrase_len'])
  metrics['log_loss'] = log_loss(y_true, y_pred)
  
  aucs = []
  for i in range(N_CHARS):
    if y_true[:, i].max() == 0 or y_true[:, i].min() == 1:
      continue
    auc = fast_auc(y_true[:, i], y_pred[:, i])
    aucs.append(auc)
    metrics[f'auc/{IDX2CHAR[i + 1]}'] = auc
    
  metrics['auc'] = np.mean(aucs)    
  
  type_pred = np.argmax(others['type_pred'], axis=-1)
  l = []
  for i, (sequence_id, idx, phrase, phrase_type, pred_len) in enumerate(zip(x['sequence_id'], x['idx'], x['phrase'], x['phrase_type'], x['pred_len'])):
    l.append({
      'sequence_id': sequence_id,
      'idx': idx,
      'phrase': phrase.decode('utf-8'),
      'phrase_pred': ''.join([IDX2CHAR[i + 1] for i, p in enumerate(y_pred[i]) if p > 0.5]),
      'phrase_type': phrase_type.decode('utf-8'),
      'phrase_type_pred': CLASSES[type_pred[i]],
      'phrase_len_true': len(phrase.decode('utf-8')),
      'phrase_len_pred': pred_len,
    })
  
  df = pd.DataFrame(l)
  df.to_csv(f'{FLAGS.model_dir}/eval.csv', index=False)
  df_ = df
  df = df[df.idx < 20]
  df = df.sort_values(by=['idx'], ascending=True)
  ic(df.head(20))
  gezi.set('eval_df', df_)
  
  return metrics

def get_eval_fn():
  if FLAGS.task == 'seq':
    return evaluate
  else:
    return eval_cls
