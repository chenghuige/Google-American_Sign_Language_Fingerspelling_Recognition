#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   dataset.py
#        \author   chenghuige  
#          \date   2023-08-08 18:41:25.200459
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
from torch.utils.data import IterableDataset as TorchDataset
from src.dataset import Dataset as TfDataset
from src.config import *


def cutmix(frames, n_frames, phrase, phrase_len, 
           frames2, n_frames2, phrase2, phrase2_len):
  # TODO 第一版做最简单的cutmix 第一个phrase从开始出发 第二个phrase 从最后开始截断
  frames = frames[:n_frames]
  frames2 = frames2[:n_frames2]
  phrase = phrase[:phrase_len]
  phrase2 = phrase2[:phrase2_len]
  frac = np.random.uniform(0.5, 0.8)
  n_frames = (n_frames.astype(float) * frac).astype(int)
  frames = frames[:n_frames]
  n_frames2 = (n_frames2.astype(float) * frac).astype(int)
  frames2 = frames2[n_frames2:]
  phrase_len = (phrase_len.astype(float) * frac).astype(int)
  phrase2_len = (phrase2_len.astype(float) * frac).astype(int)
  phrase = phrase[:phrase_len]
  phrase2 = phrase2[phrase2_len:]
  frames = np.concatenate([frames, frames2], axis=0)
  frames = frames[:FLAGS.n_frames]
  n_frames = frames.shape[0]
  if n_frames < FLAGS.n_frames:
    frames = np.concatenate([frames,
                        np.zeros([FLAGS.n_frames - n_frames, frames.shape[1]], dtype=frames.dtype)], axis=0)
  phrase = np.concatenate([phrase, phrase2], axis=0)
  phrase = phrase[:MAX_PHRASE_LEN]
  phrase_len = phrase.shape[0]
  if phrase_len < MAX_PHRASE_LEN:
    phrase = np.concatenate([phrase, np.zeros([MAX_PHRASE_LEN - phrase_len], dtype=phrase.dtype)], axis=0)
  return frames, n_frames, phrase, phrase_len


class Dataset(TorchDataset):
  def __init__(self, subset='eval'):
    self.subset = subset

    if subset == 'train':
      dataset = TfDataset('train', files=FLAGS.train_files)
      # 这里如果分布式 也是正常读取全局batch 然后内部iter分裂
      # 不同于走torch + tf_dataset然后分布式走shard的模式 那种模式当前效果不好 
      # 另外这里的方式速度更快
      datas = dataset.make_batch(mt.batch_size(), 
                             shuffle=True,
                             repeat=True,
                             drop_remainder=True,
                             return_numpy=True)
                            # return_numpy=False)
    else:
      dataset = TfDataset('valid', files=FLAGS.valid_files)
      # 这里只让一个worker做eval 所以类似单gpu情况,因为走custom loop eval dataset方便一些
      datas = dataset.make_batch(mt.eval_batch_size(),  
                             shuffle=False,
                             repeat=False, 
                             drop_remainder=False,
                             return_numpy=False) # if set return numpy = True then you could only visit 1 time..
    
    self.num_instances = dataset.num_instances
    ic(self.num_instances)
    self.num_steps = dataset.num_steps
    ic(self.num_steps)
    
    self.datas = datas
    self.data_iter = iter(datas)
    
  def post_decode(self, x, y):
    if FLAGS.cutmix_rate <= 0:
      return x, y
    frames = x['frames']
    n_frames = x['n_frames']
    phrase = x['phrase_']
    phrase_len = x['phrase_len']
    
    # ic(frames.shape, n_frames.shape, phrase.shape, phrase_len.shape)
    bs = frames.shape[0]
    indexes = np.arange(bs)
    np.random.shuffle(indexes)
    frames_list = []
    n_frames_list = []
    phrase_list = []
    phrase_len_list = []
    for i in range(bs):
      prob = np.random.uniform()
      if prob < FLAGS.cutmix_rate:
        j = indexes[i]
        frame, n_frame, phrase_, phrase_len_ = cutmix(frames[i], n_frames[i], phrase[i], phrase_len[i],
                                                      frames[j], n_frames[j], phrase[j], phrase_len[j]) 
        frames_list.append(frame) 
        n_frames_list.append(n_frame)
        phrase_list.append(phrase_)
        phrase_len_list.append(phrase_len_)
      else:
        frames_list.append(frames[i])
        n_frames_list.append(n_frames[i])
        phrase_list.append(phrase[i])
        phrase_len_list.append(phrase_len[i])

    x['frames'] = np.stack(frames_list, axis=0)
    x['n_frames'] = np.asarray(n_frames_list)
    x['phrase_'] = np.stack(phrase_list, axis=0)
    x['phrase_len'] = np.asarray(phrase_len_list)
    
    # ic(x['frames'].shape, x['n_frames'].shape, x['phrase_'].shape, x['phrase_len'].shape)
    y = x['phrase_']
    return x, y
  
  def __iter__(self):
    if self.subset == 'train':
      while True:
        x, y = next(self.data_iter)
        del x['phrase_type']
        del x['phrase']
        
        # for key in x:
        #   x[key] = x[key].numpy()
        # y = y.numpy()
        x, y = self.post_decode(x, y)
        
        if FLAGS.distributed:
          rank = gezi.get('RANK', 0)
          start = rank * FLAGS.batch_size
          end = start + FLAGS.batch_size
          for key in x:
            x[key] = x[key][start:end]
          y = y[start:end]
        yield x, y
    else:
      for x, y in self.datas:
        for key in x:
          x[key] = x[key].numpy()
        y = y.numpy()
        yield x, y
    
  def __len__(self):
    return self.num_instances
  
def get_dataloaders():
  kwargs = {
      # 'num_workers': FLAGS.num_workers,
      'num_workers': 0, # >0 will hang...
      'pin_memory': FLAGS.pin_memory,
      # 'persistent_workers': FLAGS.persistent_workers,
      'persistent_workers': False,
  }
  
  train_ds = Dataset('train')
  eval_ds = Dataset('eval')
  
  train_dl = torch.utils.data.DataLoader(train_ds,
                                         batch_size=None,
                                         sampler=None,
                                         **kwargs)
  eval_dl = torch.utils.data.DataLoader(eval_ds,
                                        batch_size=None,
                                        sampler=None,
                                        **kwargs)

  return train_dl, eval_dl  
