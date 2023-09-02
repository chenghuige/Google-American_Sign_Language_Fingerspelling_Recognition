#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   config.py
#        \author   chenghuige  
#          \date   2023-06-20 08:15:53.335725
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 

MODEL_NAME = 'aslfr'
flags.DEFINE_string('RUN_VERSION', '7', '')
flags.DEFINE_string('SUFFIX', '', '')
flags.DEFINE_string('mark', 'train', '')

flags.DEFINE_string('root', '../input/asl-fingerspelling', '')
flags.DEFINE_integer('fold_seed', 1229, '')
flags.DEFINE_integer('fold_workers', 10, '')
flags.DEFINE_alias('fs', 'fold_seed')
flags.DEFINE_integer('aug_seed', None, '')

flags.DEFINE_string('records_version', 'v0', '')
flags.DEFINE_alias('rv', 'records_version')
flags.DEFINE_bool('mix_sup', False, '')
flags.DEFINE_string('sup_version', 'sup', '')
flags.DEFINE_float('sup_weight', 0.1, '')

flags.DEFINE_string('obj', None, '')

flags.DEFINE_integer('n_frames', 128, '')
flags.DEFINE_integer('max_phrase_len', 32, '')
flags.DEFINE_integer('max_files', 0, '')

# For tfrecords and preprocess, always change default value to your best choice after experiments
flags.DEFINE_bool('filter_nan_frames', True, '')
flags.DEFINE_bool('always_filter_nan_frames', False, '')
flags.DEFINE_bool('filter_nan_hands', False, '')
flags.DEFINE_string('trunct_method', 'resize', 'for trunct of frames')
flags.DEFINE_bool('always_resize', False, '')
flags.DEFINE_bool('ignore_nan_frames', False, '')

# flags.DEFINE_bool('decode_phrase_type', False, '')
flags.DEFINE_bool('decode_phrase_type', True, '')

flags.DEFINE_bool('use_decoder', True, '')
flags.DEFINE_bool('masked_loss', True, '')
flags.DEFINE_bool('weighted_loss', False, '')
flags.DEFINE_bool('log_weights', False, '')

flags.DEFINE_bool('norm_frames', False, '')
flags.DEFINE_float('wd_ratio', 0, '')
flags.DEFINE_float('mlp_drop', 0.3, '')
flags.DEFINE_float('cls_drop', 0., '')
flags.DEFINE_float('rnn_drop', 0., '')
flags.DEFINE_bool('dominant_emb', True, '')
flags.DEFINE_string('emb_init', 'zeros', '')
flags.DEFINE_float('layer_norm_eps', 1e-6, '')
flags.DEFINE_integer('encoder_units', 256, '')
flags.DEFINE_integer('decoder_units', 256, '')
flags.DEFINE_integer('encoder_layers', 2, '')
flags.DEFINE_integer('decoder_layers', 2, '')
flags.DEFINE_integer('mhatt_heads', 8, '')
flags.DEFINE_float('mhatt_drop', 0., '')
flags.DEFINE_float('mhatt_depth_ratio', 1., '')
flags.DEFINE_integer('mlp_ratio', 4, '')

flags.DEFINE_bool('rnn_encoder', False, '')
flags.DEFINE_string('rnn', 'LSTM', '')

flags.DEFINE_string('landmark_emb', 'dense', '')

# Lips Landmark Face Ids
LIPS_LANDMARK_IDXS = np.array([
        61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    ])

LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE

CHAR2IDX = gezi.load(f'../input/asl-fingerspelling/character_to_prediction_index.json')
N_CHARS = len(CHAR2IDX)
# CHAR2IDX = {k: v + 1 for k, v in CHAR2IDX.items()}
PAD_TOKEN = N_CHARS
SOS_TOKEN = N_CHARS + 1 # Start Of Sentence
EOS_TOKEN = N_CHARS + 2 # End Of Sentence

# ADDR_TOKEN = N_CHARS + 3
# URL_TOKEN = N_CHARS + 4
# PHONE_TOKEN = N_CHARS + 5
CHAR2IDX['<SOS>'] = SOS_TOKEN
CHAR2IDX['<EOS>'] = EOS_TOKEN 
CHAR2IDX['<PAD>'] = PAD_TOKEN

# ## seems help a bit.. but not much
# # TODO HACK only use it if FLAGS.decode_phrase_type otherwise comment it out
CHAR2IDX['<ADDRESS>'] = EOS_TOKEN + 1
CHAR2IDX['<URL>'] = EOS_TOKEN + 2
CHAR2IDX['<PHONE>'] = EOS_TOKEN + 3

VOCAB_SIZE = len(CHAR2IDX)
IDX2CHAR = {v: k for k, v in CHAR2IDX.items()}
ic(VOCAB_SIZE)
ic(len(IDX2CHAR))
# ic(CHAR2IDX)
# ic(IDX2CHAR)
## only left and right hands
# N_COLS = 84
## add lips
# N_COLS = 164
## add pose
N_COLS = 184
## add z
# N_COLS = 276
# N_COLS = 126
# N_COLS = 104
N_FRAMES = 128

STATS = {}
# configs = {}
PHRASE_TYPES = {
  'address': 0,
  'url': 1,
  'phone': 2,
}

def init(gen_records=False):
  RUN_VERSION = FLAGS.RUN_VERSION 
  # SUFFIX = f'.{FLAGS.SUFFIX}'
  SUFFIX = f'.{FLAGS.SUFFIX}' if FLAGS.SUFFIX else ''
  FLAGS.run_version = FLAGS.run_version or RUN_VERSION
  
  if FLAGS.online:
    FLAGS.allow_train_valid = True
  FLAGS.run_version += f'/{FLAGS.fold}'
  
  pres = ['offline', 'online']
  pre = pres[int(FLAGS.online)]
  model_dir = f'../working/{pre}/{FLAGS.run_version}/model'  
  FLAGS.model_dir = FLAGS.model_dir or model_dir
  if FLAGS.mn == 'model':
    FLAGS.mn = ''
  has_mn = True
  if not FLAGS.mn:  
    has_mn = False 
    ignores = ['SUFFIX', 'tf']
    mt.model_name_from_args(ignores=ignores)
    FLAGS.mn += SUFFIX 
     
  if FLAGS.log_all_folds or FLAGS.fold == 0:
    wandb = True
    if FLAGS.wandb is None:
      FLAGS.wandb = wandb
    FLAGS.wandb_project = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
  FLAGS.show_keys = ['score', 'type_acc', 
                     'score/address', 'score/url', 'score/phone', 
                     'score/dup', 'score/new', 'score/long', 'score/short',
                     'score/head']
  
  FLAGS.folds = 4
  bs = 128 
  FLAGS.batch_size = FLAGS.batch_size or bs
  lr = 1e-3
  FLAGS.learning_rate = FLAGS.learning_rate or lr
  epochs = 30
  FLAGS.epochs = FLAGS.epochs or epochs
  
  # if not FLAGS.online:
  #   FLAGS.nvs = FLAGS.nvs or FLAGS.ep
  
  optimizer = 'bert-adam' 
  FLAGS.optimizer = FLAGS.optimizer or optimizer
  # FLAGS.opt_eps = 1e-7
  
  FLAGS.fp16 = False
  
  # set 5 should be better but in tf keras it will slow down training speed after first eval... TODO
  vie_ = 5
  vie = vie_ if not FLAGS.online else FLAGS.epochs
  FLAGS.vie = FLAGS.vie or vie 
  FLAGS.sie = FLAGS.vie
  
  # FLAGS.batch_parse = False
  
  if FLAGS.mark != 'train':
    FLAGS.records_version = FLAGS.mark
  if FLAGS.tf_dataset:
    records_pattern = f'{FLAGS.root}/tfrecords/{FLAGS.records_version}/*.tfrec'
    files = gezi.list_files(records_pattern) 
    if FLAGS.mix_sup:
      files += gezi.list_files(f'{FLAGS.root}/tfrecords/{FLAGS.sup_version}/*.tfrec')
      np.random.shuffle(files)
      
    if FLAGS.online:
      FLAGS.train_files = files
    else:
      FLAGS.train_files = [x for x in files if int(os.path.basename(x).split('.')[0]) % FLAGS.folds != FLAGS.fold]

    # records_pattern = f'{FLAGS.root}/tfrecords/v0/*.tfrec'
    # files = gezi.list_files(records_pattern) 
    FLAGS.valid_files = [x for x in files if int(os.path.basename(x).split('.')[0]) % FLAGS.folds == FLAGS.fold]

    ic(len(FLAGS.train_files), len(FLAGS.valid_files))
    
  if FLAGS.norm_frames:
    record_dir = f'{FLAGS.root}/tfrecords/{FLAGS.records_version}'
    means = gezi.load(f'{record_dir}/means.npy')
    stds = gezi.load(f'{record_dir}/stds.npy')
    STATS['means'] = means
    STATS['stds'] = stds
    
  if FLAGS.decode_phrase_type and (not gen_records):
    FLAGS.max_phrase_len += 1
    