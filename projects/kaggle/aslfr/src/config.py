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
RUN_VERSION = '31'
flags.DEFINE_string('SUFFIX', '', '')

flags.DEFINE_bool('tf', False, '')

flags.DEFINE_string('root', '../input/asl-fingerspelling', '')
flags.DEFINE_integer('fold_seed', 1229, '')
flags.DEFINE_integer('fold_workers', 10, '')
flags.DEFINE_alias('fs', 'fold_seed')
flags.DEFINE_integer('aug_seed', None, '')
flags.DEFINE_bool('group_fold', True, '')
flags.DEFINE_alias('gf', 'group_fold')

flags.DEFINE_bool('save_fp16', False, '')
flags.DEFINE_bool('small', False, '')
flags.DEFINE_bool('tiny', False, '')
flags.DEFINE_bool('large', False, '')

flags.DEFINE_string('records_version', '0', '')
flags.DEFINE_alias('rv', 'records_version')
flags.DEFINE_bool('mix_sup', True, 'by default mix sup will be better, then can finetune 100 epochs uing lr 1e-4 and --mix_sup=0')
flags.DEFINE_string('sup_version', 'sup', '')
flags.DEFINE_float('sup_weight', 0.1, '')

flags.DEFINE_string('task', None, 'None means default seq task, otherwise means clsassifer task which is simple for test encoder only performance')
flags.DEFINE_string('obj', 'train', 'train means training default model otherwise means sup landmarks training')
flags.DEFINE_string('method', 'encode', 'encode(encoder only arch, now using ctc loss) or seq2seq(encoder + decoder arch)')
flags.DEFINE_string('model', 'encoder', 'encode,seq2seq(task seq) ,classifier(task cls), get Model from src.tf.models.encoder for FLAGS.method==encode or src.tf.models.seq2seq for FLAGS.method==seq2seq')
flags.DEFINE_string('encoder', 'conv1d_transformer', 'get Encoder from src.tf.encoders.conv1d_transformer')
flags.DEFINE_string('decoder', 'transformer', 'get Decoder from src.tf.decoder')
flags.DEFINE_string('embedding', None, '')
flags.DEFINE_bool('emb_batchnorm', None, '')
flags.DEFINE_string('loss', None, 'ctc or softmax')
flags.DEFINE_string('ctc_loss', 'classic', 'ori or classic or simple, ori tensorflow ctc loss too slow')
flags.DEFINE_string('ctc_decode_method', 'greedy', 'greedy or beam')
flags.DEFINE_float('pad_rate', 0.6, '')
flags.DEFINE_float('pad_thre', 0.3, '')
flags.DEFINE_bool('short_rule', False, '')
flags.DEFINE_list('scales', [], '')
flags.DEFINE_list('cls_loss_weights', [], '')
flags.DEFINE_string('cls_loss', 'char', '')
flags.DEFINE_bool('len_cls', True, '')
flags.DEFINE_string('len_loss', 'ce', '')
flags.DEFINE_float('len_loss_weight', None, '')
flags.DEFINE_string('cls_pooling', 'latt', '')

flags.DEFINE_bool('decode_phrase_type', False, '')
flags.DEFINE_bool('encode_groupby', False, '')
flags.DEFINE_integer('encode_pool_size', None, '')
flags.DEFINE_integer('encode_out_feats', None, '')
flags.DEFINE_bool('no_eos', False, '')

flags.DEFINE_bool('pad_frames', True, '')
flags.DEFINE_string('pad_method', 'zero', 'zero or resize')
flags.DEFINE_bool('use_masking', False, '')
flags.DEFINE_integer('n_frames', 128, '')
flags.DEFINE_integer('min_frames', 128, '')
flags.DEFINE_integer('max_phrase_len', 32, '')
flags.DEFINE_integer('max_files', 0, '')
flags.DEFINE_string('resize_method', 'bilinear', '')
flags.DEFINE_string('pad_resize_method', 'nearest', '')

# aug
flags.DEFINE_bool('use_aug', False, '')
flags.DEFINE_float('aug_factor', 1., '')
flags.DEFINE_float('flip_rate', 0., 'seems 0.25 better then 0.5? TODO further verify')
flags.DEFINE_bool('pred_flip', False, '')
flags.DEFINE_bool('dominant_flip', False, '')
flags.DEFINE_float('resample_rate', 0., '0.8 better then 0.5 7317 to 7345')
flags.DEFINE_list('resample_range', [0.5, 1.5], '[0.8, 1.2]?')
flags.DEFINE_float('temporal_mask_rate', 0., '')
flags.DEFINE_float('temporal_mask_prob', 0.15, '')
flags.DEFINE_list('temporal_mask_range', [], '')
flags.DEFINE_float('temporal_seq_mask_rate', 0, '')
flags.DEFINE_list('temporal_seq_mask_range', [0.1, 0.2], '')
flags.DEFINE_integer('temporal_seq_mask_max', 2, '')
flags.DEFINE_float('spatio_mask_rate', 0., '')
flags.DEFINE_float('spatio_mask_prob', 0.15, '')
flags.DEFINE_integer('shift_method', 1, '')
flags.DEFINE_float('shift_rate', 0., '')
flags.DEFINE_list('shift_range', [-0.05, 0.05], '')
flags.DEFINE_integer('scale_method', 1, '')
flags.DEFINE_float('scale_rate', 0., '')
flags.DEFINE_list('scale_range', [0.8, 1.2], '')
flags.DEFINE_float('rotate_rate', 0., '')
flags.DEFINE_list('rotate_range', [-15, 15], 'or -15,15 -30,30')
flags.DEFINE_float('affine_rate', 0., '')
flags.DEFINE_float('cutmix_rate', 0., '')
flags.DEFINE_string('cutmix_method', 'concat', '')

# for preprocess
flags.DEFINE_bool('filter_nan_frames', True, '')
flags.DEFINE_bool('always_filter_nan_frames', False, '')
flags.DEFINE_bool('filter_nan_hands', False, '')
flags.DEFINE_integer('nan_hands_method', 0, '0 filter, 1 filter by 50% of the time, 2 leave the first nan hands')
flags.DEFINE_bool('mask_nan_hands', False, '')
flags.DEFINE_string('trunct_method', 'resize', 'for trunct of frames')
flags.DEFINE_bool('always_resize', False, '')
flags.DEFINE_bool('ignore_nan_frames', False, '')
flags.DEFINE_bool('add_pos', False, '')
flags.DEFINE_bool('add_pos_before_resample', False, '')
flags.DEFINE_bool('add_motion', False, '')
flags.DEFINE_bool('add_motion2', False, '')

flags.DEFINE_bool('masked_loss', False, 'weather mask loss for PAD_IDX in seq2seq method')
flags.DEFINE_bool('weighted_loss', False, '')
flags.DEFINE_bool('log_weights', False, '')
flags.DEFINE_bool('add_encoder_loss', False, 'for seq2seq model only')
flags.DEFINE_float('encoder_loss_rate', 1., '')
flags.DEFINE_float('decoder_loss_rate', 1., '')
flags.DEFINE_float('center_loss_rate', 0., '')
flags.DEFINE_bool('ctc_torch_loss', True, 'for torch only if not ctc_torch_loss then * label len, exp show True is much better, TODO maybe tf also similar')
flags.DEFINE_float('ctc_label_smoothing', 0., '')
flags.DEFINE_bool('focal_ctc_loss', False, '')
flags.DEFINE_float('focal_ctc_alpha', 0.5, '')
flags.DEFINE_float('focal_ctc_gamma', 0.5, '')
flags.DEFINE_bool('freeze_encoder', False, '')
flags.DEFINE_bool('random_phrase', False, '')
flags.DEFINE_bool('sup_no_s2s', True, '')
flags.DEFINE_bool('use_decoder_output', True, '')
flags.DEFINE_float('mask_phrase_prob', 0., '')

flags.DEFINE_bool('norm_frames', False, 'batch wise norm for each feat of frame using all frames stats')
flags.DEFINE_bool('concat_frames', False, '')
flags.DEFINE_integer('concat_frames_dim', -2, 'change to -2, which means normed data after nonorm')
flags.DEFINE_bool('norm_frame', False, 'frame wise layer norm for each feat of frame only using all feats of this frame')
flags.DEFINE_bool('norm_hands', False, '')
flags.DEFINE_bool('norm_hands_size', True, '')
flags.DEFINE_float('wd_ratio', 0.2, '')
flags.DEFINE_float('mlp_drop', 0.3, '')
flags.DEFINE_float('cls_drop', 0.1, '')
flags.DEFINE_float('cls_late_drop', 0., '')
flags.DEFINE_bool('cls_mlp', False, '')
flags.DEFINE_bool('dominant_emb', True, '')
flags.DEFINE_string('emb_init', 'zeros', '')
flags.DEFINE_float('layer_norm_eps', 1e-6, '')
flags.DEFINE_integer('encoder_units', 256, '')
flags.DEFINE_integer('decoder_units', 256, '')
flags.DEFINE_integer('encoder_layers', 2, '')
flags.DEFINE_integer('decoder_layers', 2, '')
flags.DEFINE_list('conv1d_ksize_vals', [11, 11, 11], 'change from 11 to 15')
flags.DEFINE_alias('ksize_vals', 'conv1d_ksize_vals')
flags.DEFINE_integer('transformer_layers', 1, '')
flags.DEFINE_bool('trans_emb', True, '')
flags.DEFINE_integer('ff_mult', 4, '')
flags.DEFINE_integer('mhatt_heads', 4, 'torch conformer show 8 much better(5 points) then 4 but slower 7->5')
flags.DEFINE_integer('mhatt_dimhead', 64, '')
flags.DEFINE_float('mhatt_drop', 0.2, '')
flags.DEFINE_float('mhatt_depth_ratio', 1., '')
flags.DEFINE_integer('conv1d_expansion_factor', 2, '')
flags.DEFINE_integer('mlp_ratio', 4, '')
flags.DEFINE_bool('use_eca', True, 'exp show True is better')
flags.DEFINE_bool('inst_drop', True, 'noise_shape (None,1,1) per inst drop, Stochastic Depth')
flags.DEFINE_bool('dynamic_inst_drop', False, '')
flags.DEFINE_float('inst_drop_rate', 0.2, '')
flags.DEFINE_float('skip_factor', 1., '')
flags.DEFINE_bool('relpos_att', False, '')
flags.DEFINE_integer('relpos_att_layers', 0, '')
flags.DEFINE_bool('allow_einsum', False, '')
flags.DEFINE_integer('relpos_combine_mode', 0, '0 rope + nonrope, 1 norelpos + nonrope, 2 norelpos + rope')
flags.DEFINE_bool('time_reduce', False, '')
flags.DEFINE_bool('share_reduce', False, '')
flags.DEFINE_integer('time_reduce_idx', 4, '')
flags.DEFINE_list('reduce_idxes', [], '')
flags.DEFINE_integer('time_kernel_size', 5, '')
flags.DEFINE_integer('time_stride', 2, '')
flags.DEFINE_string('time_reduce_pooling', 'avg', 'avg or conv or max, avg is fine')
flags.DEFINE_string('time_reduce_method', 'conv', 'conv or avg, used in conformer model')
flags.DEFINE_bool('causal_mask', False, '')
flags.DEFINE_string('scaling_type', 'dynamic', 'None linear or dynamic, seems dynamic is best')
flags.DEFINE_float('scaling_factor', 0.5, '')
flags.DEFINE_integer('nonrope_layers', 0, '')
flags.DEFINE_integer('relpos_att_stride', 2, '')
flags.DEFINE_float('global_drop', None, 'Notice using instance drop 0.2 here for all other drop set to 0 is best..')
flags.DEFINE_float('attn_drop', 0., '')
flags.DEFINE_float('ff_drop', 0., '')
flags.DEFINE_float('conv_drop', 0., '')
flags.DEFINE_integer('subsample_factor', 1, '')
flags.DEFINE_bool('inter_ctc', False, '')
flags.DEFINE_float('inter_ctc_rate', 0.3, '')
flags.DEFINE_bool('inter_ctc_pooling', True, '')

flags.DEFINE_bool('rnn_encoder', False, '')
flags.DEFINE_string('rnn', 'LSTM', '')
flags.DEFINE_float('rnn_drop', 0.2, '')
flags.DEFINE_string('rnn_merge', 'sum', '')
flags.DEFINE_bool('rnn_batchnorm', False, '')
flags.DEFINE_integer('rnn_block_layers', 1, '')

flags.DEFINE_string('landmark_emb', 'dense', '')
flags.DEFINE_integer('nxtvald_clusters', 0, '')

flags.DEFINE_string('arch', None, '')
flags.DEFINE_bool('save_intermediate', False, '')
flags.DEFINE_bool('torch2tf', False, '')
flags.DEFINE_bool('torch2tf_convert', False, '')
flags.DEFINE_bool('force_convert', False, '')
flags.DEFINE_bool('no_convert', False, '')
flags.DEFINE_bool('convert_only', False, '')
flags.DEFINE_integer('convert_trace', 0, '')
flags.DEFINE_bool('quantize', False, '')
flags.DEFINE_bool('init_df', False, '')
flags.DEFINE_bool('keras_init', False, '')

flags.DEFINE_bool('use_z', True, '')
flags.DEFINE_integer('n_infers', 20, '')
flags.DEFINE_bool('force_flip', False, '')
flags.DEFINE_bool('clip_input', False, '')
flags.DEFINE_bool('clip_xy', False, '')

flags.DEFINE_bool('awp', False, '')
flags.DEFINE_bool('finetune', False, '')
flags.DEFINE_integer('finetune_epochs', None, '')
flags.DEFINE_float('finetune_lr', None, '')
flags.DEFINE_float('finetune_pad_rate', None, '')

flags.DEFINE_bool('stable_train', False, '')


LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE

LIP = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
ic(len(LIP))
LLIP = [84,181,91,146,61,185,40,39,37,87,178,88,95,78,191,80,81,82]
RLIP = [314,405,321,375,291,409,270,269,267,317,402,318,324,308,415,310,311,312]
MID_LIP = [i for i in LIP if i not in LLIP + RLIP]
ic(len(LLIP), len(RLIP), len(MID_LIP))

NOSE=[
    1,2,98,327
]
LNOSE = [98]
RNOSE = [327]
MID_NOSE = [i for i in NOSE if i not in LNOSE + RNOSE]

LEYE = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    466, 388, 387, 386, 385, 384, 398,
]
REYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
]

N_HAND_POINTS = 21
N_POSE_POINTS = len(LPOSE)
N_LIP_POINTS = len(LLIP)
N_EYE_POINTS = len(LEYE)
N_NOSE_POINTS = len(LNOSE)
N_MID_POINTS = len(MID_LIP + MID_NOSE)

# TODO next version will try to add MID points and also like eye, nose

SEL_COLS = []
for i in range(N_HAND_POINTS):
  SEL_COLS.extend([f'x_left_hand_{i}', f'y_left_hand_{i}', f'z_left_hand_{i}'])
for i in range(N_HAND_POINTS):
  SEL_COLS.extend([f'x_right_hand_{i}', f'y_right_hand_{i}', f'z_right_hand_{i}'])
for i in LPOSE:
  SEL_COLS.extend([f'x_pose_{i}', f'y_pose_{i}', f'z_pose_{i}'])
for i in RPOSE:
  SEL_COLS.extend([f'x_pose_{i}', f'y_pose_{i}', f'z_pose_{i}'])
for i in LLIP:
  SEL_COLS.extend([f'x_face_{i}', f'y_face_{i}', f'z_face_{i}'])
for i in RLIP:
  SEL_COLS.extend([f'x_face_{i}', f'y_face_{i}', f'z_face_{i}'])

for i in LEYE:
  SEL_COLS.extend([f'x_face_{i}', f'y_face_{i}', f'z_face_{i}'])
for i in REYE:
  SEL_COLS.extend([f'x_face_{i}', f'y_face_{i}', f'z_face_{i}'])
  
for i in LNOSE:
  SEL_COLS.extend([f'x_face_{i}', f'y_face_{i}', f'z_face_{i}'])
for i in RNOSE:
  SEL_COLS.extend([f'x_face_{i}', f'y_face_{i}', f'z_face_{i}'])
  
for i in MID_LIP:
  SEL_COLS.extend([f'x_face_{i}', f'y_face_{i}', f'z_face_{i}'])
for i in MID_NOSE:
  SEL_COLS.extend([f'x_face_{i}', f'y_face_{i}', f'z_face_{i}'])
    
N_COLS = len(SEL_COLS)
ic(N_COLS)

CHAR2IDX = gezi.load(f'../input/asl-fingerspelling/character_to_prediction_index.json')
CHAR2IDX = {k: v + 1 for k, v in CHAR2IDX.items()}
# N_CHARS is 59 here
N_CHARS = len(CHAR2IDX)
ic(N_CHARS)

PAD_IDX = 0
SOS_IDX = PAD_IDX # Start Of Sentence
EOS_IDX = N_CHARS + 1 # End Of Sentence
ic(PAD_IDX, SOS_IDX, EOS_IDX)

PAD_TOKEN = '<PAD>'
SOS_TOKEN = PAD_TOKEN
EOS_TOKEN = '<EOS>'

CHAR2IDX[PAD_TOKEN] = PAD_IDX
CHAR2IDX[EOS_TOKEN] = EOS_IDX 

ADDRESS_TOKEN = '<ADDRESS>'
URL_TOKEN = '<URL>'
PHONE_TOKEN = '<PHONE>'
SUP_TOKEN = '<SUP>'

# ## seems help a bit.. but not much
# # TODO HACK only use it if FLAGS.decode_phrase_type otherwise comment it out
# CHAR2IDX[ADDRESS_TOKEN] = EOS_IDX + 1
# CHAR2IDX[URL_TOKEN] = EOS_IDX + 2
# CHAR2IDX[PHONE_TOKEN] = EOS_IDX + 3
# CHAR2IDX[SUP_TOKEN] = EOS_IDX + 4

# HOT_WORDS = [
#   'https://',
#   'www.', 
#   '.com',
#   '.org',
#   '.net',
# ]

# for i, word in enumerate(HOT_WORDS):
#   CHAR2IDX[word] = EOS_IDX + 1 + i

VOCAB_SIZE = len(CHAR2IDX)
IDX2CHAR = {v: k for k, v in CHAR2IDX.items()}
ic(VOCAB_SIZE)
ic(len(IDX2CHAR))

STATS = {}
CLASSES = [
  'address', 
  'url', 
  'phone', 
  'sup',
  ]
PHRASE_TYPES = dict(zip(CLASSES, range(len(CLASSES))))
N_TYPES = len(CLASSES)
MAX_PHRASE_LEN = 32

def get_vocab_size():
  # assert not FLAGS.decode_phrase_type
  assert not FLAGS.no_eos
  vocab_size = VOCAB_SIZE
  # if not FLAGS.decode_phrase_type:
  #   vocab_size -= N_TYPES
  # if FLAGS.no_eos:
  #   vocab_size -= 1
  return vocab_size

def get_n_cols(no_motion=False, use_z=None):
  n_cols = N_COLS
  if use_z is None:
    use_z = FLAGS.use_z
  
  if FLAGS.concat_frames:
    assert FLAGS.norm_frames
    n_cols += N_COLS
  
  # if FLAGS.norm_frame:
  #   n_cols += N_COLS
  
  if FLAGS.norm_hands:
    n_cols += (N_HAND_POINTS - 1) * 2 * 3
    
  if FLAGS.add_motion:
    n_cols += N_HAND_POINTS * 2 * 2 * 3

  if not use_z:
    n_cols = n_cols // 3 * 2
    
  if FLAGS.add_pos:
    n_cols += 1
  
  return n_cols
  # if no_motion:
  #   return n_cols
  
  # rate = 1
  # if FLAGS.add_motion:
  #   rate += 1
  # if FLAGS.add_motion2:
  #   rate += 1
  # n_cols *= rate
  
  # return n_cols

def init(gen_records=False):
  SUFFIX = f'.{FLAGS.SUFFIX}' if FLAGS.SUFFIX else ''
  FLAGS.run_version = FLAGS.run_version or RUN_VERSION
  
  assert FLAGS.obj in ['train', 'sup']
  if FLAGS.obj != 'train':
    FLAGS.online = True
  
  if FLAGS.online:
    FLAGS.allow_train_valid = True
  FLAGS.run_version += f'/{FLAGS.fold}'
  
  if not FLAGS.tf:
    FLAGS.torch = True
  FLAGS.cache = True
  FLAGS.disable_valid_dataset = True
  if not FLAGS.torch:
    FLAGS.tf_dataset = True
    FLAGS.tfrecords = True
  else:
    if FLAGS.tf_dataset:
      FLAGS.tfrecords = True
  
    if not FLAGS.distributed:
      if FLAGS.torch_compile is None:
        FLAGS.torch_compile = True
      
  # if FLAGS.torch:
  #   # 这里因为用手动loop eval所以每个worker都访问完整的eval dataset
  #   FLAGS.eval_distributed = False
  FLAGS.find_unused_parameters = False
  
  pres = ['offline', 'online']
  pre = pres[int(FLAGS.online)]
  model_dir = f'../working/{pre}/{FLAGS.run_version}/model'  
  FLAGS.model_dir = FLAGS.model_dir or model_dir
  if FLAGS.mn == 'model':
    FLAGS.mn = ''
  has_mn = True
  if not FLAGS.mn:  
    has_mn = False 
    ignores = ['RUN_VERSION', 'SUFFIX', 'tf', 'ctc_decode_method', 
               'pred_flip', 'pad_rate', 'pad_thre', 'short_rule',
               'save_intermediate', 
               'torch2tf_convert', 'force_convert', 'no_convert', 'convert_only', 'convert_trace',
               'torch']
    mt.model_name_from_args(ignores=ignores)
    # if '.torch' not in FLAGS.mn:
    #   if FLAGS.torch:
    #     FLAGS.mn += '.torch'
    # if '.tf' not in FLAGS.mn:
    #   if FLAGS.tf:
    #     FLAGS.mn += '.tf'
    FLAGS.mn += SUFFIX 
  
  if FLAGS.online:
    # FLAGS.awp = True
    if FLAGS.epochs is None:
      FLAGS.epochs = 300
    
  epochs = 300
  FLAGS.epochs = FLAGS.epochs or epochs
  
  if FLAGS.finetune:
    if not FLAGS.pretrain:
      if FLAGS.finetune_epochs:
        FLAGS.pretrain = FLAGS.mn.replace(f'.finetune_epochs-{FLAGS.finetune_epochs}', '').replace('.finetune', '')
      else:
        FLAGS.pretrain = FLAGS.mn.replace('.finetune', '')
        
    FLAGS.vie = 5
    FLAGS.awp = True
    FLAGS.lr = 1e-4 if not FLAGS.finetune_lr else FLAGS.finetune_lr
    FLAGS.mix_sup = False
    FLAGS.pad_rate = 0.8 if FLAGS.finetune_pad_rate is None else FLAGS.finetune_pad_rate
    if not FLAGS.finetune_epochs:
      FLAGS.epochs = 10
    else:
      FLAGS.epochs = FLAGS.finetune_epochs
    FLAGS.init_epoch = -1
    
  if FLAGS.awp:
    FLAGS.awp_train = True
    if FLAGS.adv_start_epoch is None:
      # FLAGS.adv_start_epoch = 15 if not FLAGS.pretrain else int(FLAGS.epochs * 0.15 + 0.5)
      FLAGS.adv_start_epoch = int(FLAGS.epochs * 0.15 + 0.5)
    if FLAGS.adv_lr is None:
      FLAGS.adv_lr = 0.2
      # if FLAGS.epochs <= 100:
      #   FLAGS.adv_lr = 0.2
      # else:
      #   FLAGS.adv_lr = 0.1
    FLAGS.adv_eps = 0
    # FLAGS.pretrain_restart = 0
  
  if FLAGS.fp16 is None:
    if FLAGS.torch:
      FLAGS.fp16 = True
      if FLAGS.awp_train:
        logger.info('awp_train set to fp16 False for torch, otherwise will cause nan')
        FLAGS.fp16 = False
    else:
      FLAGS.fp16 = False
     
  if FLAGS.log_all_folds or FLAGS.fold == 0:
    wandb = True
    if FLAGS.wandb is None:
      FLAGS.wandb = wandb
    FLAGS.wandb_project = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

  folds = 4
  FLAGS.folds = FLAGS.folds or folds
  bs = 128 
  FLAGS.batch_size = FLAGS.batch_size or bs
  # 2023-0805 change from 1e-3 to 2e-3
  lr = 2e-3
  FLAGS.learning_rate = FLAGS.learning_rate or lr
  
  if not FLAGS.torch:
    optimizer = 'bert-adam' 
    FLAGS.optimizer = FLAGS.optimizer or optimizer
    # FLAGS.opt_eps = 1e-7
  else:
    # change from adamw back to adam
    # optimizer = 'adamw' 
    optimizer = 'Adam'
    FLAGS.optimizer = FLAGS.optimizer or optimizer
    # FLAGS.opt_eps = 1e-7
    FLAGS.opt_eps = 1e-6
  
    scheduler = 'linear' 
    FLAGS.scheduler = FLAGS.scheduler or scheduler
    
  # set 5 should be better but in tf keras it will slow down training speed after first eval... TODO
  vie_ = 5 if not FLAGS.online else 100
  # vie = vie_ if not FLAGS.online else FLAGS.epochs
  vie = vie_
  FLAGS.vie = FLAGS.vie or vie 
  FLAGS.sie = FLAGS.sie or 1
  
  FLAGS.batch_parse = False
  
  if FLAGS.group_fold:
    FLAGS.records_version += '.1'
    
  train_obj_version = FLAGS.records_version
  if FLAGS.obj != 'train':
    FLAGS.records_version = FLAGS.obj
  
  # if FLAGS.tfrecords:
  ## cache tfrecords is faster like 5it -> 5.3it/s but might be killed in epoch 50 TODO 
  ## but even if using cache still on intel is much slow then amd 7+it/s, and amd about 10it/s 2 gpus, TODO
  ## and on amd 80+ gpu usage but on intel only 60 WHY 
  # FLAGS.cache = True 
  records_pattern = f'{FLAGS.root}/tfrecords/{FLAGS.records_version}/*.tfrec'
  files = gezi.list_files(records_pattern) 
  
  if FLAGS.online:
    FLAGS.train_files = files
  else:
    FLAGS.train_files = [x for x in files if int(os.path.basename(x).split('.')[0]) % FLAGS.folds != FLAGS.fold]

  if FLAGS.obj != 'train' or FLAGS.mix_sup:
    records_pattern = f'{FLAGS.root}/tfrecords/{train_obj_version}/*.tfrec'
    files = gezi.list_files(records_pattern) 
    
  FLAGS.valid_files = [x for x in files if int(os.path.basename(x).split('.')[0]) % FLAGS.folds == FLAGS.fold]
  
  if FLAGS.mix_sup:
    FLAGS.train_files += gezi.list_files(f'{FLAGS.root}/tfrecords/{FLAGS.sup_version}/*.tfrec')
    np.random.shuffle(FLAGS.train_files)
    
  if FLAGS.eval_train:
    FLAGS.valid_files = FLAGS.train_files
    
  ic(len(FLAGS.train_files), len(FLAGS.valid_files))
    
  assert FLAGS.method in ['seq2seq', 'encode']
  
  # by default arch now is 7-192-320 
  if FLAGS.small:
    FLAGS.arch = '4-128-256'
  if FLAGS.tiny:
    FLAGS.arch = '1-64-64'
  if FLAGS.large:
    FLAGS.arch = '10-192-320'
    
  if FLAGS.arch:
    FLAGS.encoder_layers, FLAGS.encoder_units, FLAGS.n_frames = [int(x) for x in FLAGS.arch.split('-')]
        
  if FLAGS.task is None:
    FLAGS.task = 'seq'
  assert FLAGS.task in ['seq', 'cls']
  if FLAGS.task == 'cls':
    FLAGS.model = 'classifier'
  
  if FLAGS.loss is None:
    FLAGS.loss = 'softmax' if FLAGS.method == 'seq2seq' else 'ctc'
    
  if FLAGS.loss == 'ctc':
    FLAGS.add_encoder_loss = True
    # FLAGS.decode_phrase_type = False
  else:
    FLAGS.pad_rate = 1.
  
  if FLAGS.decode_phrase_type and (not gen_records):
    if not FLAGS.encode_groupby:
      FLAGS.max_phrase_len += 1
    
  # if FLAGS.task == 'seq' and (not gen_records) and (FLAGS.mode is None):
  #   if FLAGS.decode_phrase_type:
  #     assert FLAGS.method == 'seq2seq', 'only use decode phrase type in seq2seq'
    
  if not FLAGS.pad_frames:
    FLAGS.dynamic_pad = True
    
  if FLAGS.task == 'cls':
    if FLAGS.cls_loss:
      m = {
        'char': [1, 0, 0, 0, 0],
        'type': [0, 1, 0, 0, 0],
        'first': [0, 0, 1, 0, 0],
        'last': [0, 0, 0, 1, 0],
        'char-type': [1, 1, 0, 0],
        'len': [0, 0, 0, 0, 1],
      }
      FLAGS.cls_loss_weights = m[FLAGS.cls_loss]
    
  ic(FLAGS.n_frames, FLAGS.encode_pool_size, FLAGS.encode_out_feats)
  if FLAGS.encode_pool_size:
    FLAGS.encode_out_feats = FLAGS.n_frames // FLAGS.encode_pool_size 
  else:
    # FLAGS.encode_pool_size = FLAGS.n_frames // FLAGS.encode_out_feats
    if FLAGS.n_frames <=128:
      FLAGS.encode_out_feats = 64
      FLAGS.encode_pool_size = FLAGS.n_frames // FLAGS.encode_out_feats
    else:
      FLAGS.encode_pool_size = 4
    if FLAGS.time_reduce:
      FLAGS.encode_pool_size = FLAGS.encode_pool_size // FLAGS.time_stride    
  
  ic(FLAGS.n_frames, FLAGS.encode_pool_size, FLAGS.encode_out_feats)
    
  # cur best for 320 0.5 for 256 0.4
  FLAGS.temporal_mask_prob *= (FLAGS.n_frames / 320.)
  
  FLAGS.cls_loss_weights = [float(x) for x in FLAGS.cls_loss_weights]
  FLAGS.resample_range = [float(x) for x in FLAGS.resample_range]
  FLAGS.shift_range = [float(x) for x in FLAGS.shift_range]
  FLAGS.scale_range = [float(x) for x in FLAGS.scale_range]
  FLAGS.rotate_range = [float(x) for x in FLAGS.rotate_range]
  FLAGS.temporal_mask_range = [float(x) for x in FLAGS.temporal_mask_range]
  FLAGS.temporal_seq_mask_range = [float(x) for x in FLAGS.temporal_seq_mask_range]
  FLAGS.conv1d_ksize_vals = [int(x) for x in FLAGS.conv1d_ksize_vals]
  FLAGS.reduce_idxes = [int(x) for x in FLAGS.reduce_idxes]
  FLAGS.scales = [float(x) for x in FLAGS.scales]
  
  if FLAGS.aug_factor != 1:
    FLAGS.shift_rate *= FLAGS.aug_factor
    FLAGS.scale_rate *= FLAGS.aug_factor
    FLAGS.rotate_rate *= FLAGS.aug_factor
    
  if FLAGS.stable_train:
    FLAGS.global_drop = 0.
    FLAGS.inst_drop = 0.
    FLAGS.cls_drop = 0.
    FLAGS.use_aug = False

  if not FLAGS.finetune:
    FLAGS.inter_models = [int(FLAGS.epochs * 0.15), int(FLAGS.epochs * 0.8), int(FLAGS.epochs * 0.9)]
    # FLAGS.inter_models = [int(FLAGS.epochs * 0.8), int(FLAGS.epochs * 0.9), FLAGS.epochs - 20, FLAGS.epochs - 10]
  ic(FLAGS.inter_models)
    
  if FLAGS.task == 'seq':
    FLAGS.show_keys = ['score', 'distance', 'phrase_len_rate', 'len/l1', 'len/l1_', 'len/l1__',
                       'char/max_idx_max', 'char/max_idx', 'char/ori_rate', 'char/true_rate', 'char/pred_rate', 
                       'acc/char', 'acc/type', 'acc/first', 'acc/last',
                      'score/address', 'score/url', 'score/phone', 
                      'score/dup', 'score/new', 'score/long', 'score/short',
                      'score/head']
  else:
    FLAGS.show_keys = ['acc/char', 'acc/type', 'acc/first', 'acc/last', 'auc', 'len/l2', 'len/l1', 'len/acc']
  
def show():
  ic(
    get_vocab_size(),
    get_n_cols(),
    FLAGS.add_pos,
    FLAGS.task,
    FLAGS.obj,
    FLAGS.method,
    FLAGS.model,
    FLAGS.embedding,
    FLAGS.emb_batchnorm,
    FLAGS.pad_rate,
    FLAGS.encoder,
    FLAGS.decoder,
    FLAGS.decode_phrase_type,
    FLAGS.loss,
    FLAGS.add_encoder_loss,
    FLAGS.encoder_loss_rate,
    FLAGS.decoder_loss_rate,
    FLAGS.max_phrase_len,
    FLAGS.n_frames,
    FLAGS.norm_frames,
    FLAGS.pad_frames,
    FLAGS.dynamic_pad,
    FLAGS.encoder_units,
    FLAGS.encoder_layers,
    FLAGS.mhatt_heads,
    FLAGS.conv1d_expansion_factor,
    FLAGS.conv1d_ksize_vals,
    FLAGS.encode_pool_size,
    FLAGS.encode_out_feats,
    FLAGS.decoder_units,
    FLAGS.decoder_layers,
    FLAGS.cls_drop,
    FLAGS.rnn_drop,
    FLAGS.concat_frames,
    FLAGS.use_masking,
    FLAGS.add_pos,
    FLAGS.add_pos_before_resample,
    FLAGS.use_aug,
    FLAGS.flip_rate,
    FLAGS.resample_rate,
    FLAGS.resample_range,
    FLAGS.temporal_mask_rate,
    FLAGS.temporal_mask_prob,
    FLAGS.temporal_seq_mask_rate,
    FLAGS.temporal_seq_mask_range,
    FLAGS.spatio_mask_rate,
    FLAGS.spatio_mask_prob,
    FLAGS.awp_train,
    FLAGS.adv_start_epoch,
    FLAGS.adv_lr,
    FLAGS.adv_eps,
    FLAGS.cls_late_drop,
    FLAGS.latedrop_start_epoch,
    FLAGS.shift_rate,
    FLAGS.shift_range,
    FLAGS.scale_rate,
    FLAGS.scale_range,
    FLAGS.rotate_rate,
    FLAGS.rotate_range,
    FLAGS.affine_rate,
    FLAGS.add_motion,
    FLAGS.add_motion2,
  )
