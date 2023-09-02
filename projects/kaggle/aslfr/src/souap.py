#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('/work/pikachu/utils/')
sys.path.append('/work/pikachu/third/')
sys.path.append('..')
from gezi.common import *
import gezi
from src import config
from src.config import *
from src import util
from src.dataset import Dataset
from src.eval import eval_seq, calc_score
from src.tf.decode import adjust_pad
import copy

gezi.init_flags()
# physical_devices = tf.config.list_physical_devices('GPU')
# if physical_devices:
#   tf.config.set_visible_devices([physical_devices[x] for x in range(4)], 'GPU')
# tf.config.set_soft_device_placement(True)
# for gpu in physical_devices:
#   tf.config.experimental.set_memory_growth(gpu, True)

models = [
  'final-14layers.finetune.finetune_epochs-15',
  'final-14layers.finetune.finetune_epochs-20',
  'final-14layers.finetune',
  # 'final-14layers.finetune',
]

model = models[-1]
model_dir = f'../working/offline/30/0/{model}'

# df = pd.read_csv(f'{model_dir}/eval.csv')
# from collections import defaultdict
# m, m_ = defaultdict(int), defaultdict(int)
# count, count_ = 0, 0
# for phrase, phrase_ in zip(df.phrase_true, df.phrase_pred):
#   for c in phrase:
#     m[c] += 1
#     count += 1
#   for c in phrase_:
#     m_[c] += 1
#     count_ += 1
    
# for c in CHAR2IDX:
#   print(c, m[c], m_[c], m_[c] / m[c] if m[c] else m_[c] / 0.1)
#   print(count, count_, count_ / count)
  
# scales = np.ones(61)
# for i, c in enumerate(CHAR2IDX):
#   if m[c] > 100 and m_[c] > 10:
#     scales[i + 1] = (count_ / count) / (m_[c] / m[c])
    
# ic(scales)
# FLAGS.scales = list(scales)
FLAGS.scales = []

FLAGS.mn = 'souap'
FLAGS.model_dir = '../working/offline/30/0/souap'
gezi.try_mkdir(FLAGS.model_dir)
config.init()
mt.init()
config.show()

gezi.restore_configs(model_dir)

FLAGS.mn = 'souap'
FLAGS.model_dir = '../working/offline/30/0/souap'
FLAGS.pad_rate = 0.9
FLAGS.distributed = False

record_dir = f'{FLAGS.root}/tfrecords/0.1'
records_pattern = f'{record_dir}/*.tfrec'
files = gezi.list_files(records_pattern) 
FLAGS.valid_files = [x for x in files if int(os.path.basename(x).split('.')[0]) % FLAGS.folds == FLAGS.fold]
dataset = Dataset('valid', files=FLAGS.valid_files)
datas = dataset.make_batch(FLAGS.batch_size, return_numpy=True)
from src.torch.dataset import get_dataloaders
train_loader, eval_loader = get_dataloaders()

config.show()
model = util.get_model()
ic(model_dir)
lele.load_weights(model, f'{model_dir}/model.pt')

ic(model)
ic(len(eval_loader))
num_steps = dataset.num_steps

# metrics = eval_seq(eval_loader, model, num_steps)
# ic(metrics['score'])


model_dict = model.state_dict()
soups = {key:[] for key in model_dict}

for i, model_dir in tqdm(enumerate(models), total=len(models)):
  ic(model_dir)
  lele.load_weights(model, f'../working/offline/30/0/{model_dir}/model.pt')
  metrics = eval_seq(eval_loader, model, num_steps)
  ic(metrics['score'])
  for i, (k, v) in enumerate(model.state_dict().items()):
    if i == 0:
      ic(model_dir, k, v)
    soups[k].append(copy.deepcopy(v))

for k, v in soups.items():
  ic(k, v, len(v))
  break
soups = {k:(torch.sum(torch.stack(v), axis = 0) / len(v)).type(v[0].dtype) for k, v in soups.items() if len(v) != 0}
model_dict.update(soups)
model.load_state_dict(model_dict)

metrics = eval_seq(eval_loader, model, num_steps)
ic(metrics['score'])

gezi.dump_flags(FLAGS.model_dir)
gezi.save_model(model, FLAGS.model_dir, fp16=True)
