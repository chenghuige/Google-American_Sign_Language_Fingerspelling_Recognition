#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('capture', '', "import sys\nsys.path.append('/work/pikachu/utils/')\nsys.path.append('/work/pikachu/third/')\nsys.path.append('..')\nfrom gezi.common import *\nimport gezi\nfrom src import config\nfrom src.config import *\nfrom src import util\nfrom src.dataset import Dataset\nfrom src.eval import eval_seq, calc_score\nfrom src.tf.decode import adjust_pad\ngezi.init_flags()\nphysical_devices = tf.config.list_physical_devices('GPU')\nif physical_devices:\n  tf.config.set_visible_devices([physical_devices[x] for x in range(4)], 'GPU')\ntf.config.set_soft_device_placement(True)\nfor gpu in physical_devices:\n  tf.config.experimental.set_memory_growth(gpu, True)\n")


# In[2]:


get_ipython().run_cell_magic('capture', '', "FLAGS.decode_phrase_type = False\nconfig.init()\nrecord_dir = f'{FLAGS.root}/tfrecords/0.1'\nrecords_pattern = f'{record_dir}/*.tfrec'\nfiles = gezi.list_files(records_pattern) \nFLAGS.valid_files = [x for x in files if int(os.path.basename(x).split('.')[0]) % FLAGS.folds == FLAGS.fold]\ndataset = Dataset('valid', files=FLAGS.valid_files)\ndatas = dataset.make_batch(FLAGS.batch_size, return_numpy=True)\nfrom src.torch.dataset import get_dataloaders\ntrain_loader, eval_loader = get_dataloaders()\n")


# In[9]:


models = [
  'final-14layers.finetune',
  'final-14layers.finetune.finetune_epochs-5',
]
model = models[0]
model_dir = f'../working/offline/30/0/{model}'


# In[ ]:


gezi.restore_configs(model_dir)
FLAGS.pad_rate = 0.9
FLAGS.mn = 'souap'
gezi.try_mkdir('../working/offline/30/0/souap')
model = util.get_model()


# In[ ]:


lele.load_weights(model, f'{model_dir}/model.pt')


# In[ ]:


len(eval_loader)


# In[ ]:


get_ipython().run_cell_magic('capture', '', "num_steps = dataset.num_steps\nmetrics = eval_seq(eval_loader, model, num_steps)\nic(metrics['score'])\n")


# In[ ]:


ic(metrics['score'])


# In[ ]:


model_dict = model.state_dict()
soups = {key:[] for key in model_dict}


# In[ ]:


for i, model_dir in tqdm(enumerate(models), total=len(models)):
  lele.load_weights(model, f'../working/offline/30/0/{model_dir}/model.pt')
  for k, v in model.state_dict().items():
    soups[k].append(v)
soups = {k:(torch.sum(torch.stack(v), axis = 0) / len(v)).type(v[0].dtype) for k, v in soups.items() if len(v) != 0}
model_dict.update(soups)
model.load_state_dict(model_dict)


# In[ ]:


get_ipython().run_cell_magic('capture', '', "metrics = eval_seq(eval_loader, model, num_steps)\nic(metrics['score'])\n")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




