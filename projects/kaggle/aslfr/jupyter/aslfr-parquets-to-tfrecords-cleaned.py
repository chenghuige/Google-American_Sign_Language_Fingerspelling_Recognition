#!/usr/bin/env python
# coding: utf-8

# This notebook prepares the tfrecords for my simple transformer notebook and is almost an identical copy of Rohith Ingilela's [notebook](https://www.kaggle.com/code/irohith/aslfr-preprocess-dataset), with two small changes according to his latest update and remark:
# 1. Only hands landmarks + ten pose landmarks.
# 2. Only movies with a number of frames > twice the length of the phrase.

# In[ ]:


import pandas as pd
import numpy as np
import tensorflow as tf
from skimage.transform import resize
import json
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


# In[ ]:


inpdir = "/kaggle/input/asl-fingerspelling"
df = pd.read_csv(f'{inpdir}/train.csv')
df["phrase_bytes"] = df["phrase"].map(lambda x: x.encode("utf-8"))
#display(df.head())


# In[ ]:


LPOSE = [13, 15, 17, 19, 21]
RPOSE = [14, 16, 18, 20, 22]
POSE = LPOSE + RPOSE

RHAND_LBLS = [f'x_right_hand_{i}' for i in range(21)] + [f'y_right_hand_{i}' for i in range(21)] + [f'z_right_hand_{i}' for i in range(21)]
LHAND_LBLS = [ f'x_left_hand_{i}' for i in range(21)] + [ f'y_left_hand_{i}' for i in range(21)] + [ f'z_left_hand_{i}' for i in range(21)]
POSE_LBLS = [f'x_pose_{i}' for i in POSE] + [f'y_pose_{i}' for i in POSE] + [f'z_pose_{i}' for i in POSE]

X = [f'x_right_hand_{i}' for i in range(21)] + [f'x_left_hand_{i}' for i in range(21)] + [f'x_pose_{i}' for i in POSE]
Y = [f'y_right_hand_{i}' for i in range(21)] + [f'y_left_hand_{i}' for i in range(21)] + [f'y_pose_{i}' for i in POSE]
Z = [f'z_right_hand_{i}' for i in range(21)] + [f'z_left_hand_{i}' for i in range(21)] + [f'z_pose_{i}' for i in POSE]

SEL_COLS = X + Y + Z
FRAME_LEN = 128

X_IDX = [i for i, col in enumerate(SEL_COLS)  if "x_" in col]
Y_IDX = [i for i, col in enumerate(SEL_COLS)  if "y_" in col]
Z_IDX = [i for i, col in enumerate(SEL_COLS)  if "z_" in col]

RHAND_IDX = [i for i, col in enumerate(SEL_COLS)  if "right" in col]
LHAND_IDX = [i for i, col in enumerate(SEL_COLS)  if  "left" in col]
RPOSE_IDX = [i for i, col in enumerate(SEL_COLS)  if  "pose" in col and int(col[-2:]) in RPOSE]
LPOSE_IDX = [i for i, col in enumerate(SEL_COLS)  if  "pose" in col and int(col[-2:]) in LPOSE]

print('SEL_COLS size:' + str(len(SEL_COLS)))


# In[ ]:


train_landmarks = pd.read_parquet('/kaggle/input/asl-fingerspelling/train_landmarks/1019715464.parquet')
keys = train_landmarks.keys()[1:]
train_landmarks.head()


# In[ ]:


'''
np.unique([keys[i].split('_')[1] for i in range(len(keys))])
len(keys)/3
[keys[i].split('_') for i in range(len(keys)) if 'face' in keys[i].split('_')]
'''


# In[ ]:


def load_relevant_data_subset(pq_path):
    return pd.read_parquet(pq_path, columns=SEL_COLS)

counter = 0
for file_id in tqdm(df.file_id.unique()):
    
    print(counter)
    counter+=1
    
    pqfile = f"{inpdir}/train_landmarks/{file_id}.parquet"
    if not os.path.isdir("tfds"): os.mkdir("tfds")
    tffile = f"tfds/{file_id}.tfrecord"
    seq_refs = df.loc[df.file_id == file_id]
    seqs = load_relevant_data_subset(pqfile)
    seqs_numpy = seqs.to_numpy()
    with tf.io.TFRecordWriter(tffile) as file_writer:
        for seq_id, phrase in zip(seq_refs.sequence_id, seq_refs.phrase_bytes):
            frames = seqs_numpy[seqs.index == seq_id]
            
            r_nonan = np.sum(np.sum(np.isnan(frames[:, RHAND_IDX]), axis = 1) == 0)
            l_nonan = np.sum(np.sum(np.isnan(frames[:, LHAND_IDX]), axis = 1) == 0)
            no_nan = max(r_nonan, l_nonan)
            
            if 2*len(phrase)<no_nan:
                features = {SEL_COLS[i]: tf.train.Feature(
                    float_list=tf.train.FloatList(value=frames[:, i])) for i in range(len(SEL_COLS))}
                features["phrase"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[phrase]))
                record_bytes = tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()
                file_writer.write(record_bytes)


# In[ ]:


def decode_fn(record_bytes):
    schema = {COL: tf.io.VarLenFeature(dtype=tf.float32) for COL in SEL_COLS}
    schema["phrase"] = tf.io.FixedLenFeature([], dtype=tf.string)
    return tf.io.parse_single_example(record_bytes, schema)

for file_id in df.file_id[:1]:
    pqfile = f"{inpdir}/train_landmarks/{file_id}.parquet"
    if not os.path.isdir("tfds"): os.mkdir("tfds")
    tffile = f"tfds/{file_id}.tfrecord"
    for batch in tf.data.TFRecordDataset([tffile]).map(decode_fn).take(2):
        print(list(batch.keys())[0])
    break

