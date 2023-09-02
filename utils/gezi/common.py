#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   common.py
#        \author   chenghuige  
#          \date   2022-04-19 11:42:25.883552
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.simplefilter("ignore", RuntimeWarning) 

import sys 
import os
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import glob

from absl import app, flags
FLAGS = flags.FLAGS

import math
import numpy as np
import random
import scipy
import sklearn
import pandas as pd
from functools import partial
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterable
import json
import re
from pathlib import Path
import itertools
try:
  import pickle5 as pickle
except Exception:
  try:
    # import dill as pickle
    import cPickle as pickle
  except Exception:
    import pickle
import dill
import shutil
import gzip
import gc
import subprocess
from sklearn.preprocessing import normalize

# use some cpu memory
try:
  import cudf
except Exception:
  pass

def use_itables():
  from itables import init_notebook_mode
  init_notebook_mode(all_interactive=True)

from gezi import tqdm, rtqdm, logging
# increase +1G cpu memory...
tqdm.pandas()
logger = logging.logger
logger2 = logging.logger2
from icecream import ic

from multiprocessing import Pool, Manager, cpu_count
from joblib.externals.loky import set_loky_pickler
set_loky_pickler("dill")
from joblib import Parallel, delayed
try:
  import pymp 
except Exception:
  pass

try:
  from pandarallel import pandarallel as pdl
  pdl.initialize(nb_workers=16, progress_bar=True)
except Exception:
  pass

import tensorflow as tf
import torch
from torch import nn
import torch.nn.functional as F

try:
  from rich_dataframe import prettify
except Exception:
  pass
from IPython.display import display_html, display, HTML
import plotly.express as px

import melt
import melt as mt
import lele
import husky 
import gezi
import gezi.plot

PERCENTILES = [.25,.5,.75,.9,.95,.99]
PERCENTILES2 = [.25,.5,.75,.9,.95,.99,.999]

SPECIAL_CHAR = 'ʶ'
