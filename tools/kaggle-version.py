#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   kaggle-version.py
#        \author   chenghuige  
#          \date   2023-06-28 16:02:09.256386
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import * 
import gezi

version = sys.argv[1]
x = gezi.load('./dataset-metadata.json')
x['title'] = x['title'].split('model')[0] + 'model'
x['id'] = x['id'].split('model')[0] + 'model'
x['title'] += version
x['id'] += version 
gezi.save(x, './dataset-metadata.json')

