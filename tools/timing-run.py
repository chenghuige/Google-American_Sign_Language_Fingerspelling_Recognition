#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   run.py
#        \author   chenghuige  
#          \date   2015-02-26 17:48:25.559868
#   \Description  
# ==============================================================================

import sys,os
import time,datetime
from tqdm.auto import tqdm

after = int(sys.argv[1])
command = ' '.join(sys.argv[2:])
print('command:\n', command)
print('after:', after)

start = time.time()
pbar = tqdm(total=after)
cur = 0
while(True):
  now = time.time()
  interval = int((now - start) / 60)
  if interval > after:
    os.system(command)
    break
  pbar.update(interval - cur)
  cur = interval
  time.sleep(60)




 
