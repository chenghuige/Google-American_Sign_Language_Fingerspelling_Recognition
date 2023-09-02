#!/usr/bin/env python
#coding=gbk
# ==============================================================================
#          \file   win-path.py
#        \author   chenghuige  
#          \date   2014-02-22 12:04:40.880360
#   \Description  
# ==============================================================================

import sys,os
from subprocess import *
path = ''

#deal absoulte path
if (len(sys.argv) > 1):
	if not (path.endswith('/') or path == ''):
		path += '/'
	path += sys.argv[1]

if (path.startswith('~')):
	path = path.replace('~','/home/users/chenghuige/')

#deal relative path
if not (path.startswith('/')):
		path2 = Popen('pwd', shell = True, stdout = PIPE).stdout.readline().strip() 
		if (path != ''):
			path = path2 + '/' + path 
		else:
			path = path2 
#to win path
path = path.replace('/home/users/chenghuige/','W:/')
path = path.replace('/','\\')

print(path)
 
