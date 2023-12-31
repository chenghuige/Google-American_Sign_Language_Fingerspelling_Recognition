#!/usr/bin/env python
# ==============================================================================
#          \file   evaluate.py
#        \author   chenghuige  
#          \date   2014-01-04 08:58:40.965360
#   \Description   similary to TLC show : confusion matrix , pic of auc
#                  input is one file : instance,true,probability,assigned,..
#                  for libsvm test, need to file as input feature(.libsvm) and result(.predict) ->svm-evluate.py or svm-gen-evaluate.py first
#                  for tlc the header format is: instance,true, assigned,output, probability 
#                  TODO understancd other output of tlc and add more
# ==============================================================================

import sys,os,glob
import warnings 



warnings.filterwarnings("ignore") 
#import pylab as pl
#import matplotlib.pyplot as pl
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve

DEFINE_boolean('show', False, 'wehter to show the roc pic')
DEFINE_float('thre', 0.5, 'thre for desciding predict')
DEFINE_string('image', 'temp.roc.pr.png', 'output image')
DEFINE_integer('max_num', 20, 'most to deal')
DEFINE_string('regex', '', 'use regex to find files to deal')
DEFINE_string('column', 'probability', 'score index name')
DEFINE_boolean('header', False, 'wehter has header')
DEFINE_boolean('recall', False, 'get precison by recall')
DEFINE_boolean('precision', False, 'get max recall by precison')
DEFINE_integer('label_idx', 0, '')
DEFINE_integer('prob_idx', 1, '')

#@TODO auc remove sklearn dependence 
def evaluate_calc(tp, fp, tn, fn, label_list, predicts):
    num_pos = tp + fn
    num_neg = fp + tn
    total_instance = num_pos + num_neg
    pratio = num_pos * 1.0 / total_instance
    
    #true positive rate
    tpr = tp * 1.0 / num_pos
    tnr = tn * 1.0 / num_neg
    
    #num of predicted positive
    num_pp = tp + fp
    num_pn = fn + tn
    #ture postive accuracy
    tpa = 1
    tna = 1
    if num_pp != 0:
        tpa = tp * 1.0 / num_pp
    if num_pn != 0:
        tna = tn * 1.0 / num_pn
    
    ok_num = tp + tn
    accuracy = ok_num * 1.0 / total_instance
    
    print("""
    TEST POSITIVE RATIO:    %.4f (%d/(%d+%d))
    
    Confusion table:
             ||===============================|
             ||            PREDICTED          |
      TRUTH  ||    positive    |   negative   | RECALL
             ||===============================|
     positive||    %-5d       |   %-5d      | [%.4f] (%d / %d)
     negative||    %-5d       |   %-5d      |  %.4f  (%d / %d) wushang:[%.4f]
             ||===============================|
     PRECISION [%.4f] (%d/%d)   %.4f(%d/%d)
    
    OVERALL 0/1 ACCURACY:        %.4f (%d/%d)
    """%(pratio, num_pos, num_pos, num_neg, tp, fn, tpr, tp, num_pos, fp, tn, tnr, tn, num_neg, 1 - tnr, tpa, tp, num_pp, tna, tn, num_pn, accuracy, ok_num, total_instance))
    #----------------------------------------------------- auc area
    #from sklearn.metrics import roc_auc_score
    #auc = roc_auc_score(label_list, predicts)
    
    fpr_, tpr_, thresholds = roc_curve(label_list, predicts)
    roc_auc = auc(fpr_, tpr_)
    
    print("""
    ACCURACY:            %.4f
    POS. PRECISION:      %.4f
    POS. RECALL:         %.4f
    NEG. PRECISION:      %.4f
    NEG. RECALL:         %.4f
    AUC:                [%.4f]
    """%(accuracy, tpa, tpr, tna, tnr, roc_auc))
    
    #------------------------------------------------------roc curve 
    #pl.clf()
    #pl.plot(fpr_, tpr_, label='%s: (area = %0.4f)' % (file_name, roc_auc))
    #pl.plot([0, 1], [0, 1], 'k--')
    #pl.xlim([0.0, 1.0])
    #pl.ylim([0.0, 1.0])
    #pl.xlabel('False Positive Rate')
    #pl.ylabel('True Positive Rate')
    #pl.title('Roc Curve:')
    #pl.legend(loc="lower right") 

#give precison find the best thre with max recall
def evaluate_precision(label_list, predicts, precision):
    tp_list = [0] * (len(label_list) + 1)
    for i in range(len(label_list)):
        if (label_list[i] == 1):
            tp_list[i + 1] = tp_list[i] + 1
        else:
            tp_list[i + 1] = tp_list[i]
    
    tp = fp = tn = fn = 0
    for i in reversed(range(len(label_list))):
        if tp_list[i + 1] / float(i + 1) >= precision:
            tp = tp_list[i + 1]
            fp = i + 1 - tp 
            print("Choosing thre: %f"%predicts[i])
            break
        else:
            if label_list[i] == 1:
                fn += 1
            else:
                tn += 1
    evaluate_calc(tp, fp, tn, fn, label_list, predicts)

def evaluate_recall(label_list, predicts, recall):
    num_pos = 0
    for label in label_list:
        if label == 1:
            num_pos += 1

    tp = fp = tn = fn = 0
    next_pos = 0
    for i in range(len(label_list)):
        if (label_list[i] == 1):
            tp += 1
        else:
            fp += 1
        now_recall = tp / float(num_pos)
        #print tp, num_pos, recall, now_recall
        if now_recall >= recall:
            next_pos = i + 1
            print("Choosing thre: %f"%predicts[i])
            break
    for i in range(next_pos, len(label_list)):
        if (label_list[i] == 1):
            fn += 1
        else:
            tn += 1
    evaluate_calc(tp, fp, tn, fn, label_list, predicts)

def evaluate(label_list, predicts, assigned_list, file_name):
    #---------------------------------confusion table
    tp = fp = tn = fn = 0
    for i in range(len(label_list)):
      if (assigned_list[i] == 1):
        if (label_list[i] == 1):
          tp += 1
        else:
          fp += 1
      else:
        if (label_list[i] == 1):
          fn += 1
        else:
          tn += 1
    evaluate_calc(tp, fp, tn, fn, label_list, predicts)
                           

def parse_input(input):
  lines = open(input).readlines()
  
  label_idx = FLAGS.label_idx
  output_idx = 3
  probability_idx = FLAGS.prob_idx
  if FLAGS.header: 
    header = lines[0]
    lines = lines[1:]
    
    names = header.split()
    for i in range(len(names)):
      if (names[i].lower() == 'label' or names[i].lower() == 'true'):
        label_idx = i
      if (names[i].lower() == 'output'):
        output_idx = i
      if (names[i].lower() == FLAGS.column.lower()):
        probability_idx = i
  try:
    line_list = [line.strip().split() for line in lines]
    label_list = [int(float((l[label_idx]))) for l in line_list]
    predicts = [float(l[probability_idx]) for l in line_list] 
    #predicts = [float(l[output_idx]) for l in line_list] 
    
    #sort by predicts large to small
    label_list, predicts = (list(x) for x in zip(*sorted(zip(label_list, predicts), key=lambda pair: -pair[1])))

    assigned_list = [int(item > FLAGS.thre) for item in predicts]
    return label_list, predicts, assigned_list 
  except Exception:
    print("label_idx: " + str(label_idx) + " prob_idx: " + str(probability_idx))
    exit(1)
  
def precision_recall(label_list, predicts, file_name):
  # Compute Precision-Recall and plot curve
  precision, recall, thresholds = precision_recall_curve(label_list, predicts)
  area = auc(recall, precision)
  #print("Area Under Curve: %0.2f" % area)
  #pl.clf()
  #pl.plot(recall, precision, label='%s (area = %0.4f)'%(file_name, area))
  #pl.xlabel('Recall')
  #pl.ylabel('Precision')
  #pl.ylim([0.0, 1.05])
  #pl.xlim([0.0, 1.0])
  #pl.title('Precision-Recall curve')
  #pl.legend(loc="lower left")

def main(argv):
  try:
    argv = FLAGS(argv)  # parse flags
  except Exception:
    print('%s\nUsage: %s ARGS\n%s' % (e, sys.argv[0], FLAGS))
    sys.exit(1)
  
  pos = len(argv) - 1
  try:
    FLAGS.thre = float(argv[-1])
    pos -= 1
  except Exception:
    pass
  #---------------------------------thre
  print("Thre: %.4f"%FLAGS.thre)
  #---------------------------------deal input
  l = []
  if (FLAGS.regex != ""):
    print("regex: " + FLAGS.regex)
    l = glob.glob(FLAGS.regex)
    print(l)
  else:
    input = argv[1]
    l = input.split()
  if (len(l) > 1):
    FLAGS.show = True
    if (len(l) > FLAGS.max_num):
      l = l[:FLAGS.max_num]
    #deal with more than 1 input
    #f = pl.figure("Model Evaluation",figsize=(32,12), dpi = 100)
    #f.add_subplot(1, 2, 1)
    for input in l:
      print("--------------- " + input)
      label_list, predicts, assigned_list = parse_input(input)
      #if FLAGS.recall:
      #  evaluate_recall(label_list, predicts, input)
      #elif FLAGS.precison:
      #  evaluate_precison(label_list, predicts, input)
      #else：
      #  evaluate(label_list, predicts, assigned_list, input)
    #f.add_subplot(1, 2, 0)
    for input in l:
      label_list, predicts, assigned_list = parse_input(input)
      precision_recall(label_list, assigned_list, input)
  else:
    input2 = ""  
    if (pos > 1):
      input2 = argv[2]
      #FLAGS.show = True
    print("--------------- " + input)
    label_list, predicts, assigned_list = parse_input(input)
    #f = pl.figure(figsize=(32,12))
    #f.add_subplot(1, 2, 1)
    if FLAGS.recall:
      evaluate_recall(label_list, predicts, FLAGS.thre)
    elif FLAGS.precision:
      evaluate_precision(label_list, predicts, FLAGS.thre)
    else:
      evaluate(label_list, predicts, assigned_list, FLAGS.thre)
    
    print("--------------- " + input2)
    label_list2 = []
    predicts2 = []
    predict_list2 = []
    if (input2 != ""):
      label_list2, predicts2, predict_list2 = parse_input(input2)
      evaluate(label_list2, predicts2, predict_list2, input2)
  
    #f.add_subplot(1, 2, 0)
    precision_recall(label_list, predicts, input)
  
    if (input2 != ""):  
      precision_recall(label_list2, predicts2, input2)

  #pl.savefig(FLAGS.image)
  #if (FLAGS.show):
  #  pl.show()
  
    
if __name__ == "__main__":  
  main(sys.argv)  
