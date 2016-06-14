# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:14:41 2016

@author: sdemyanov
"""

import numpy as np
from sklearn import metrics


def get_pred_acc(pred, lab):
  return np.mean(np.argmax(pred, axis=1) == lab)


def get_auc_score(pred, lab):
  fpr, tpr, thresholds = metrics.roc_curve(lab, pred, pos_label=0)
  return metrics.auc(fpr, tpr)


def get_f1_score(confmat):
  assert(confmat.shape[0] == 2 and confmat.shape[1] == 2)
  precision = float(confmat[0, 0]) / np.sum(confmat[:, 0])
  recall = float(confmat[0, 0]) / np.sum(confmat[0, :])
  print('precision: %f' %precision)
  print('recall: %f' %recall)
  return 2 * precision * recall / (precision + recall)


def get_accuracy(confmat):
  correct = np.sum(np.diagonal(confmat))
  overall = np.sum(confmat)
  return correct.astype(float) / overall


def get_sensitivities(confmat):
  correct = np.diagonal(confmat)
  overall = np.sum(confmat, 1)
  return np.divide(np.array(correct, dtype=np.float), overall);


def get_pred_confmat(pred, lab):
  classnum = pred.shape[1]
  mat = np.zeros((classnum, classnum), dtype=int)
  for pind in xrange(pred.shape[0]):
    mat[int(lab[pind]), np.argmax(pred[pind, :])] += 1
    #mat = np.transpose(mat)
  return mat


def get_block_confmat(confmat, blocks):
  assert(confmat.shape[0] == confmat.shape[1])
  classnum = confmat.shape[0]
  assert(np.sum(blocks) == classnum)
  blocknum = len(blocks)

  blockconf = np.zeros((blocknum, blocknum))
  indi = 0
  for bi in xrange(blocknum):
    indj = 0
    for bj in xrange(blocknum):
      bl = confmat[indi:indi+blocks[bi],indj:indj+blocks[bj]]
      blockconf[bi,bj] = np.sum(np.sum(bl))
      indj += blocks[bj]
    indi += blocks[bi]
  assert(np.sum(np.sum(blockconf)) == np.sum(np.sum(confmat)))
  return blockconf


def get_block_pred(pred, blocks):
  ind = 0
  blocknum = len(blocks)
  newpred = np.zeros((pred.shape[0], blocknum))
  for i in xrange(blocknum):
    newpred[:,i] = np.sum(pred[:,ind:ind+blocks[i]], 1)
    ind += blocks[i]
  return newpred


def get_block_labels(labels, blocks):
  newlab = np.zeros(labels.shape)
  for i in xrange(labels.shape[0]):
    for j in xrange(len(blocks)):
      if (labels[i] < np.sum(blocks[:j+1])):
        newlab[i] = j
        break
  return newlab