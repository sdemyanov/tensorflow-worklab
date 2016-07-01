# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:14:41 2016

@author: sdemyanov
"""

import numpy as np
from sklearn import metrics


def get_prob_acc(prob, lab):
  return np.mean(np.argmax(prob, axis=1) == lab)


def get_auc_score(prob, lab):
  fpr, tpr, thresholds = metrics.roc_curve(lab, prob, pos_label=0)
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
  return np.divide(np.array(correct, dtype=np.float), overall)


def get_prob_confmat(prob, lab):
  classnum = prob.shape[1]
  mat = np.zeros((classnum, classnum), dtype=int)
  for pind in range(prob.shape[0]):
    mat[int(lab[pind]), np.argmax(prob[pind, :])] += 1
    #mat = np.transpose(mat)
  return mat


def get_block_confmat(confmat, blocks):
  assert(confmat.shape[0] == confmat.shape[1])
  classnum = confmat.shape[0]
  #assert(np.sum(blocks) == classnum)
  blocknum = len(blocks)

  blockconf = np.zeros((blocknum, blocknum))
  for bi in range(blocknum):
    for bj in range(blocknum):
      blockconf[bi, bj] = 0
      for i in blocks[bi]:
        for j in blocks[bj]:
          blockconf[bi, bj] += confmat[i, j]
  return blockconf

  """
  indi = 0
  for bi in range(blocknum):
    indj = 0
    for bj in range(blocknum):
      bl = confmat[indi:indi+blocks[bi],indj:indj+blocks[bj]]
      blockconf[bi,bj] = np.sum(np.sum(bl))
      indj += blocks[bj]
    indi += blocks[bi]
  assert(np.sum(np.sum(blockconf)) == np.sum(np.sum(confmat)))
  return blockconf
  """

def get_block_prob_labels(prob, labels, blocks):
  # IMPORTANT: blocks must not intersect, otherwise the result is not unique
  blocknum = len(blocks)
  assert prob.shape[0] == labels.shape[0]
  newprob = np.zeros((prob.shape[0], blocknum))
  for i in range(blocknum):
    newprob[:, i] = np.sum(prob[:, blocks[i]], 1)
  #normalize to have sum = 1
  mult_coefs = np.sum(newprob, 1, keepdims=True)
  newprob /= np.tile(mult_coefs, (1, blocknum))

  newlab = np.zeros(prob.shape[0])
  missing = []
  for i in range(prob.shape[0]):
    is_missing = True
    for j in range(len(blocks)):
      if (labels[i] in blocks[j]):
        newlab[i] = j
        is_missing = False
        break
    if (is_missing):
      missing.append(i)

  newprob = np.delete(newprob, missing, axis=0)
  newlab = np.delete(newlab, missing, axis=0)
  return newprob, newlab