# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:14:41 2016

@author: sdemyanov
"""

import numpy as np
from sklearn import metrics

def get_prob_acc(probs, labels):
  return np.mean(np.argmax(probs, axis=1) == labels)


def get_auc_score(scores, labels):
  fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
  return metrics.auc(fpr, tpr)


def get_f1_score(confmat):
  assert confmat.shape[0] == 2 and confmat.shape[1] == 2
  precision = float(confmat[0, 0]) / np.sum(confmat[:, 0])
  recall = float(confmat[0, 0]) / np.sum(confmat[0, :])
  print 'precision: %f' % precision
  print 'recall: %f' % recall
  return 2 * precision * recall / (precision + recall)


def get_accuracy(confmat):
  correct = np.sum(np.diagonal(confmat))
  overall = np.sum(confmat)
  return correct.astype(float) / overall


def get_sensitivities(confmat):
  correct = np.diagonal(confmat)
  overall = np.sum(confmat, 1)
  return np.divide(np.array(correct, dtype=np.float), overall)


def get_pred_confmat(classes, preds, labels):
  classnum = len(classes)
  mat = np.zeros((classnum, classnum), dtype=int)
  for pind in range(preds.shape[0]):
    labind = np.where(classes == labels[pind])
    predind = np.where(classes == preds[pind])
    mat[labind[0], predind[0]] += 1
    # mat = np.transpose(mat)
  return mat


def get_prob_confmat(probs, labels):
  classnum = probs.shape[1]
  mat = np.zeros((classnum, classnum), dtype=int)
  for pind in range(probs.shape[0]):
    mat[int(labels[pind]), np.argmax(probs[pind, :])] += 1
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
  assert np.sum(blockconf) == np.sum(confmat), 'Blocks should represent a splitting of confmat'
  return blockconf


def get_block_probs_labels(prob, labels, blocks):
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


def get_spec_for_sens(scores, labels, sens):
  fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
  curind = np.size(tpr) - 1
  while (tpr[curind-1] >= sens):
    curind -= 1
  return tpr[curind], 1 - fpr[curind], thresholds[curind]


def get_sens_for_spec(scores, labels, spec):
  fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=0)
  curind = 0
  while (1 - fpr[curind+1] >= spec):
    curind += 1
  return tpr[curind], 1 - fpr[curind], thresholds[curind]


def get_average_precisions(probs, labels):
  print 'probshape:', np.shape(probs)
  classnum = np.size(probs, 1)
  labels_arr = np.zeros_like(probs)
  for i in xrange(classnum):
    labels_arr[labels == i, i] = 1
  print 'macro:', metrics.average_precision_score(labels_arr, probs, average='macro')
  print 'weighted:', metrics.average_precision_score(labels_arr, probs, average='weighted')
  skap = metrics.average_precision_score(labels_arr, probs, average=None)
  return {i: round(skap[i] * 1000) / 10 for i in xrange(classnum)}