# -*- coding: utf-8 -*-
"""
Created on Thu May 19 14:48:48 2016

@author: sdemyanov
"""

import tensorflow as tf
import numpy as np
import os
import time
import sys
from datetime import datetime

sys.path.append('./')

import utils.stats
reload(utils.stats)

import reader
reload(reader)
from reader import Reader

import network
reload(network)
from network import Network

import session
reload(session)
from session import Session

class Tester(object):

  MODELS_DIR = 'results'

  BATCH_SIZE = 32

  def __init__(self, main_dir, fold_name, writer=None):
    self._graph = tf.Graph()
    with self._graph.as_default():
      reader = Reader(main_dir, fold_name)
      self.fold_size = reader.fold_size
      with tf.device('/gpu:0'):
        images, self._labels, self._filenames = reader.inputs(Tester.BATCH_SIZE,
                                                              is_train=False)
        self._network = Network(images, is_train=False)
        self._probs = self._network.probs()
        self._loss = self._network.loss(self._labels)
        self._all_summaries = tf.merge_all_summaries()

    self.models_dir = os.path.join(main_dir, Tester.MODELS_DIR)
    print('Tester model folder: %s' %self.models_dir)
    assert(tf.gfile.Exists(self.models_dir))

    self.writer = writer


  def test(self, step_num=None, train_step=None):
    print('%s: testing...' % datetime.now())
    sys.stdout.flush()

    session = Session(self._graph, self.models_dir)
    if (train_step is not None):
      assert train_step > 0
    train_step = session.init(self._network, step=train_step)
    print('%s: train_step=%d' % (datetime.now(), train_step))
    session.start()

    if (step_num is None):
      step_num = np.int(np.ceil(np.float(self.fold_size) / Tester.BATCH_SIZE))
    test_num = step_num * Tester.BATCH_SIZE
    print('%s: test_num=%d' % (datetime.now(), test_num))

    loss_value = 0
    prob_values = np.zeros((test_num, Reader.CLASSES_NUM), dtype=np.float32)
    label_values = np.zeros(test_num, dtype=np.int64)
    filename_values = []
    begin = 0
    start_time = time.time()

    for step in xrange(step_num):
      loss_batch, prob_batch, label_batch, filename_batch = session.run(
        [self._loss, self._probs, self._labels, self._filenames]
      )
      loss_value += loss_batch
      begin = step * Tester.BATCH_SIZE
      prob_values[begin:begin+Tester.BATCH_SIZE, :] = prob_batch
      label_values[begin:begin+Tester.BATCH_SIZE] = label_batch
      filename_values.extend(filename_batch)

    duration = time.time() - start_time
    print('%s: duration = %.1f sec' %(datetime.now(), float(duration)))
    sys.stdout.flush()

    loss_value /= step_num
    print('%s: test_loss = %.2f' %(datetime.now(), loss_value))

    mult_acc = self.get_pred_stat(
      prob_values, label_values, filename_values
    )
    if (self.writer):
      summary_str = session.run(self._all_summaries)
      self.writer.write_summaries(summary_str, train_step)
      self.writer.write_scalars({'losses/testing/total_loss': loss_value,
                                 'accuracy': mult_acc}, train_step)
    session.stop()


  def get_pred_stat(self, prob, labels, filenames):

    confmat = utils.stats.get_pred_confmat(prob, labels)
    print('Total number of examples: %d' %np.sum(confmat))
    print('Confusion matrix:')
    print(confmat)

    mult_sens = utils.stats.get_sensitivities(confmat)
    np.set_printoptions(precision=1)
    print('Sensitivities:')
    print(mult_sens*100)

    mult_accuracy = utils.stats.get_accuracy(confmat)
    print('Multiclass accuracy: %.1f%%' %(mult_accuracy*100))

    return mult_accuracy
