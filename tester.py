#  Copyright 2016-present Sergey Demyanov. All Rights Reserved.
#
#  Contact: my_name@my_sirname.net
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# =============================================================================

import tensorflow as tf
import numpy as np
import os
import time
import sys
from datetime import datetime

sys.path.append('../')

import utils.stats
reload(utils.stats)

import utils.molemap
reload(utils.molemap)

import reader
reload(reader)
from reader import Reader

import resnet
reload(resnet)
from resnet import Network

import session
reload(session)
from session import Session

class Tester(object):

  BATCH_SIZE = 32

  def __init__(self, models_dir, fold_name, writer=None, hyper=None):
    self._graph = tf.Graph()
    with self._graph.as_default():
      reader = Reader(fold_name)
      self.fold_size = reader.fold_size
      with tf.device('/gpu:1'):
        images, self._labels, scores, self._filenames = reader.inputs(Tester.BATCH_SIZE, is_train=False)
        self._network = Network(images, is_train=False, hyper=hyper, features=scores)
        self._probs = self._network.probs()
        self._loss = self._network.loss(self._labels)
        self._all_summaries = tf.merge_all_summaries()

    self.models_dir = models_dir
    print('Tester model folder: %s' %self.models_dir)
    assert os.path.exists(self.models_dir)

    self.writer = writer


  def test(self, step_num=None, init_step=None, restoring_file=None):
    print('%s: testing...' %datetime.now())
    sys.stdout.flush()

    session = Session(self._graph, self.models_dir)
    init_step = session.init(self._network, init_step, restoring_file)
    session.start()

    if (init_step == 0):
      print('WARNING: testing an untrained model')
    if (step_num is None):
      step_num = np.int(np.ceil(np.float(self.fold_size) / Tester.BATCH_SIZE))
    test_num = step_num * Tester.BATCH_SIZE
    print('%s: test_num=%d' %(datetime.now(), test_num))

    loss_value = 0
    prob_values = np.zeros((test_num, Reader.CLASSES_NUM), dtype=np.float32)
    label_values = np.zeros(test_num, dtype=np.int64)
    filename_values = []
    begin = 0
    start_time = time.time()

    for step in range(step_num):
      #print('%s: eval_iter=%d' %(datetime.now(), i))
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
    #return loss_value, probs_values, labels_values
    print('%s: test_loss = %.3f' %(datetime.now(), loss_value))

    mult_acc, bin_acc, auc, bin_sens = self.get_pred_stat(
      prob_values, label_values, filename_values
    )
    if (self.writer):
      summary_str = session.run(self._all_summaries)
      self.writer.write_summaries(summary_str, init_step)
      self.writer.write_scalars({'losses/testing/total_loss': loss_value,
                                 'accuracy/multiclass': mult_acc,
                                 'accuracy/binary': bin_acc,
                                 'stats/AUC': auc,
                                 'stats/sensitivity': bin_sens[0],
                                 'stats/specificity': bin_sens[1]}, init_step)
    session.stop()
    return init_step, loss_value


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

    blocks = [3, 12]
    binconf = utils.stats.get_block_confmat(confmat, blocks)
    bin_sens = utils.stats.get_sensitivities(binconf)
    print('Binary sensitivities:')
    print(bin_sens*100)
    print('F1-score: %f' %utils.stats.get_f1_score(binconf))
    bin_max_accuracy = utils.stats.get_accuracy(binconf)
    print('Binary max-accuracy: %.1f%%' %(bin_max_accuracy*100))

    binpred = utils.stats.get_block_pred(prob, blocks)
    binlab = utils.stats.get_block_labels(labels, blocks)
    bin_sum_accuracy = utils.stats.get_pred_acc(binpred, binlab)
    print('Binary sum-accuracy: %.1f%%' %(bin_sum_accuracy*100))
    auc = utils.stats.get_auc_score(binpred[:,0], binlab)
    print('AUC-score: %f' %auc)
    #print ('')
    return mult_accuracy, bin_sum_accuracy, auc, bin_sens
