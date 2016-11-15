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

import paths
reload(paths)

import stats
reload(stats)

import utils
reload(utils)

from session import Session
from mnist_reader import MnistReader as Reader
from mnist_classifier import MnistClassifier as Classifier

class Tester(object):

  RESULTS_FILE = 'results'

  def __init__(self, params, writer=None):
    self.fold_name = params['fold_name']
    self._batch_size = params['batch_size']
    self._graph = tf.Graph()
    with self._graph.as_default():
      with tf.device('/gpu:' + str(params['gpu'])):
        reader = Reader(self.fold_name)
        self.fold_size = reader.fold_size
        params['classes_num'] = reader.CLASSES_NUM
        self._input = reader.inputs(self._batch_size, params['is_train'])
        self._classifier = Classifier(params, self._input['images'])
        self._probs = self._classifier.probs()
        self._cross_entropy_losses = self._classifier.cross_entropy_losses(self._input['labels'])
        self._all_summaries = tf.merge_all_summaries()

    self.results_dir = params['results_dir']
    print 'Tester model folder: %s' % self.results_dir
    assert os.path.exists(self.results_dir)
    self.writer = writer


  def test(self, params):
    print('\n%s: testing...' %datetime.now())
    sys.stdout.flush()

    session = Session(self._graph, self.results_dir, params['model_name'])
    if 'init_step' not in params or params['init_step'] is None:
      init_step = session.init_step
    else:
      init_step = params['init_step']

    if 'step_num' not in params or params['step_num'] is None:
      step_num = int(np.ceil(np.float(self.fold_size) / self._batch_size))
    else:
      step_num = params['step_num']

    results_file_name = Tester.RESULTS_FILE + '-' + str(init_step) + '-' + \
                        self.fold_name + '-' + str(step_num) + '.json'
    results_file = os.path.join(self.results_dir, results_file_name)
    if not params['load_results'] or not os.path.isfile(results_file):
      session.init(self._classifier, init_step, params['restoring_file'])
      session.start()
      if init_step == 0:
        print 'WARNING: testing an untrained model'
      total_step_num = step_num * params['epoch_num']
      test_num = total_step_num * self._batch_size
      print('%s: test_num=%d' % (datetime.now(), step_num * self._batch_size))
      print('%s: epoch_num=%d' % (datetime.now(), params['epoch_num']))

      results = {}
      results['losses'] = np.zeros(test_num, dtype=np.float32)
      results['probs'] = np.zeros((test_num, Reader.CLASSES_NUM), dtype=np.float32)
      results['labels'] = np.zeros(test_num, dtype=np.int64)

      start_time = time.time()
      for step in range(total_step_num):
        #print('%s: eval_iter=%d' %(datetime.now(), i))
        loss_batch, prob_batch, label_batch = session.run(
          [self._cross_entropy_losses, self._probs, self._input['labels']]
        )
        begin = step * self._batch_size
        results['losses'][begin:begin+self._batch_size] = loss_batch
        results['probs'][begin:begin+self._batch_size, :] = prob_batch
        results['labels'][begin:begin + self._batch_size] = label_batch
        if (step+1) % step_num == 0:
          print "Epoch num: %d" % ((step+1)/step_num)
        if session.should_stop():
          break

      duration = time.time() - start_time
      print('%s: duration = %.1f sec' %(datetime.now(), float(duration)))
      sys.stdout.flush()
      if self.writer is not None:
        summary_str = session.run(self._all_summaries)
        self.writer.write_summaries(summary_str, init_step)

      session.stop()
    else:
      print 'WARNING: using precomputed results'
      results = utils.load_from_file(results_file)

    results['loss'] = np.mean(results['losses']).item()
    results = self.get_all_stats(results)
    if self.writer is not None and not params['load_results']:
      self.writer.write_scalars({'losses/testing/cross_entropy_loss': results['loss'],
                                 'accuracy': results['accuracy']}, init_step)
    utils.dump_to_file(results, results_file)

    return init_step, results['loss']


  def get_all_stats(self, results):
    confmat = stats.get_prob_confmat(results['probs'], results['labels'])
    print confmat
    print 'Total test num:', np.sum(confmat)
    results['accuracy'] = stats.get_accuracy(confmat) * 100
    print 'Accuracy: %.1f%%' % results['accuracy']
    return results
