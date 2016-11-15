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
import time
import sys
from datetime import datetime

from session import Session
from mnist_reader import MnistReader as Reader
from mnist_classifier import MnistClassifier as Classifier

class Trainer(object):

  def __init__(self, params, writer=None):
    self._batch_size = params['batch_size']
    self._graph = tf.Graph()
    with self._graph.as_default():
      with tf.device('/gpu:' + str(params['gpu'])):
        reader = Reader(params['fold_name'])
        params['classes_num'] = reader.CLASSES_NUM
        self._input = reader.inputs(self._batch_size, params['is_train'])
        self._classifier = Classifier(params, self._input['images'])
        self._cross_entropy, self._total_loss = self._classifier.losses(self._input['labels'])
        self._lr_placeholder = tf.placeholder(tf.float32)
        self._train = self._train_op()
        self._all_summaries = tf.merge_all_summaries()

    self.results_dir = params['results_dir']
    print('Trainer model folder: %s' %self.results_dir)
    if not tf.gfile.Exists(self.results_dir):
      tf.gfile.MakeDirs(self.results_dir)

    self.writer = writer
    if self.writer is not None and params['write_graph']:
      self.writer.write_graph(self._graph)


  def _train_op(self):
    tf.scalar_summary('learning_rate', self._lr_placeholder)
    with tf.variable_scope('train_operation'):
      opt = tf.train.GradientDescentOptimizer(self._lr_placeholder)
      #opt = tf.train.MomentumOptimizer(self._lr_placeholder, Trainer.MOMENTUM)
      grads_and_vars = opt.compute_gradients(self._total_loss)
      grads_and_vars_mult = []
      for grad, var in grads_and_vars:
        grad *= self._classifier.lr_multipliers[var.op.name]
        grads_and_vars_mult.append((grad, var))
        #tf.histogram_summary('variables/' + var.op.name, var)
        #tf.histogram_summary('gradients/' + var.op.name, grad)
      return opt.apply_gradients(grads_and_vars_mult)



  def train(self, params):
    print('\n%s: training...' % datetime.now())
    sys.stdout.flush()

    print_loss = 0
    train_loss = None
    save_loss = 0
    save_step = 0
    total_loss = 0
    feed_dict={self._lr_placeholder: params['learning_rate']}

    session = Session(self._graph, self.results_dir, params['model_name'])
    if 'init_step' not in params or params['init_step'] is None:
      init_step = session.init_step
    else:
      init_step = params['init_step']
    session.init(self._classifier, init_step, params['restoring_file'])
    session.start()

    last_step = init_step + params['step_num']
    print('%s: training till: %d steps' % (datetime.now(), last_step))
    for step in range(init_step+1, last_step+1):

      start_time = time.time()
      _, cross_entropy_batch, total_loss_batch = session.run(
        [self._train, self._cross_entropy, self._total_loss], feed_dict=feed_dict
      )
      duration = time.time() - start_time
      assert not np.isnan(total_loss_batch), 'Model diverged with loss = NaN'
      cross_entropy_loss_value = np.mean(cross_entropy_batch)
      print_loss += cross_entropy_loss_value
      save_loss += cross_entropy_loss_value
      total_loss += total_loss_batch
      save_step += 1

      if (step - init_step) % params['print_frequency'] == 0:
        examples_per_sec = self._batch_size / duration
        format_str = ('%s: step %d, loss = %.2f, lr = %f, '
                      '(%.1f examples/sec; %.3f sec/batch)')
        print_loss /= params['print_frequency']
        print(format_str % (datetime.now(), step, print_loss, params['learning_rate'],
                            examples_per_sec, float(duration)))
        print_loss = 0

      # Save the model checkpoint and summaries periodically.
      if (step == last_step or
        (params['save_frequency'] is not None and (step - init_step) % params['save_frequency'] == 0)):
        session.save(step)
        total_loss /= save_step
        train_loss = save_loss / save_step
        print('%s: train_loss = %.3f' % (datetime.now(), train_loss))
        if self.writer is not None:
          summary_str = session.run(self._all_summaries, feed_dict=feed_dict)
          self.writer.write_summaries(summary_str, step)
          self.writer.write_scalars({'losses/training/cross_entropy_loss': train_loss,
                                     'losses/training/total_loss': total_loss}, step)
        total_loss = 0
        save_loss = 0
        save_step = 0

      if session.should_stop():
        break

    session.stop()
    return step, train_loss