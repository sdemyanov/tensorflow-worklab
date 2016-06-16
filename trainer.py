# -*- coding: utf-8 -*-
"""
Created on Thu May 19 14:48:48 2016

@author: sdemyanov
"""

import tensorflow as tf
import numpy as np
import time
import sys
from datetime import datetime

import reader
reload(reader)
from reader import Reader

import resnet
reload(resnet)
from resnet import Network

import session
reload(session)
from session import Session

class Trainer(object):

  BATCH_SIZE = 32
  PRINT_FREQUENCY = 10
  SAVE_FREQUENCY = 500

  def __init__(self, models_dir, fold_name, writer=None):
    self._graph = tf.Graph()
    with self._graph.as_default():
      reader = Reader(fold_name)
      with tf.device('/gpu:0'):
        images, labels, _ = reader.inputs(Trainer.BATCH_SIZE, is_train=True)
        self._network = Network(images, is_train=True)
        self._loss = self._network.loss(labels)
        self._lr_placeholder = tf.placeholder(tf.float32)
        self._train = self._train_op()
        self._all_summaries = tf.merge_all_summaries()

    self.models_dir = models_dir
    print('Trainer model folder: %s' %self.models_dir)
    if not tf.gfile.Exists(self.models_dir):
      tf.gfile.MakeDirs(self.models_dir)

    self.writer = writer
    if (self.writer):
      self.writer.write_graph(self._graph)


  def _train_op(self):
    tf.scalar_summary('learning_rate', self._lr_placeholder)
    opt = tf.train.GradientDescentOptimizer(self._lr_placeholder)
    grads_and_vars = opt.compute_gradients(self._loss)
    grads_and_vars_mult = []
    for grad, var in grads_and_vars:
      grad *= self._network.lr_multipliers[var.op.name]
      grads_and_vars_mult.append((grad, var))
      tf.histogram_summary('variables/' + var.op.name, var)
      tf.histogram_summary('gradients/' + var.op.name, grad)
    return opt.apply_gradients(grads_and_vars_mult)


  def train(self, learning_rate, step_num, init_step=None):
    print('%s: training...' % datetime.now())
    sys.stdout.flush()

    session = Session(self._graph, self.models_dir)
    init_step = session.init(self._network, init_step)
    session.start()

    last_step = init_step+step_num
    print('%s: training till: %d steps' %(datetime.now(), last_step))

    print_loss = 0
    save_loss = 0
    feed_dict={self._lr_placeholder: learning_rate}
    for step in xrange(init_step+1, last_step+1):
      start_time = time.time()
      _, loss_batch = session.run([self._train, self._loss],
                                  feed_dict=feed_dict)
      duration = time.time() - start_time
      assert not np.isnan(loss_batch), 'Model diverged with loss = NaN'
      print_loss += loss_batch
      save_loss += loss_batch

      if step % Trainer.PRINT_FREQUENCY == 0:
        examples_per_sec = Trainer.BATCH_SIZE / duration
        format_str = ('%s: step %d, loss = %.2f, lr = %f, '
                      '(%.1f examples/sec; %.3f sec/batch)')
        print_loss /= Trainer.PRINT_FREQUENCY
        print(format_str % (datetime.now(), step, print_loss, learning_rate,
                            examples_per_sec, float(duration)))
        print_loss = 0

      # Save the model checkpoint and summaries periodically.
      if (step % Trainer.SAVE_FREQUENCY == 0 or step == last_step):
        session.save(step)
        if (self.writer):
          summary_str = session.run(self._all_summaries, feed_dict=feed_dict)
          self.writer.write_summaries(summary_str, step)
          save_loss /= Trainer.SAVE_FREQUENCY
          self.writer.write_scalars({'losses/training/total_loss': save_loss}, step)
          train_loss = save_loss
          save_loss = 0

    session.stop()
    return step, train_loss