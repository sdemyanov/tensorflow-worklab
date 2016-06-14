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

import reader
reload(reader)
from reader import Reader

import network
reload(network)
from network import Network

import session
reload(session)
from session import Session

#import writer
#reload(writer)

class Trainer(object):

  MODELS_DIR = 'results'

  BATCH_SIZE = 32
  DECAY_FACTOR = 0.1

  #CHANGE
  LEARNING_RATE = 0.01
  STEP_VALUES = [30000, 40000, 50000, 60000]
  LAST_STEP = STEP_VALUES[-1]

  PRINT_FREQUENCY = 10
  SAVE_FREQUENCY = 500

  def __init__(self, main_dir, fold_name, writer=None):
    self._graph = tf.Graph()
    with self._graph.as_default():
      reader = Reader(main_dir, fold_name)
      with tf.device('/gpu:0'):
        images, labels, _ = reader.inputs(Trainer.BATCH_SIZE, is_train=True)
        self._network = Network(images, is_train=True)
        self._loss = self._network.loss(labels)
        self._lr_placeholder = tf.placeholder(tf.float32)
        self._train = self._train_op()
        self._all_summaries = tf.merge_all_summaries()

    self.models_dir = os.path.join(main_dir, Trainer.MODELS_DIR)
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
    return opt.apply_gradients(grads_and_vars)


  def _learning_rate(self, step):
    lr = Trainer.LEARNING_RATE
    for i in xrange(len(Trainer.STEP_VALUES)):
      if (step > Trainer.STEP_VALUES[i]):
        lr *= Trainer.DECAY_FACTOR
      else:
        break
    return lr


  def train(self, step_num=None):
    print('%s: training...' % datetime.now())
    sys.stdout.flush()

    session = Session(self._graph, self.models_dir)
    first_step = session.init(self._network)
    session.start()

    last_step = Trainer.LAST_STEP
    if (step_num and first_step+step_num < last_step):
      last_step = first_step+step_num

    print_loss = 0
    save_loss = 0
    for step in xrange(first_step+1, last_step+1):
      lr = self._learning_rate(step)
      feed_dict={self._lr_placeholder: lr}
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
        print(format_str % (datetime.now(), step, print_loss,
                            lr, examples_per_sec, float(duration)))
        print_loss = 0

      # Save the model checkpoint and summaries periodically.
      if (step % Trainer.SAVE_FREQUENCY == 0 or step == Trainer.LAST_STEP):
        session.save(step)
        if (self.writer):
          summary_str = session.run(self._all_summaries, feed_dict=feed_dict)
          self.writer.write_summaries(summary_str, step)
          save_loss /= Trainer.SAVE_FREQUENCY
          self.writer.write_scalars({'losses/training/total_loss': save_loss}, step)
          save_loss = 0

    session.stop()
    return step