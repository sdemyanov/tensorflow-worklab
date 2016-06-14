# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:57:34 2016

@author: sdemyanov
"""

import tensorflow as tf
import os

class Session(object):

  GPU_SIZE = 12 #GB
  MODEL_FILE = 'model.ckpt'
  #RESTORING_FILE = None
  RESTORING_FILE = '/path/to/resnet-pretrained/ResNet-L50.ckpt'

  def __init__(self, graph, path, memory=None):

    config = tf.ConfigProto()
    config.allow_soft_placement=True
    config.log_device_placement=False
    config.gpu_options.allow_growth=True
    if (memory is not None):
      fraction = float(memory) / Session.GPU_SIZE
      config.gpu_options.per_process_gpu_memory_fraction = fraction
    self._sess = tf.Session(graph=graph, config=config)
    self._coord = tf.train.Coordinator()
    self._threads = []
    self._path = path
    with graph.as_default():
      self._saver = tf.train.Saver(tf.all_variables())


  def _get_checkpoint(self):
    ckpt = tf.train.get_checkpoint_state(self._path)
    if ckpt and ckpt.model_checkpoint_path:
      filename = ckpt.model_checkpoint_path
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    else:
      filename = None
      step = 0
    return filename, step


  def _init_vars(self, initnames=None):
    with self._sess.graph.as_default():
      if (initnames is None):
        self.run(tf.initialize_all_variables())
      else:
        self.run(tf.initialize_variables(initnames))


  def _restore_vars(self, filename, restnames=None):
    print('Restoring from %s' %filename)
    with self._sess.graph.as_default():
      if (restnames is None):
        restorer = tf.train.Saver(tf.all_variables())
      else:
        restorer = tf.train.Saver(restnames)
    restorer.restore(self._sess, filename)


  def save(self, step):
    checkpoint_path = os.path.join(self._path, Session.MODEL_FILE)
    self._saver.save(self._sess, checkpoint_path, global_step=step)
    print('Saving model to %s, step=%d' %(checkpoint_path, step))


  def init(self, network, step=None):
    if (step is not None):
      filename = os.path.join(self._path, Session.MODEL_FILE+'-'+str(step))
      assert os.path.isfile(filename)
    else:
      filename, step = self._get_checkpoint()

    if (filename is not None):
      print('Restoring from saved model')
      self._restore_vars(filename)
    else:
      if (Session.RESTORING_FILE is not None):
        print('Restoring from external model')
        self._restore_vars(Session.RESTORING_FILE, network.restnames)
        self._init_vars(network.initnames)
      else:
        print('Initializing by random variables...')
        self._init_vars()
    return step


  def start(self):
    with self._sess.graph.as_default():
      self.run(tf.assert_variables_initialized())
      # create and launch threads for all queue_runners
      # it is like start_queue_runners, but manually
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        self._threads.extend(qr.create_threads(
          self._sess, coord=self._coord, daemon=True, start=True
        ))
      # Use this if you get an error message about empty queues!!
      # That error might be caused by other non-visible errors
      #tf.train.start_queue_runners(sess=self._sess)


  def run(self, operations, feed_dict=None):
    return self._sess.run(operations, feed_dict=feed_dict)


  def stop(self):
    self._coord.request_stop()
    self._coord.join(self._threads) #, stop_grace_period_secs=10)
    self._sess.close()
