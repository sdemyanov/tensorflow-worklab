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
import os

class Session(object):

  GPU_SIZE = 12 #GB
  RESTORE_ANYWAY = False
  #RESTORE_ANYWAY = True

  def __init__(self, graph, path, model_name, memory=None):

    config = tf.ConfigProto()
    config.allow_soft_placement=True
    config.log_device_placement=False
    config.gpu_options.allow_growth=True
    if memory is not None:
      fraction = float(memory) / Session.GPU_SIZE
      config.gpu_options.per_process_gpu_memory_fraction = fraction
    self._sess = tf.Session(graph=graph, config=config)
    self._coord = tf.train.Coordinator()
    self._threads = []
    self._path = path
    self._model_name = model_name
    self.model_file, self.init_step = self._get_checkpoint()
    with graph.as_default():
      self._saver = tf.train.Saver(tf.all_variables())


  def _get_model_file(self, step):
    return os.path.join(self._path, self._model_name+'-'+str(step))


  def _get_checkpoint(self):
    ckpt = tf.train.get_checkpoint_state(self._path)
    if ckpt and ckpt.model_checkpoint_path:
      model_file = ckpt.model_checkpoint_path
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
      return model_file, step
    return None, 0


  def _init_vars(self, init_names=None):
    with self._sess.graph.as_default():
      if (init_names is None):
        self.run(tf.initialize_all_variables())
      else:
        self.run(tf.initialize_variables(init_names))


  def _restore_vars(self, model_file, rest_names=None):
    with self._sess.graph.as_default():
      if rest_names is None:
        restorer = tf.train.Saver(tf.all_variables())
      else:
        restorer = tf.train.Saver(rest_names)
    restorer.restore(self._sess, model_file)


  def save(self, step):
    model_file = self._get_model_file(step)
    self._saver.save(self._sess, model_file, write_meta_graph=False)
    print 'Saving model to %s, step=%d' % (model_file, step)


  def init(self, network, init_step=None, restoring_file=None):
    # use step=None to init from the last model, if there is any
    # use step=0 to init from a restoring model, or to init from scratch
    # use step>0 to init from a particular saved model
    if init_step is not None:
      if init_step == 0:
        model_file = None
      else:
        model_file = self._get_model_file(init_step)
        if not os.path.isfile(model_file):
          print model_file
          assert False, 'Model file for the specified step does not exist'
    else:
      model_file, init_step = self.model_file, self.init_step

    if model_file is not None:
      print 'Restoring from saved model %s' % model_file
      self._restore_vars(model_file)
      print 'Restoring finished'
    else:
      print 'Initializing by random variables...'
      self._init_vars()
      print 'Initialization finished'

    if restoring_file is not None and len(network.rest_names) > 0:
      if model_file is None or Session.RESTORE_ANYWAY:
        print 'WARNING: Restoring from external model %s' % restoring_file
        #print(network.rest_names)
        self._restore_vars(restoring_file, network.rest_names)
        print 'Restoring finished'


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


  def should_stop(self):
    return self._coord.should_stop()
