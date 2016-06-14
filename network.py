# -*- coding: utf-8 -*-
"""
Created on Thu May 19 11:55:46 2016

@author: sdemyanov
"""

import tensorflow as tf

import reader
reload(reader)
from reader import Reader

class Network(object):

  TRAIN_DECAY = 0.99    # The decay to use for the moving average.
  STDDEV = 1e-4
  BN_EPS = 1e-5

  TOWER_NAME = 'tower'
  OUTPUT_NAME = 'output'
  LOSSES_NAME = 'losses'
  TRAINING_PREFIX = 'training'
  TESTING_PREFIX = 'testing'

  def __init__(self, input, is_train):
    self.is_train = is_train
    self._decay = Network.TRAIN_DECAY
    if (is_train):
      self._prefix = Network.TRAINING_PREFIX
    else:
      self._prefix = Network.TESTING_PREFIX
    self.lr_multipliers = {}
    self.restnames = {}
    self.initnames = []
    self._output = self._construct(input)


  def dims(self, output, dim=None):
    dims = output.get_shape().as_list()
    if (dim is not None):
      return dims[dim]
    return dims


  def _output_elem(self, output):
    dims = self.dims(output)
    return dims[1] * dims[2] * dims[3]


  def _append(self, scope, appendix):
    if (scope is not None):
      return scope + '/' + appendix
    return None


  def _set_weight_decay(self, var, weight_decay):
    if (weight_decay is not None) and (weight_decay > 0):
      weight_decay = tf.mul(tf.nn.l2_loss(var), weight_decay) #, name=var.name)
      tf.add_to_collection(Network.LOSSES_NAME, weight_decay)


  def _set_lr_mult(self, var, lr_mult):
    self.lr_multipliers[var.op.name] = lr_mult


  def _set_restoring(self, var, restore, restname=None):
    if (restore):
      if (restname is None):
        restname = var.op.name
      self.restnames[restname] = var
      #print("%s: %s" %(var.op.name, restname))
    else:
      self.initnames.append(var)


  def _variable(self, name, shape, initializer,
                weight_decay, lr_mult, restore, restname=None):
    is_trainable = (lr_mult > 0)
    var = tf.get_variable(name=name, shape=shape,
                          initializer=initializer, trainable=is_trainable)
    if (is_trainable):
      self._set_weight_decay(var, weight_decay)
      self._set_lr_mult(var, lr_mult)

    self._set_restoring(var, restore, restname)
    return var


  def _normal_variable(self, name, shape, stddev,
                       weight_decay, lr_mult, restore, restname=None):
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    return self._variable(name, shape, initializer,
                          weight_decay, lr_mult, restore, restname)


  def _constant_variable(self, name, shape, value,
                         weight_decay, lr_mult, restore, restname=None):
    initializer = tf.constant_initializer(value)
    return self._variable(name, shape, initializer,
                          weight_decay, lr_mult, restore, restname)


  ### BASIC LAYERS ###

  def _activation_summary(self, output, scope='summary'):
    # no summary for test network
    if (not self.is_train):
      return
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    with tf.variable_scope(scope):
      #tensor_name = re.sub('%s_[0-9]*/' % Network.TOWER_NAME, '', scope)
      tensor_name = output.op.name
      tf.histogram_summary('activations/' + tensor_name, output)
      zero_fraction = tf.nn.zero_fraction(output)
      tf.scalar_summary('sparsity/' + tensor_name, zero_fraction)


  def _nonlinearity(self, output):
    output = tf.nn.relu(output, name='relu')
    return output


  def _batch_norm(self, output,
                  lr_mult=1.0, restore=True,
                  scope='bn', restscope=None):
    with tf.variable_scope(scope):
      dims = self.dims(output)
      # we don't squeeze only the last dimension, i.e. feature maps
      squeeze_dims = range(len(dims)-1)
      input_maps = dims[-1]
      batch_mean, batch_var = tf.nn.moments(output, squeeze_dims, name='moments')
      ema = tf.train.ExponentialMovingAverage(decay=self._decay)
      ema_apply_op = ema.apply([batch_mean, batch_var])
      # Needed for partial restoration from an existing model
      self._set_restoring(ema.average(batch_mean), restore,
                          self._append(restscope, 'moving_mean'))
      self._set_restoring(ema.average(batch_var), restore,
                          self._append(restscope, 'moving_variance'))
      if (self.is_train):
        with tf.control_dependencies([ema_apply_op]):
          mean, var = tf.identity(batch_mean), tf.identity(batch_var)
      else:
        #mean, var = batch_mean, batch_var
        mean, var = ema.average(batch_mean), ema.average(batch_var)

      beta = self._constant_variable('beta', [input_maps], 0.0, 0.0,
                                     lr_mult, restore,
                                     self._append(restscope, 'beta'))
      gamma = self._constant_variable('gamma', [input_maps], 1.0, 0.0,
                                      lr_mult, restore,
                                      self._append(restscope, 'gamma'))
      output = tf.nn.batch_normalization(output, mean, var, beta, gamma, Network.BN_EPS)
      return output


  def _pool_layer(self, output,
                  filter_size=3, stride=2, func='max', scope='pool'):
    with tf.variable_scope(scope):
      ksize = [1, filter_size, filter_size, 1]
      strides = [1, stride, stride, 1]
      if (func=='max'):
        output = tf.nn.max_pool(output, ksize=ksize,
                                strides=strides,  padding='SAME')
      elif (func=='avg'):
        output = tf.nn.avg_pool(output, ksize=ksize,
                                strides=strides,  padding='SAME')
      else:
        assert False, 'unsupported pooling function'
    return output


  def _conv_layer(self, output, output_maps, filter_size, stride,
                  weight_decay=0.0, lr_mult=1.0, restore=True,
                  scope='conv', restscope=None):
    input_maps = self.dims(output, 3)
    filter_shape = [filter_size, filter_size, input_maps, output_maps]
    with tf.variable_scope(scope):
      kernel = self._normal_variable('weights', filter_shape, Network.STDDEV,
                                     weight_decay, lr_mult, restore,
                                     self._append(restscope, 'weights'))
      output = tf.nn.conv2d(output, kernel, [1, stride, stride, 1], padding='SAME')
      # No biases in resnet convolutional layers
      """
      biases = self._constant_variable('biases', [output_maps], 0.0,
                                       weight_decay, lr_mult, restore,
                                       self._append(restscope, 'biases'))
      output = tf.nn.bias_add(output, biases)
      """
      return output


  def _full_layer(self, output, output_maps,
                  weight_decay=0.0, lr_mult=1.0, restore=True,
                  scope='full', restscope=None):
    input_maps = self._output_elem(output)
    with tf.variable_scope(scope):
      output = tf.reshape(output, [-1, input_maps])
      weights = self._normal_variable('weights', [input_maps, output_maps],
                                      1.0/input_maps, weight_decay, lr_mult, restore,
                                      self._append(restscope, 'weights'))
      output = tf.matmul(output, weights)
      biases = self._constant_variable('biases', [output_maps], 0.0,
                                       weight_decay, lr_mult, restore,
                                       self._append(restscope,'biases'))
      output = tf.add(output, biases)
      return output


  ### COMPOSED LAYERS ###

  def _conv_block(self, output, output_maps,
                  filter_size=3, stride=1,
                  weight_decay=0.0, lr_mult=1.0, restore=True,
                  scope='conv_block', restscope=None):
    with tf.variable_scope(scope):
      output = self._conv_layer(output, output_maps, filter_size, stride,
                                weight_decay, lr_mult, restore,
                                restscope=restscope)
      output = self._batch_norm(output, lr_mult, restore,
                                restscope=restscope)
      output = self._nonlinearity(output)
      self._activation_summary(output)
      return output


  def _full_block(self, output, output_maps,
                  weight_decay=0.0, lr_mult=1.0, restore=True,
                  scope='full_block', restscope=None):
    with tf.variable_scope(scope):
      output = self._full_layer(output, output_maps,
                                weight_decay, lr_mult, restore,
                                restscope=restscope)
      output = self._batch_norm(output, lr_mult, restore,
                                restscope=restscope)
      output = self._nonlinearity(output)
      self._activation_summary(output)
      return output


  def _resn_block(self, output, inside_maps,
                  output_maps=None, stride=1,
                  weight_decay=0.0, lr_mult=1.0, restore=True,
                  scope='resn_block', restscope=None):
    with tf.variable_scope(scope):
      input = output
      input_maps = self.dims(output, 3)
      if (not output_maps):
        output_maps = input_maps
      output = self._conv_block(output, inside_maps, 1, stride,
                                weight_decay, lr_mult, restore, scope='in',
                                restscope=self._append(restscope, 'a'))
      output = self._conv_block(output, inside_maps, 3, 1,
                                weight_decay, lr_mult, restore, scope='middle',
                                restscope=self._append(restscope, 'b'))
      output = self._conv_block(output, output_maps, 1, 1,
                                weight_decay, lr_mult, restore, scope='out',
                                restscope=self._append(restscope, 'c'))
      if (output_maps != input_maps or stride != 1):
        with tf.variable_scope('projection'):
          input = self._conv_layer(input, output_maps, 1, stride,
                                   weight_decay, lr_mult, restore,
                                   restscope=self._append(restscope, 'shortcut'))
          input = self._batch_norm(input, lr_mult, restore,
                                   restscope=self._append(restscope, 'shortcut'))
      output += input
      self._activation_summary(output)
      return output


  def _last_block(self, output,
                  weight_decay=0.0, lr_mult=1.0, restore=True,
                  scope='last_block', restscope=None):
    with tf.variable_scope(scope):
      # First do pooling to size 1x1
      input_dims = self.dims(output)
      # make sure that the input is squared
      assert input_dims[1] == input_dims[2]
      map_size = input_dims[1]
      if (map_size > 1):
        output = self._pool_layer(output,
                                  filter_size=map_size, stride=map_size, func='avg')
      output = self._full_layer(output, Reader.CLASSES_NUM,
                                weight_decay, lr_mult, restore,
                                restscope=restscope)
      # No batch_norm in resnet for the fc-layer
      #output = self._batch_norm(output, lr_mult, restore, restscope=restscope)

      # Last layer does not have nonlinearity!
      self._activation_summary(output)
      return output


  def _construct(self, output):
    #with tf.variable_scope('0'):
    #  output = self._batch_norm(output, restore=False)

    with tf.variable_scope('1'):
      output = self._conv_block(output, output_maps=64,
                                filter_size=7, stride=2, lr_mult=0.0,
                                scope='conv_block', restscope='scale1')

    with tf.variable_scope('2'):
      output = self._pool_layer(output)
      for i in xrange(0, 3):
        output = self._resn_block(output, inside_maps=64,
                                  output_maps=256, stride=1, lr_mult=0.0,
                                  scope=str(i+1), restscope='scale2/block'+str(i+1))

    with tf.variable_scope('3'):
      output = self._resn_block(output, inside_maps=128,
                                output_maps=512, stride=2, lr_mult=0.0,
                                scope='1', restscope='scale3/block1')
      for i in xrange(1, 4):
        output = self._resn_block(output, inside_maps=128, lr_mult=0.0,
                                  scope=str(i+1), restscope='scale3/block'+str(i+1))

    with tf.variable_scope('4'):
      output = self._resn_block(output, inside_maps=256,
                                output_maps=1024, stride=2, lr_mult=1.0,
                                scope='1', restscope='scale4/block1')
      for i in xrange(1, 6):
        output = self._resn_block(output, inside_maps=256, lr_mult=1.0,
                                  scope=str(i+1), restscope='scale4/block'+str(i+1))

    with tf.variable_scope('5'):
      output = self._resn_block(output, inside_maps=512,
                                output_maps=2048, stride=2, lr_mult=1.0,
                                scope='1', restscope='scale5/block1')
      for i in xrange(1, 3):
        output = self._resn_block(output, inside_maps=512, lr_mult=1.0,
                                  scope=str(i+1), restscope='scale5/block'+str(i+1))

    with tf.variable_scope('6'):
      output = self._last_block(output, lr_mult=1.0, restore=False)

    return output

  """
  def _add_loss_summaries(self, total_loss):
    # loss for test is obtained directly from each batch
    if (not self.is_train):
      return
    losses = tf.get_collection(Network.LOSSES_NAME)
    ema = tf.train.ExponentialMovingAverage(decay=self._decay, name='moving_average')
    ema_apply_op = ema.apply(losses + [total_loss])
    with tf.control_dependencies([ema_apply_op]):
      for l in losses + [total_loss]:
        self._set_restoring(ema.average(l), restore=False)
        tf.scalar_summary(l.op.name, tf.identity(ema.average(l)))
  """

  def loss(self, labels):
    with tf.variable_scope('losses/' + self._prefix):
      # labels must be already tf.int64
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        self._output, labels, name='cross_entropy_per_example'
      )
      cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
      tf.add_to_collection(Network.LOSSES_NAME, cross_entropy_mean)
      total_loss = tf.add_n(tf.get_collection(Network.LOSSES_NAME), name='total_loss')
      # We will be tracking training error manually as well until they fix ema
      #self._add_loss_summaries(total_loss)
      return total_loss


  def probs(self):
    return tf.nn.softmax(self._output)


  """
  def output(self):
    return self._output
  """

  """
  def accuracy(self, labels):
    with tf.variable_scope('accuracy'):
      #predicted = tf.argmax(self._output, dimension=1)
      #matched = tf.equal(predicted, labels)
      #accuracy = tf.reduce_mean(matched)
      accuracy = tf.nn.in_top_k(self._output, labels, 1)
      return accuracy
  """