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

class Network(object):

  STDDEV = 1e-4
  BN_EPS = 1e-5

  INIT_SCOPE = 'init_scope'
  LOSSES_NAME = 'losses'
  TRAINING_PREFIX = 'training'
  TESTING_PREFIX = 'testing'

  def __init__(self, params):
    #self.batch_size = self.shape(images, 0)
    self.is_train = params['is_train']
    self.decay = params['decay']
    self.classes_num = params['classes_num']
    if self.is_train:
      self._prefix = Network.TRAINING_PREFIX
    else:
      self._prefix = Network.TESTING_PREFIX
    self.lr_multipliers = {}
    self.rest_names = {}
    self._output = None # redefine it in a derived class

  @staticmethod
  def shape(output, dim=None):
    shape = output.get_shape().as_list()
    if dim is not None:
      return shape[dim]
    return shape

  @classmethod
  def _output_elem(cls, output):
    shape = cls.shape(output)
    elem = 1
    for i in range(1, len(shape)):
      elem *= shape[i]
    return elem

  @staticmethod
  def _append(restore, appendix):
    if (restore == False or restore == True):
      return restore
    return restore + '/' + appendix

  @staticmethod
  def _set_weight_decay(var, weight_decay):
    if (weight_decay is not None) and (weight_decay > 0):
      weight_decay = tf.mul(tf.nn.l2_loss(var), weight_decay) #, name=var.name)
      tf.add_to_collection(Network.LOSSES_NAME, weight_decay)


  def _set_lr_mult(self, var, lr_mult):
    self.lr_multipliers[var.op.name] = lr_mult


  def _set_restoring(self, var, restore=True):
    if restore == False:
      return
    if restore == True:
      restore = var.op.name
    self.rest_names[restore] = var
    #print("%s: %s" %(var.op.name, restore))


  def _variable(self, name, shape, initializer,
                weight_decay, lr_mult, restore=True):
    is_trainable = (lr_mult > 0)
    var = tf.get_variable(name=name, shape=shape,
                          initializer=initializer, trainable=is_trainable)
    if is_trainable:
      Network._set_weight_decay(var, weight_decay)
      self._set_lr_mult(var, lr_mult)

    self._set_restoring(var, restore)
    return var


  def _normal_variable(self, name, shape, stddev,
                       weight_decay, lr_mult, restore=True):
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    return self._variable(name, shape, initializer,
                          weight_decay, lr_mult, restore)


  def _constant_variable(self, name, shape, value,
                         weight_decay, lr_mult, restore=True):
    initializer = tf.constant_initializer(value)
    return self._variable(name, shape, initializer,
                          weight_decay, lr_mult, restore)


  ### BASIC LAYERS ###

  def _activation_summary(self, output, scope='summary'):
    return
    # no summary for test network
    #if not self.is_train:
    #  return
    #with tf.variable_scope(scope):
    #  tensor_name = output.op.name
    #  tf.histogram_summary('activations/' + tensor_name, output)
    #  zero_fraction = tf.nn.zero_fraction(output)
    #  tf.scalar_summary('sparsity/' + tensor_name, zero_fraction)


  def _nonlinearity(self, output):
    output = tf.nn.relu(output, name='relu')
    return output


  def _batch_norm(self, output,
                  lr_mult=1.0, scope='bn', restore=True):
    with tf.variable_scope(scope):
      shape = Network.shape(output)
      # we don't squeeze only the last dimension, i.e. feature maps
      squeeze_dims = range(len(shape)-1)
      input_maps = shape[-1]
      batch_mean, batch_var = tf.nn.moments(output, squeeze_dims, name='moments')
      ema = tf.train.ExponentialMovingAverage(decay=self.decay)
      ema_apply_op = ema.apply([batch_mean, batch_var])
      # Needed for partial restoration from an existing model
      self._set_restoring(ema.average(batch_mean),
                          Network._append(restore, 'moving_mean'))
      self._set_restoring(ema.average(batch_var),
                          Network._append(restore, 'moving_variance'))
      if self.is_train: # and lr_mult > 0):
        with tf.control_dependencies([ema_apply_op]):
          mean, var = tf.identity(batch_mean), tf.identity(batch_var)
      else:
        #mean, var = batch_mean, batch_var
        mean, var = ema.average(batch_mean), ema.average(batch_var)

      beta = self._constant_variable('beta', [input_maps], 0.0, 0.0,
                                     lr_mult, Network._append(restore, 'beta'))
      gamma = self._constant_variable('gamma', [input_maps], 1.0, 0.0,
                                      lr_mult, Network._append(restore, 'gamma'))
      output = tf.nn.batch_normalization(output, mean, var, beta, gamma, Network.BN_EPS)
      return output


  def _pool_layer(self, output, filter_size=3, stride=2, func='max', scope='pool'):
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
                  weight_decay=0.0, lr_mult=1.0,
                  scope='conv', restore=True):
    input_maps = Network.shape(output, 3)
    filter_shape = [filter_size, filter_size, input_maps, output_maps]
    with tf.variable_scope(scope):
      kernel = self._normal_variable('weights', filter_shape, Network.STDDEV,
                                     weight_decay, lr_mult,
                                     Network._append(restore, 'weights'))
      output = tf.nn.conv2d(output, kernel, [1, stride, stride, 1], padding='SAME')
      # No biases in resnet convolutional layers
      """
      biases = self._constant_variable('biases', [output_maps], 0.0,
                                       weight_decay, lr_mult,
                                       Network._append(restore, 'biases'))
      output = tf.nn.bias_add(output, biases)
      """
      return output


  def _full_layer(self, output, output_maps, features=None,
                  weight_decay=0.0, lr_mult=1.0,
                  scope='full', restore=True):
    input_maps = Network._output_elem(output)
    with tf.variable_scope(scope):
      output = tf.reshape(output, [-1, input_maps])
      if (features is not None):
        output = tf.concat(concat_dim=1, values=[output, features])
        input_maps += Network._output_elem(features)
      weights = self._normal_variable('weights', [input_maps, output_maps],
                                      1.0/input_maps, weight_decay, lr_mult,
                                      Network._append(restore, 'weights'))
      output = tf.matmul(output, weights)
      biases = self._constant_variable('biases', [output_maps], 0.0,
                                       weight_decay, lr_mult,
                                       Network._append(restore,'biases'))
      output = tf.add(output, biases)
      return output


  ### COMPOSED LAYERS ###

  def _full_pool(self, output, scope='full_pool'):
    input_dims = Network.shape(output)
    if (len(input_dims) == 2):
      return
    assert len(input_dims) == 4
    # make sure that the input is squared
    assert input_dims[1] == input_dims[2]
    with tf.variable_scope(scope):
      map_size = input_dims[1]
      if (map_size > 1):
        output = self._pool_layer(output, filter_size=map_size,
                                  stride=map_size, func='avg')
      output = tf.reshape(output, [-1, input_dims[3]])
    return output


  def _conv_block(self, output, output_maps,
                  filter_size=3, stride=1,
                  weight_decay=0.0, lr_mult=1.0,
                  scope='conv_block', restore=True):
    with tf.variable_scope(scope):
      output = self._conv_layer(output, output_maps, filter_size, stride,
                                weight_decay, lr_mult, restore=restore)
      output = self._batch_norm(output, lr_mult, restore=restore)
      output = self._nonlinearity(output)
      self._activation_summary(output)
      return output


  def _resn_block(self, output, inside_maps,
                  output_maps=None, stride=1,
                  weight_decay=0.0, lr_mult=1.0,
                  scope='resn_block', restore=True):
    with tf.variable_scope(scope):
      residual = tf.identity(output)
      input_maps = Network.shape(output, 3)
      if not output_maps:
        output_maps = input_maps
      if (output_maps != input_maps or stride != 1):
        with tf.variable_scope('projection'):
          output = self._conv_layer(output, output_maps, 1, stride,
                                    weight_decay, lr_mult,
                                    restore=Network._append(restore, 'shortcut'))
          output = self._batch_norm(output, lr_mult,
                                    restore=Network._append(restore, 'shortcut'))

      residual = self._conv_block(residual, inside_maps, 1, stride,
                                weight_decay, lr_mult, scope='in',
                                restore=Network._append(restore, 'a'))
      residual = self._conv_block(residual, inside_maps, 3, 1,
                                weight_decay, lr_mult, scope='middle',
                                restore=Network._append(restore, 'b'))
      residual = self._conv_block(residual, output_maps, 1, 1,
                                weight_decay, lr_mult, scope='out',
                                restore=Network._append(restore, 'c'))
      output += residual
      self._activation_summary(output)
      return output


  def _full_block(self, output, output_maps, features=None,
                  weight_decay=0.0, lr_mult=1.0,
                  scope='full_block', restore=True):
    with tf.variable_scope(scope):
      output = self._full_layer(output, output_maps, features,
                                weight_decay, lr_mult, restore=restore)
      output = self._batch_norm(output, lr_mult, restore=restore)
      output = self._nonlinearity(output)
      self._activation_summary(output)
      return output


  def _last_block(self, output, features=None,
                  weight_decay=0.0, lr_mult=1.0,
                  scope='last_block', restore=True):
    with tf.variable_scope(scope):
      output = self._full_layer(output, self.classes_num, features,
                                weight_decay, lr_mult, restore=restore)
      # No batch_norm in resnet for the fc-layer
      # output = self._batch_norm(output, lr_mult, restore, restscope=restscope)
      # Last layer does not have nonlinearity!
      self._activation_summary(output)
      return output


  def probs(self):
    return tf.nn.softmax(self._output)


  def cross_entropy_losses(self, labels):
    # labels must be already tf.int64
    with tf.variable_scope('cross_entropy_losses/' + self._prefix):
      return tf.nn.sparse_softmax_cross_entropy_with_logits(
        self._output, labels, name='cross_entropy_per_example'
      )


  def losses(self, labels):
    cross_entropy = self.cross_entropy_losses(labels)
    with tf.variable_scope('total_loss/' + self._prefix):
      cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
      tf.add_to_collection(Network.LOSSES_NAME, cross_entropy_mean)
      total_loss = tf.add_n(tf.get_collection(Network.LOSSES_NAME))
      # We will be tracking training error manually as well until they fix ema
      # self._add_loss_summaries(total_loss)
    return cross_entropy, total_loss


  """
  def output(self):
    return self._output
  """
