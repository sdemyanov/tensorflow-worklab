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

from network import Network

class ResNet50(Network):

  def __init__(self, params, images):
    super(ResNet50, self).__init__(params)
    output = images

    with tf.variable_scope('1'):
      output = self._conv_block(output, output_maps=64,
                                filter_size=7, stride=2, lr_mult=0.0,
                                scope='conv_block', restore='scale1')

    with tf.variable_scope('2'):
      output = self._pool_layer(output)
      for i in range(0, 3):
        output = self._resn_block(output, inside_maps=64,
                                  output_maps=256, stride=1, lr_mult=0.0,
                                  scope=str(i + 1), restore='scale2/block' + str(i + 1))

    with tf.variable_scope('3'):
      output = self._resn_block(output, inside_maps=128,
                                output_maps=512, stride=2, lr_mult=0.0,
                                scope='1', restore='scale3/block1')
      for i in range(1, 4):
        output = self._resn_block(output, inside_maps=128, lr_mult=0.0,
                                  scope=str(i + 1), restore='scale3/block' + str(i + 1))

    with tf.variable_scope('4'):
      output = self._resn_block(output, inside_maps=256,
                                output_maps=1024, stride=2, lr_mult=1.0,
                                scope='1', restore='scale4/block1')
      for i in range(1, 6):
        output = self._resn_block(output, inside_maps=256, lr_mult=1.0,
                                  scope=str(i + 1), restore='scale4/block' + str(i + 1))

    with tf.variable_scope('5'):
      output = self._resn_block(output, inside_maps=512,
                                output_maps=2048, stride=2, lr_mult=1.0,
                                scope='1', restore='scale5/block1')
      for i in range(1, 3):
        output = self._resn_block(output, inside_maps=512, lr_mult=1.0,
                                  scope=str(i + 1), restore='scale5/block' + str(i + 1))

    with tf.variable_scope('6'):
      output = self._full_pool(output)
      output = self._last_block(output, features=None, lr_mult=1.0,
                                scope='last_block', restore=False)
    self._output = output