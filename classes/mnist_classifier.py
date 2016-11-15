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

class MnistClassifier(Network):

  def __init__(self, params, images):
    super(MnistClassifier, self).__init__(params)
    output = images

    with tf.variable_scope('mnist'):
      output = self._conv_block(output, output_maps=32,
                                filter_size=3, stride=1, lr_mult=1.0,
                                scope='conv_block1', restore=False)
      output = self._pool_layer(output, filter_size=3, stride=2, func='max', scope='pool1')
      output = self._conv_block(output, output_maps=32,
                                filter_size=5, stride=1, lr_mult=1.0,
                                scope='conv_block2', restore=False)
      output = self._pool_layer(output, filter_size=3, stride=2, func='max', scope='pool2')
      output = self._full_block(output, output_maps=256, features=None,
                                weight_decay=0.0, lr_mult=1.0,
                                scope='full_block', restore=False)
      output = self._last_block(output, features=None, lr_mult=1.0,
                                scope='last_block', restore=False)
    self._output = output