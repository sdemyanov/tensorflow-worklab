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
import numpy as np

from reader import Reader

import paths

class MnistReader(Reader):

  IMAGE_SIZE = 28
  CLASSES_NUM = 10
  CHANNEL_NUM = 1
  MAX_PIXEL_VALUE = 255
  MEAN_CHANNEL_VALUES = [33.3184214498]
  IMAGE_SHAPE = [IMAGE_SIZE, IMAGE_SIZE, CHANNEL_NUM]


  def _get_lists(self, fold_name):
    data_dir = paths.DATA_DIR

    lists = {}
    if fold_name == 'train':
      fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
      loaded = np.fromfile(file=fd, dtype=np.uint8)
      lists['images'] = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)
      fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
      loaded = np.fromfile(file=fd, dtype=np.uint8)
      lists['labels'] = loaded[8:].reshape((60000)).astype(np.float)
    elif fold_name == 'test':
      fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
      loaded = np.fromfile(file=fd, dtype=np.uint8)
      lists['images'] = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)
      fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
      loaded = np.fromfile(file=fd, dtype=np.uint8)
      lists['labels'] = loaded[8:].reshape((10000)).astype(np.float)
    else:
      assert False, 'Wrong fold name: %s' % fold_name

    lists['images'] = np.asarray(lists['images'])
    return lists


  def _collect_tensors(self, keys, input_queue, is_train):
    tensors = []
    for i, key in enumerate(keys):
      with tf.variable_scope(key):
        if key == 'images':
          image = tf.cast(input_queue[i], tf.float32)
          image.set_shape(MnistReader.IMAGE_SHAPE)
          #print 'this', image.get_shape().as_list()
          #print 'scale_size', self._get_scale_size()
          #image = self._random_zoom_and_crop(image, self._get_scale_size(), MnistReader.MAX_ZOOM)
          #print 'last', image.get_shape().as_list()
          #if is_train:
          #  image = MnistReader._train_transform(image)
          #else:
          #  image = MnistReader._test_transform(image)
          image = (image - MnistReader.MEAN_CHANNEL_VALUES) / MnistReader.MAX_PIXEL_VALUE
          tensors.append(image)
        elif key == 'labels':
          tensors.append(tf.cast(input_queue[i], tf.int64))
    return tensors