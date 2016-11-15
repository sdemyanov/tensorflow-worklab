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
import json
import os

from reader import Reader

import paths

class FileReader(Reader):

  IMAGE_SIZE = 224
  CLASSES_NUM = 1000
  CHANNEL_NUM = 3
  MAX_PIXEL_VALUE = 255
  MEAN_CHANNEL_VALUES = [122.67891434, 116.66876762, 104.00698793]
  MIN_INPUT_SIZE = 256
  IMAGE_SHAPE = [IMAGE_SIZE, IMAGE_SIZE, CHANNEL_NUM]
  MAX_IMAGE_NUM = None


  def _get_lists(self, fold_name):
    fold_path = os.path.join(paths.DICTS_DIR, fold_name + '.json')
    with open(fold_path, 'r') as handle:
      file_dict = json.load(handle)
    fold_size = len(file_dict)
    if FileReader.MAX_IMAGE_NUM is not None:
      fold_size = FileReader.MAX_IMAGE_NUM
    lists = {}
    lists['images'] = []
    lists['labels'] = []
    #lists['fnames'] = []
    i = 0
    for fpath, info in file_dict.items():
      lists['images'].append(fpath) # images
      path_parts = fpath.split('/')
      lists['fnames'].append(path_parts[-1])  # filenames
      #lists['labels'].append(info)  # labels
      i += 1
      if i >= fold_size:
        break
    return lists


  def _collect_tensors(self, keys, input_queue, is_train):
    tensors = []
    for i, key in enumerate(keys):
      with tf.variable_scope(key):
        if key == 'images':
          file_contents = tf.read_file(input_queue[i])
          image = tf.image.decode_jpeg(file_contents, ratio=1.0)
          image = tf.cast(image, tf.float32)
          image.set_shape([None, None, FileReader.CHANNEL_NUM])
          image = self._scale_and_crop(image, FileReader.MIN_INPUT_SIZE)
          if is_train:
            image = FileReader._train_transform(image)
          else:
            image = FileReader._test_transform(image)
          image = (image - FileReader.MEAN_CHANNEL_VALUES) / FileReader.MAX_PIXEL_VALUE
          tensors.append(image)
        elif key == 'labels':
          tensors.append(tf.cast(input_queue[i], tf.int64))
    return tensors