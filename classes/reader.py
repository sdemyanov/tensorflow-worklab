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
from tensorflow.python.framework import ops
import math

class Reader(object):

  TRAINING_PREFIX = 'training'
  TESTING_PREFIX = 'testing'

  IMAGE_SIZE = None
  MAX_ZOOM = 1.0
  SHIFT_RATIO = 0.0
  BRIGHTNESS_DELTA = 0.0
  MIN_CONTRAST_FACTOR = 1.0
  MAX_CONTRAST_FACTOR = 1.0
  RANDOM_ROTATE = False
  RANDOM_FLIP = False

  QUEUE_CAPACITY = 5
  NUM_THREADS = 8


  def __init__(self, fold_name):
    lists = self._get_lists(fold_name)
    self._opkeys = []
    self._ops = []
    i = 0
    self.fold_size = None
    for key, list in lists.items():
      self._opkeys.append(key)
      self._ops.append(ops.convert_to_tensor(list))
      i += 1
      if self.fold_size is None:
        self.fold_size = len(list)
      else:
        assert self.fold_size == len(list)


  def _get_lists(cls, fold_name):
    raise NotImplementedError


  @staticmethod
  def _rotate90(image):
    with tf.variable_scope('rotate90'):
      image = tf.image.transpose_image(image)
      image = tf.image.flip_left_right(image)
      return image


  @classmethod
  def _random_rotate90(cls, image):
    with tf.variable_scope('random_rotate90'):
      rotnum = tf.random_uniform(shape=[1], minval=0, maxval=4, dtype=tf.int32)
      rotind = tf.constant(0, dtype=tf.int32)
      for i in range(4):
        image = tf.cond(tf.less(rotind, rotnum[0]), lambda: cls._rotate90(image), lambda: image)
        rotind += 1
      return image


  @staticmethod
  def _zoom_and_crop(image, size, zoom=None):
    # if no zoom is given, we use max possible zoom
    with tf.variable_scope('zoom_and_crop'):
      channel_num = image.get_shape().as_list()[2]
      if zoom is not None:
        imshape = tf.to_float(tf.shape(image))
        minsize = tf.minimum(imshape[0], imshape[1])
        maxcoef = float(size+1) / minsize
        rescoef = maxcoef * zoom
        if rescoef != 1.0:
          new_size = tf.to_int32(imshape * rescoef) + 1
          image = tf.image.resize_images(image, new_size[0:2])

      image.set_shape([size, size, channel_num])
      image = tf.image.resize_image_with_crop_or_pad(image, size, size)
      image.set_shape([size, size, channel_num])
      return image


  @classmethod
  def _get_scale_size(cls):
    return int(cls.IMAGE_SIZE * (1 + 2 * cls.SHIFT_RATIO))


  @classmethod
  def _scale_and_crop(cls, image, size):
    return cls._zoom_and_crop(image, size, 1.0)


  @classmethod
  def _central_crop(cls, image, size):
    return cls._zoom_and_crop(image, size)


  @classmethod
  def _random_zoom_and_crop(cls, image, size, max_zoom):
    with tf.variable_scope('random_zoom_and_crop'):
      zoom = tf.random_uniform(shape=[1], minval=1.0, maxval=max_zoom, dtype=tf.float32)
      return cls._zoom_and_crop(image, size, zoom[0])


  @classmethod
  def _train_transform(cls, image):
    with tf.variable_scope('train_transform'):
      image = cls._random_zoom_and_crop(image, cls._get_scale_size(), cls.MAX_ZOOM)
      if cls.SHIFT_RATIO > 0:
        channel_num = image.get_shape().as_list()[2]
        image_shape = (cls.IMAGE_SIZE, cls.IMAGE_SIZE, channel_num)
        image = tf.random_crop(image, image_shape)
      if cls.RANDOM_ROTATE:
        image = cls._random_rotate90(image)
      if cls.RANDOM_FLIP:
        image = tf.image.random_flip_left_right(image)
      if cls.BRIGHTNESS_DELTA > 0:
        delta = cls.BRIGHTNESS_DELTA
        image = tf.image.random_brightness(image, max_delta=delta)
      if cls.MIN_CONTRAST_FACTOR < cls.MAX_CONTRAST_FACTOR:
        image = tf.image.random_contrast(image,
                                         lower=cls.MIN_CONTRAST_FACTOR,
                                         upper=cls.MAX_CONTRAST_FACTOR)
      return image


  @classmethod
  def _test_transform(cls, image):
    with tf.variable_scope('test_transform'):
      zoom_mean = (1.0 + cls.MAX_ZOOM) / 2
      image = cls._zoom_and_crop(image, cls._get_scale_size(), zoom_mean)
      if cls.SHIFT_RATIO > 0:
        image = cls._central_crop(image, cls.IMAGE_SIZE)
      if cls.MIN_CONTRAST_FACTOR < cls.MAX_CONTRAST_FACTOR:
        mean_contrast = math.sqrt(cls.MIN_CONTRAST_FACTOR * cls.MAX_CONTRAST_FACTOR)
        image = tf.image.adjust_contrast(image, mean_contrast)
      return image


  @classmethod
  def _generate_batches(cls, tensors, batch_size):
    args = {'tensors': tensors,
            'batch_size': batch_size,
            'num_threads': cls.NUM_THREADS,
            'capacity': cls.QUEUE_CAPACITY * batch_size}
    #if is_train:
    #  args['min_after_dequeue'] = min_queue_examples
    #  return tf.train.shuffle_batch(**args)
    #else:
    return tf.train.batch(**args)


  def _collect_tensors(self, keys, input_queue, is_train):
    raise NotImplementedError


  def inputs(self, batch_size, is_train):
    with tf.variable_scope('inputs'):
      input_queue = tf.train.slice_input_producer(self._ops, shuffle=is_train)
      tensors = self._collect_tensors(self._opkeys, input_queue, is_train)
      # Generate a batch of images and labels by building up a queue of examples.
      input_batches = Reader._generate_batches(tensors, batch_size)
      input_dict = {}
      for i, key in enumerate(self._opkeys):
        input_dict[key] = input_batches[i]

      # Display the training images in the visualizer
      #if is_train:
      #  prefix = Reader.TRAINING_PREFIX
      #else:
      #  prefix = Reader.TESTING_PREFIX
      # images are too heavy for summary, but uncomment for check if needed
      # tf.image_summary(prefix, images, max_images=batch_size)
      #tf.histogram_summary(prefix + '/image_values', input_dict['images'])
      #tf.histogram_summary(prefix + '/labels', input_dict['labels'])

      return input_dict