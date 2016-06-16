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
from tensorflow.python.framework import dtypes
import json
import os
import math
#import functools

import sys
sys.path.append('/home/sdemyanov/synapse/molemap')
import utils.resize_image_patch
reload(utils.resize_image_patch)

class Reader(object):

  LIST_DIR = '/home/sdemyanov/synapse/molemap/tensorflow/filedicts'
  FOLD_NAMES = {'train': 'train.json',
                'valid': 'valid.json',
                'test': 'test.json'}

  CLASSES_NUM = 15
  CHANNEL_NUM = 3
  MEAN_CHANNEL_VALUES = [194, 161, 155]
  MAX_PIXEL_VALUE = 255
  MIN_INPUT_SIZE = 600

  SCALE_RATIO = 2.0
  ZOOM_RANGE = [1.1, 1.8]
  ZOOM_MEAN = (ZOOM_RANGE[0] + ZOOM_RANGE[0]) / 2
  SCALE_SIZE = 256
  SCALE_SIZE = 300
  #assert MIN_INPUT_SIZE / SCALE_RATIO / ZOOM_RANGE[1] >= SCALE_SIZE
  CROP_SHAPE = (SCALE_SIZE, SCALE_SIZE, CHANNEL_NUM)
  IMAGE_SIZE = 224

  TRAINING_PREFIX = 'training'
  TESTING_PREFIX = 'testing'

  #CHANGE
  MIN_QUEUE_FRACTION = 0.1
  BRIGHTNESS_DELTA = 0.25
  MIN_CONTRAST_FACTOR = 0.5
  MAX_CONTRAST_FACTOR = 1.8


  def __init__(self, fold_name):
    fold_path = os.path.join(Reader.LIST_DIR, Reader.FOLD_NAMES[fold_name])
    assert os.path.exists(fold_path)
    self._read_fold_list(fold_path)
    self._get_lists()


  def _get_lists(self):
    self._image_list = []
    self._label_list = []
    self._fname_list = []
    for fpath, label in self._file_dict.items():
      self._image_list.append(fpath)
      self._label_list.append(label)
      self._fname_list.append(fpath.split('/')[-1])


  def _read_fold_list(self, fold_path):
    with open(fold_path, 'rb') as handle:
      self._file_dict = json.load(handle)
    self.fold_size = len(self._file_dict)


  def _read_image_from_disk(self, input_queue):
    with tf.variable_scope('reading'):
      label = input_queue[1]
      filename = input_queue[2]
      file_contents = tf.read_file(input_queue[0])
      image = tf.image.decode_jpeg(file_contents, ratio=Reader.SCALE_RATIO)
      image = tf.cast(image, tf.float32)
      image.set_shape([None, None, None])
      return image, label, filename


  def _generate_image_and_label_batch(self, image, label, filename,
                                      batch_size, min_queue_examples, shuffle):
    """Construct a queued batch of images and labels.

    Args:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      label: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.

    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if (shuffle):
      images, labels, filenames = tf.train.shuffle_batch(
        [image, label, filename],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples,
        name='0/training'
      )
    else:
      images, labels, filenames = tf.train.batch(
        [image, label, filename],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        name='0/testing'
      )
    return images, labels, filenames


  def _rotate90(self, image):
    with tf.variable_scope('rotate90'):
      image = tf.image.transpose_image(image)
      image = tf.image.flip_left_right(image)
      return image


  def _random_rotate90(self, image):
    with tf.variable_scope('random_rotate90'):

      def rotate():
        return self._rotate90(image)

      def skip():
        return image

      rotnum = tf.random_uniform(shape=[1], minval=0, maxval=4, dtype=tf.int32)
      rotind = tf.constant(0, dtype=tf.int32)
      for i in xrange(4):
        image = tf.cond(tf.less(rotind, rotnum[0]), rotate, skip)
        rotind += 1
      return image


  def _zoom_and_crop(self, image, size, zoom=None):
    # if no zoom is given, we use max possible zoom
    with tf.variable_scope('zoom_and_crop'):
      if (zoom is not None):
        imshape = tf.to_float(tf.shape(image))
        minsize = tf.minimum(imshape[0], imshape[1])
        maxcoef = float(size+1) / minsize
        zoomcoef = 1.0/zoom
        rescoef = maxcoef / zoomcoef
        new_size = tf.to_int32(imshape * rescoef) + 1
        image = tf.image.resize_images(image, new_size[0], new_size[1])

      image = tf.image.resize_image_with_crop_or_pad(
        image, size, size, dynamic_shape=True
      )
      image.set_shape([size, size, Reader.CHANNEL_NUM])
      return image

  def _scale_and_crop(self, image, size):
    return self._zoom_and_crop(image, size, 1.0)


  def _central_crop(self, image, size):
    return self._zoom_and_crop(image, size)


  def _random_zoom_and_crop(self, image, zoom_range, size):
    with tf.variable_scope('random_zoom_and_crop'):
      zoom = tf.random_uniform(shape=[1], minval=zoom_range[0],
                                   maxval=zoom_range[1], dtype=tf.float32)
      return self._zoom_and_crop(image, size, zoom[0])


  def _train_transform(self, image):
    with tf.variable_scope('train_transform'):
      #image = self._random_zoom_and_crop(image, Reader.ZOOM_RANGE, Reader.SCALE_SIZE)
      image = self._scale_and_crop(image, Reader.SCALE_SIZE)
      image_shape = (Reader.IMAGE_SIZE, Reader.IMAGE_SIZE, Reader.CHANNEL_NUM)
      image = tf.random_crop(image, image_shape)
      image = self._random_rotate90(image)
      image = tf.image.random_flip_left_right(image)
      delta = Reader.BRIGHTNESS_DELTA * Reader.MAX_PIXEL_VALUE
      image = tf.image.random_brightness(image, max_delta=delta)
      image = tf.image.random_contrast(image,
                                       lower=Reader.MIN_CONTRAST_FACTOR,
                                       upper=Reader.MAX_CONTRAST_FACTOR)
      return image


  def _test_transform(self, image):
    with tf.variable_scope('test_transform'):
      #image = self._zoom_and_crop(image, Reader.SCALE_SIZE, Reader.ZOOM_MEAN)
      image = self._scale_and_crop(image, Reader.SCALE_SIZE)
      image = self._central_crop(image, Reader.IMAGE_SIZE)
      mean_contrast = math.sqrt(Reader.MIN_CONTRAST_FACTOR * Reader.MAX_CONTRAST_FACTOR)
      image = tf.image.adjust_contrast(image, mean_contrast)

      return image


  def inputs(self, batch_size, is_train):

    with tf.variable_scope('inputs'):
      image_op = ops.convert_to_tensor(self._image_list, dtype=dtypes.string)
      label_op = ops.convert_to_tensor(self._label_list, dtype=dtypes.int64)
      fname_op = ops.convert_to_tensor(self._fname_list, dtype=dtypes.string)

      input_queue = tf.train.slice_input_producer(
        [image_op, label_op, fname_op], shuffle=is_train
      )
      (image, label, filename) = self._read_image_from_disk(input_queue)

      image = self._scale_and_crop(image, Reader.MIN_INPUT_SIZE)
      # now the image have shape, which we can use
      if (is_train):
        image = self._train_transform(image)
      else:
        image = self._test_transform(image)

      # In order to avoid batch_norm on the input, so we can save our time
      # by not propagating gradients till the end if we do fine-tuning
      image = (image - Reader.MEAN_CHANNEL_VALUES) / Reader.MAX_PIXEL_VALUE

      # Ensure that the random shuffling has good mixing properties
      min_queue_examples = int(self.fold_size * Reader.MIN_QUEUE_FRACTION)

      # Generate a batch of images and labels by building up a queue of examples.
      images, labels, filenames = self._generate_image_and_label_batch(
        image, label, filename, batch_size, min_queue_examples, shuffle=is_train
      )
      # Display the training images in the visualizer
      if (is_train):
        prefix = Reader.TRAINING_PREFIX
      else:
        prefix = Reader.TESTING_PREFIX
      # images are too heavy for summary, but uncomment for check if needed
      #tf.image_summary(prefix, images, max_images=batch_size)
      tf.histogram_summary(prefix + 'image_values', images)
      tf.histogram_summary(prefix + 'labels', labels)

      return images, labels, filenames
