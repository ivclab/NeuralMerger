# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# MIT License
#
# Modifications copyright (c) 2018 Image & Vision Computing Lab, Institute of Information Science, Academia Sinica
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===================================================================================================================
import os
import tensorflow as tf



slim = tf.contrib.slim

_FILE_PATTERN = 'sound20_%s_%d.tfrecords'

SPLITS_TO_SIZES = {'train': 16636, 'test': 3727}

_NUM_CLASSES = 20

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 4',
}

IMAGE_HEIGHT = 32
IMAGE_WIDTH  = 32
def get_split(config,split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading flowers.
  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.
  Returns:
    A `Dataset` namedtuple.
  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
  all_file      = []
  reader        = tf.TFRecordReader()
  batch_size    = config.batch_size
  data_splitnum = config.data_split_num
  file_pattern  = _FILE_PATTERN

  if split_name == 'train':
    num_epochs = None
    for i in range(data_splitnum):
      all_file.append(os.path.join(dataset_dir, 'sound20/', file_pattern%(split_name,i)))
  elif split_name == 'test':
    num_epochs, batch_size = 1, 1
    all_file.append(os.path.join(dataset_dir, 'sound20/', file_pattern%(split_name,0)))
  elif split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  filename_queue = tf.train.string_input_producer(
  all_file, num_epochs=num_epochs, shuffle=False)

  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    features={
      'height': tf.FixedLenFeature([], tf.int64),
      'width': tf.FixedLenFeature([], tf.int64),
      'image_string': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.float32)
      })

  image = tf.decode_raw(features['image_string'], tf.uint8)
  label = tf.cast(features['label'], tf.float32)
  height = tf.cast(features['height'], tf.int32)
  width = tf.cast(features['width'], tf.int32)
  
  image = tf.reshape(image, [height, width, 1])
  resized_image = tf.image.resize_images(image,
    size=[IMAGE_HEIGHT,IMAGE_WIDTH])

  min_after_dequeue = 10000
  capacity = min_after_dequeue +3*batch_size

  images, labels = tf.train.shuffle_batch(
    [resized_image, label],
    batch_size=batch_size,
    capacity=capacity,
    num_threads=1,
    min_after_dequeue=min_after_dequeue,
    seed=config.random_seed)

  return images, labels, SPLITS_TO_SIZES[split_name]
