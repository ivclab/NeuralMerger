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
import os
import tensorflow as tf
import numpy as np


slim = tf.contrib.slim

_FILE_PATTERN = 'mvc_clothing_%s_%d.tfrecords'

SPLITS_TO_SIZES = {'train': 37485, 'test': 3000 }#origin 86423

_NUM_CLASSES = 13


IMAGE_HEIGHT = 224
IMAGE_WIDTH  = 224
def get_split(config, split_name, dataset_dir, batch_size,file_pattern=None, reader=None):
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
      all_file.append(os.path.join(dataset_dir, 'mvc_clothing/', file_pattern%(split_name,i)))
  elif split_name == 'test':
    num_epochs, batch_size = 1, 1
    all_file.append(os.path.join(dataset_dir, 'mvc_clothing/', file_pattern%(split_name,0)))
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
  
  image = tf.reshape(image, [height, width, 3])
  
  resized_image = tf.image.resize_images(images=image,
    size=[IMAGE_HEIGHT,IMAGE_WIDTH])
  
  mean = tf.constant(np.array(np.load(dataset_dir + 'mvc_clothing/mvc_mean.npy').transpose(1,2,0).reshape(224,224,3),dtype=np.float32))
  resized_image = resized_image - mean
  min_after_dequeue = 10000
  capacity = min_after_dequeue +3*batch_size

  images, labels = tf.train.shuffle_batch(
    [resized_image, label],
    batch_size = batch_size,
    capacity   = capacity,
    num_threads=1,
    min_after_dequeue= min_after_dequeue,
    seed=config.random_seed)

  return images, labels, SPLITS_TO_SIZES[split_name]