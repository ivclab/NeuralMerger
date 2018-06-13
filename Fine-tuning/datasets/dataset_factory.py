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

import sound20
import mnist_fashion
import mvc_clothing
import gender
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
datasets_map = {
    'lenetsound': sound20,
    'lenetfashion': mnist_fashion,
    'vggclothing': mvc_clothing,
    'zfgender': gender,
}


def get_dataset(config,name, split_name, dataset_dir, file_pattern=None, reader=None):
  """Given a dataset name and a split_name returns a Dataset.
  Args:
    name: String, the name of the dataset.
    split_name: A train/test split name.
    dataset_dir: The directory where the dataset files are stored.
    file_pattern: The file pattern to use for matching the dataset source files.
    reader: The subclass of tf.ReaderBase. If left as `None`, then the default
      reader defined by each dataset is used.
  Returns:
    A `Dataset` class.
  Raises:
    ValueError: If the dataset `name` is unknown.
  """
  if name not in datasets_map:
    raise ValueError('Name of dataset unknown %s' % name)
  return datasets_map[name].get_split(
      config,
      split_name,
      dataset_dir,
      file_pattern,
      reader,
      )







