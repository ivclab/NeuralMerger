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

from datasets import sound20
from datasets import mnist_fashion
from datasets import mvc_clothing
from datasets import gender
import numpy as np
#import tensorflow as tf
#slim = tf.contrib.slim
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







