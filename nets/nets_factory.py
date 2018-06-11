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
import tensorflow as tf

import vgg16avg_zfnet
import lenetsound_lenetfashion
networks_map = {'lenetsound_lenetfashion': lenetsound_lenetfashion.lenetsound_lenetfashion,
                'vggclothing_zfgender': vgg16avg_zfnet.vgg16avg_zfnet,
               }



def get_network(name,shared_codebook,weight1,bias1,index1,outlayer1,weight2,bias2,index2,outlayer2):

  if name not in networks_map:
    raise ValueError('Name of network unknown %s' % name)

  return networks_map[name](
      shared_codebook,
      weight1,
      bias1,
      index1,
      outlayer1,
      weight2,
      bias2,
      index2,
      outlayer2,
      )