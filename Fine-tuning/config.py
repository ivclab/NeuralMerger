# Copyright 2017 The BEGAN-tensorflow Authors(Taehoon Kim). All Rights Reserved.
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
# ==============================================================================
import argparse

arg_lists = []
parser = argparse.ArgumentParser()
def str2bool(v):
    return v.lower() in ('true', '1')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

def setting():
    net_arg = add_argument_group('Calibration Setting')
    net_arg.add_argument('--merger_dir', type=str, default="./weight_loader/weight/lenetsound_lenetfashion/merge_ACCU/")
    net_arg.add_argument('--net', type=str, default="lenetsound_lenetfashion", choices=['lenetsound_lenetfashion', 'vggclothing_zfgender'])
    net_arg.add_argument('--save_model', type=str2bool, default="False")
    net_arg.add_argument('--max_step', type=int, default=20000)
    net_arg.add_argument('--decay_step', type=int, default=16000)
    net_arg.add_argument('--log_step', type=int, default=500)
    net_arg.add_argument('--save_step', type=int, default=5000)
    net_arg.add_argument('--random_seed', type=int, default=100)
    net_arg.add_argument('--batch_size', type=int, default=64)
    net_arg.add_argument('--lr_rate', type=float, default=0.0002)
    net_arg.add_argument('--data_split_num', type=int, default=5,choices=[1,2,3,4,5],
                                            help='data is divided into five groups. 5 means using 100% data')
    net_arg.add_argument('--data_path', type=str, default="./TFRecord/")
    net_arg.add_argument('--weight_dir', type=str, default="./weight_loader/weight/")
    config, unparsed = parser.parse_known_args()

    return config

