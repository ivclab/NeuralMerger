# MIT License
#
# Copyright (c) 2018 Image & Vision Computing Lab, Institute of Information Science, Academia Sinica
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
# ====================================================================================================
import numpy as np
def load_bin(data_path,data_type="float32"):

    if data_type != "float32" and data_type != "uint8":
        raise ValueError('Data_type should be float32 or uint8')
    dimension = np.fromfile(data_path, dtype=np.int32)
    dim = dimension[0]
    data_shape =[]
    for i in range(dim):
        data_shape.append(dimension[i+1])
    
    if data_type == "float32":
        num = dim+1
        data = np.fromfile(data_path, dtype=np.float32)
        data = data[num:]
    elif data_type is "uint8":
        num = (dim+1)*4
        data = np.fromfile(data_path, dtype=np.uint8)
        data = data[num:]

    data = data.reshape(data_shape)
    return data

def load_txt(path):
    with open (path, "r") as f:
        data = f.readlines()
    data = [x.strip() for x in data]
    return data

