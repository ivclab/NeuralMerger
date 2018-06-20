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
from datetime import datetime
import os
import shutil
from weight_loader import read_bin


def saver(config,shared_codebook,folder_name):
    NEW_PATH = "./logs/" + folder_name + '/'
    PATH_MERGE  = config.merger_dir
    
    # copy txtfile
    i1_name     = read_bin.load_txt(PATH_MERGE + 'model1.txt')
    i2_name     = read_bin.load_txt(PATH_MERGE + 'model2.txt')
    shutil.copy2(PATH_MERGE + 'model1.txt', NEW_PATH + 'model1.txt')
    shutil.copy2(PATH_MERGE + 'model2.txt', NEW_PATH + 'model2.txt')
    shutil.copy2(PATH_MERGE + 'merged_codebook.txt', NEW_PATH + 'merged_codebook.txt')

    # copy bias
    MERGER_NAME = config.net
    task1,task2 = MERGER_NAME.split("_")
    base_path  = config.weight_dir
    
    bias1_path  = base_path + config.net + '/' + task1 + '/'
    b1_name     = read_bin.load_txt(bias1_path + "/bias.txt")
    shutil.copy2(bias1_path + 'bias.txt', NEW_PATH + 'bias1.txt')
    for i in range(len(b1_name)):
        shutil.copy2(bias1_path + b1_name[i], NEW_PATH + b1_name[i])
    
    bias2_path  = base_path + config.net + '/' + task2 + '/'
    b2_name     = read_bin.load_txt(bias2_path + "/bias.txt")
    shutil.copy2(bias2_path + 'bias.txt', NEW_PATH + 'bias2.txt')
    for i in range(len(b2_name)):
        shutil.copy2(bias2_path + b2_name[i], NEW_PATH + b2_name[i])

    # copy asmt
    for i in range(len(i1_name)):
        shutil.copy2(PATH_MERGE + i1_name[i], NEW_PATH + i1_name[i])
    for i in range(len(i2_name)):
        shutil.copy2(PATH_MERGE + i2_name[i], NEW_PATH + i2_name[i])
    
    # save ctrd
    for i in range(len(shared_codebook)-2):
        file_path = NEW_PATH + "/" + config.net+'.ctrd.%02d'%(i+1)
        data_np   = shared_codebook[i]
        write_bin(file_path,data_np)
    
    # save outputlayer
    file_path = NEW_PATH + "/"  + 'M1_outputlayer'
    data_np   = shared_codebook[len(shared_codebook)-2].transpose(1,0)
    write_bin(file_path,data_np)

    file_path = NEW_PATH + "/"  + 'M2_outputlayer'
    data_np   = shared_codebook[len(shared_codebook)-1].transpose(1,0)
    write_bin(file_path,data_np)
	

def write_bin(file_path,data_np):
    file_path   = file_path + '.bin'
    dim_np      = np.array((data_np.shape),dtype=np.int32)
    dim_num     = np.int32(len(dim_np))
    
    output_file = open(file_path, 'wb')
    output_file.write(dim_num)
    output_file.write(np.array(dim_np))
    c_order = data_np.flags['C_CONTIGUOUS']
    if(c_order):
        output_file.write((data_np))
    else:
        data_np1 = data_np.copy(order='C')
        
        output_file.write((data_np1))
    output_file.close()
    return
    
