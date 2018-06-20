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
import sys
np.set_printoptions(threshold=np.nan)
from read_bin import load_bin,load_txt
def weight_loader(config):
    MERGER_NAME = config.net
    task1,task2 = MERGER_NAME.split("_")
    PATH        = config.weight_dir

    # Load Vgg Well-trained
    conv_num    = 2
    all_num     = 4
    sound_w     = []
    sound_b     = []
    PATH_M1     = PATH + config.net + '/' + task1 + '/'
    w1_name      = load_txt(PATH_M1 + "/weight.txt")
    b1_name      = load_txt(PATH_M1 + "/bias.txt")
    for i in range(conv_num):
        sound_w.append(load_bin(PATH_M1 + w1_name[i]).transpose(2,3,1,0))
        sound_b.append(load_bin(PATH_M1 + b1_name[i]))
    for i in range(conv_num,all_num):
        sound_w.append(load_bin(PATH_M1 + w1_name[i]).transpose(1,0))
        sound_b.append(load_bin(PATH_M1 + b1_name[i]))
    sound_w[2] = sound_w[2].reshape(64,8,8,1024).transpose(1,2,0,3).reshape(4096,1024)   #caffe(ch*h*w,out) -> tensorflow(h*w*ch,out)
    
    # Load ZF Well-trained
    conv_num    = 2
    all_num     = 4
    fashion_w   = []
    fashion_b   = []
    PATH_M2     = PATH + config.net + '/' + task2 + '/'
    w2_name      = load_txt(PATH_M2 + "/weight.txt")
    b2_name      = load_txt(PATH_M2 + "/bias.txt")
    for i in range(conv_num):
        fashion_w.append(load_bin(PATH_M2 + w2_name[i]).transpose(2,3,1,0))
        fashion_b.append(load_bin(PATH_M2 + b2_name[i]))
    for i in range(conv_num,all_num):
        fashion_w.append(load_bin(PATH_M2 + w2_name[i]).transpose(1,0))
        fashion_b.append(load_bin(PATH_M2 + b2_name[i]))
    fashion_w[2] = fashion_w[2].reshape(64,8,8,1024).transpose(1,2,0,3).reshape(4096,1024)   #caffe(ch*h*w,out) -> tensorflow(h*w*ch,out)
    
    # Load Merged Model
    conv_num    = 2
    all_num     = 3
    M_codebook  = []
    M1_index    = []
    M2_index    = []

    PATH_MERGE  = config.merger_dir
    c_name      = load_txt(PATH_MERGE + 'merged_codebook.txt')
    i1_name     = load_txt(PATH_MERGE + 'model1.txt')
    i2_name     = load_txt(PATH_MERGE + 'model2.txt')
    for i in range(conv_num):
        M_codebook.append(load_bin(PATH_MERGE + c_name[i]))

        M1_index.append(np.array(load_bin(PATH_MERGE + i1_name[i],data_type="uint8")-1,dtype=np.int32))
        M1_index[i] = M1_index[i].transpose(3,1,2,0).reshape([M1_index[i].shape[3],M1_index[i].shape[1]*M1_index[i].shape[2]*M1_index[i].shape[0]])

        M2_index.append(np.array(load_bin(PATH_MERGE + i2_name[i],data_type="uint8")-1,dtype=np.int32))
        M2_index[i] = M2_index[i].transpose(3,1,2,0).reshape([M2_index[i].shape[3],M2_index[i].shape[1]*M2_index[i].shape[2]*M2_index[i].shape[0]])

    for i in range(conv_num,all_num):
        M_codebook.append(load_bin(PATH_MERGE + c_name[i]))

        M1_index.append(np.array(load_bin(PATH_MERGE + i1_name[i],data_type="uint8")-1,dtype=np.int32))
        M1_index[i] = M1_index[i].transpose(1,0)

        M2_index.append(np.array(load_bin(PATH_MERGE + i2_name[i],data_type="uint8")-1,dtype=np.int32))
        M2_index[i] = M2_index[i].transpose(1,0)


    M1_output_layer = load_bin(PATH_MERGE + 'M1_outputlayer.bin').transpose(1,0)
    M2_output_layer = load_bin(PATH_MERGE + 'M2_outputlayer.bin').transpose(1,0)

    print('----- Codebook  Parameter  Setting -----')
    print('Codebook  subspace: [%3d, %3d, %3d]'%(M_codebook[0].shape[2],M_codebook[1].shape[2],M_codebook[2].shape[2]))
    print('Codeworkd numbers : [%3d, %3d, %3d]'%(M_codebook[0].shape[1],M_codebook[1].shape[1],M_codebook[2].shape[1]))
    print('Max   Iteration   : %d'%config.max_step)
    sys.stdout.write('Learning Rate     : ')
    print(config.lr_rate)
    print('Batch Size        : %d'%config.batch_size)

    return  M_codebook,sound_w,sound_b,M1_index,M1_output_layer,fashion_w,fashion_b,M2_index,M2_output_layer

