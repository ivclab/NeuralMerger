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
from read_bin import load_bin,load_txt
def weight_loader(config):
    MERGER_NAME = config.net
    task1,task2 = MERGER_NAME.split("_")
    PATH        = config.weight_dir

    # Load Vgg Well-trained
    conv_num    = 13
    all_num     = 16
    vgg_w =[]
    vgg_b =[]
    PATH_M1      = PATH + config.net + '/' + task1 + '/'
    w1_name      = load_txt(PATH_M1 + "weight.txt")
    b1_name      = load_txt(PATH_M1 + "bias.txt")
    for i in range(conv_num):
        vgg_w.append(load_bin(PATH_M1 + w1_name[i]).transpose(2,3,1,0))
        vgg_b.append(load_bin(PATH_M1 + b1_name[i]))
    for i in range(conv_num,all_num):
        vgg_w.append(load_bin(PATH_M1 + w1_name[i]).transpose(1,0))
        vgg_b.append(load_bin(PATH_M1 + b1_name[i]))
    
    # Load ZF Well-trained
    conv_num    = 5
    all_num     = 8
    zf_w        = []
    zf_b        = []
    PATH_M2     = PATH + config.net + '/' + task2 + '/'
    w2_name      = load_txt(PATH_M2 + "/weight.txt")
    b2_name      = load_txt(PATH_M2 + "/bias.txt")
    for i in range(conv_num):
        zf_w.append(load_bin(PATH_M2 + w2_name[i]).transpose(2,3,1,0))
        zf_b.append(load_bin(PATH_M2 + b2_name[i]))
    for i in range(conv_num,all_num):
        zf_w.append(load_bin(PATH_M2 + w2_name[i]).transpose(1,0))
        zf_b.append(load_bin(PATH_M2 + b2_name[i]))
    zf_w[5]     = zf_w[5].reshape(256,6,6,4096).transpose(1,2,0,3).reshape(9216,4096)   #caffe(ch*h*w,out) -> tensorflow(h*w*ch,out)
    
    # Load Merged Model
    conv_num    = 13
    all_num     = 15
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

    for i in range(conv_num,all_num):
        M_codebook.append(load_bin(PATH_MERGE + c_name[i]))

        M1_index.append(np.array(load_bin(PATH_MERGE + i1_name[i],data_type="uint8")-1,dtype=np.int32))
        M1_index[i] = M1_index[i].transpose(1,0)

    conv_num    = 5
    all_num     = 7
    for i in range(conv_num):
        M2_index.append(np.array(load_bin(PATH_MERGE + i2_name[i],data_type="uint8")-1,dtype=np.int32))
        M2_index[i] = M2_index[i].transpose(3,1,2,0).reshape([M2_index[i].shape[3],M2_index[i].shape[1]*M2_index[i].shape[2]*M2_index[i].shape[0]])

    for i in range(conv_num,all_num):
        M2_index.append(np.array(load_bin(PATH_MERGE + i2_name[i],data_type="uint8")-1,dtype=np.int32))
        M2_index[i] = M2_index[i].transpose(1,0)

    print_r = []
    print_c = []
    shared  = [0,3,6,9,12,13,14]    # VGG&ZF Shared Layer in paper
    for i in range(15):
        print_r.append(M_codebook[i].shape[2])
        print_c.append(M_codebook[i].shape[1])
    print('----- Codebook  Parameter  Setting -----')
    sys.stdout.write('Codebook Subspace : ')
    print(print_r)
    sys.stdout.write('Codeworkd Numbers : ')
    print(print_c)
    sys.stdout.write('Shared Layers     : ')
    print(shared)
    print('Max   Iteration   : %d'%config.max_step)
    print('Learning Rate     : %.4f'%config.lr_rate)
    print('Batch Size        : %d'%config.batch_size)

    M1_output_layer = load_bin(PATH_MERGE + 'M1_outputlayer.bin').transpose(1,0)
    M2_output_layer = load_bin(PATH_MERGE + 'M2_outputlayer.bin').transpose(1,0)

    return M_codebook, vgg_w, vgg_b, M1_index,M1_output_layer, zf_w, zf_b, M2_index, M2_output_layer


