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
    
