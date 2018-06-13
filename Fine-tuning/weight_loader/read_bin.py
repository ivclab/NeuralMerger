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

