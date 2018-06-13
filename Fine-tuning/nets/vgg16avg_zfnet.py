import tensorflow as tf
import numpy as np
def embedding_idx(index,k_num):
    index = np.copy(index)
    for i in range(index.shape[0]):
        index[i] = index[i] + k_num*i
    return index

class vgg16avg_zfnet:
    def __init__(self,c,w1,b1,i1,outlayer1,w2,b2,i2,outlayer2):
        with tf.variable_scope("vgg16avg_zfnet"):

            self.c   = []
            codebook = []
            for i in range(15):
                codebook.append(tf.Variable(c[i],dtype=tf.float32))
                self.c.append(tf.reshape(codebook[i],[c[i].shape[0]*c[i].shape[1],c[i].shape[2]]))
                
            self.w1 = []
            self.b1 = []
            for i in range(16):
                self.w1.append(tf.constant(w1[i],tf.float32))
                self.b1.append(tf.constant(b1[i],tf.float32))

            self.i1         =[]
            for i in range(len(i1)):
                self.i1.append(tf.constant(embedding_idx(i1[i],c[i].shape[1]),dtype=tf.int32))

            self.w2 = []
            self.b2 = []
            for i in range(8):
                self.w2.append(tf.constant(w2[i],tf.float32))
                self.b2.append(tf.constant(b2[i],tf.float32))

            self.shared2_index=[0,3,6,9,12,13,14]
            self.i2 = []
            for i in range(len(i2)):
                self.i2.append(tf.constant(embedding_idx(i2[i],c[self.shared2_index[i]].shape[1]),dtype=tf.int32))

            self.outlayer1 = outlayer1
            self.outlayer2 = outlayer2

    def model1_target(self,image):

        output=[]
        output.append(tf.nn.relu(conv2d(image,self.w1[0]) + self.b1[0]))
        output.append(tf.nn.relu(conv2d(output[0],self.w1[1]) + self.b1[1]))
        output.append(max_pool_2x2(output[1]))

        output.append(tf.nn.relu(conv2d(output[2],self.w1[2]) + self.b1[2]))
        output.append(tf.nn.relu(conv2d(output[3],self.w1[3]) + self.b1[3]))
        output.append(max_pool_2x2(output[4]))

        output.append(tf.nn.relu(conv2d(output[5],self.w1[4]) + self.b1[4]))
        output.append(tf.nn.relu(conv2d(output[6],self.w1[5]) + self.b1[5]))
        output.append(tf.nn.relu(conv2d(output[7],self.w1[6]) + self.b1[6]))
        output.append(max_pool_2x2(output[8]))

        output.append(tf.nn.relu(conv2d(output[9],self.w1[7]) + self.b1[7]))
        output.append(tf.nn.relu(conv2d(output[10],self.w1[8]) + self.b1[8]))
        output.append(tf.nn.relu(conv2d(output[11],self.w1[9]) + self.b1[9]))
        output.append(max_pool_2x2(output[12]))

        output.append(tf.nn.relu(conv2d(output[13],self.w1[10]) + self.b1[10]))
        output.append(tf.nn.relu(conv2d(output[14],self.w1[11]) + self.b1[11]))
        output.append(tf.nn.relu(conv2d(output[15],self.w1[12]) + self.b1[12]))
        output.append(avg_pool(output[16]))
        
        output.append(tf.nn.relu(tf.matmul(tf.contrib.layers.flatten(output[17]),self.w1[13]) +self.b1[13]))
        output.append(tf.nn.relu(tf.matmul(output[18],self.w1[14]) +self.b1[14]))
        output.append(tf.matmul(output[19],self.w1[15]) +self.b1[15])
        return output
    
    def model1(self,image):
        w1 = []
        for i in range(13):
            w1.append(tf.transpose(tf.reshape(tf.transpose(tf.nn.embedding_lookup(self.c[i],self.i1[i]),[0,2,1]),[self.w1[i].shape[2],self.w1[i].shape[0] ,self.w1[i].shape[1] ,self.w1[i].shape[3]]),(1,2,0,3)))
        for i in range(13,15):
            w1.append(tf.reshape(tf.transpose(tf.nn.embedding_lookup(self.c[i],self.i1[i]),[0,2,1]),[self.w1[i].shape[0],self.w1[i].shape[1]]))
        w1.append(tf.Variable(self.outlayer1,dtype=tf.float32))

        output=[]
        output.append(tf.nn.relu(conv2d(image,w1[0]) + self.b1[0]))
        output.append(tf.nn.relu(conv2d(output[0],w1[1]) + self.b1[1]))
        output.append(max_pool_2x2(output[1]))

        output.append(tf.nn.relu(conv2d(output[2],w1[2]) + self.b1[2]))
        output.append(tf.nn.relu(conv2d(output[3],w1[3]) + self.b1[3]))
        output.append(max_pool_2x2(output[4]))

        output.append(tf.nn.relu(conv2d(output[5],w1[4]) + self.b1[4]))
        output.append(tf.nn.relu(conv2d(output[6],w1[5]) + self.b1[5]))
        output.append(tf.nn.relu(conv2d(output[7],w1[6]) + self.b1[6]))
        output.append(max_pool_2x2(output[8]))

        output.append(tf.nn.relu(conv2d(output[9],w1[7]) + self.b1[7]))
        output.append(tf.nn.relu(conv2d(output[10],w1[8]) + self.b1[8]))
        output.append(tf.nn.relu(conv2d(output[11],w1[9]) + self.b1[9]))
        output.append(max_pool_2x2(output[12]))

        output.append(tf.nn.relu(conv2d(output[13],w1[10]) + self.b1[10]))
        output.append(tf.nn.relu(conv2d(output[14],w1[11]) + self.b1[11]))
        output.append(tf.nn.relu(conv2d(output[15],w1[12]) + self.b1[12]))
        output.append(avg_pool(output[16]))

        output.append(tf.nn.relu(tf.matmul(tf.contrib.layers.flatten(output[17]),w1[13]) +self.b1[13]))
        output.append(tf.nn.relu(tf.matmul(output[18],w1[14]) +self.b1[14]))
        output.append(tf.matmul(output[19],w1[15]) +self.b1[15])

        return output
        

    def model2_target(self,image):

        output=[]
        output.append(tf.nn.relu(conv2d_2(image,self.w2[0]) + self.b2[0]))
        output.append(max_pool_3(spatial_lrn(output[0],local_size=3,alpha=0.00005*9,beta=0.75)))

        output.append(tf.nn.relu(conv2d_2(output[1],self.w2[1]) + self.b2[1]))
        output.append(max_pool_3(spatial_lrn(output[2],local_size=3,alpha=0.00005*9,beta=0.75)))

        output.append(tf.nn.relu(conv2d(output[3],self.w2[2]) + self.b2[2]))

        output.append(tf.nn.relu(conv2d(output[4],self.w2[3]) + self.b2[3]))

        output.append(max_pool_v(tf.nn.relu(conv2d(output[5],self.w2[4]) + self.b2[4])))
        
        output.append(tf.nn.relu(tf.matmul(tf.contrib.layers.flatten(output[6]),self.w2[5]) +self.b2[5]))
        output.append(tf.nn.relu(tf.matmul(output[7],self.w2[6]) +self.b2[6]))
        output.append(tf.matmul(output[8],self.w2[7]) +self.b2[7])

        return output
    
    def model2(self,image):
        w2 = []
        for i in range(5):
            w2.append(tf.transpose(tf.reshape(tf.transpose(tf.nn.embedding_lookup(self.c[self.shared2_index[i]],self.i2[i]),[0,2,1]),[self.w2[i].shape[2],self.w2[i].shape[0] ,self.w2[i].shape[1] ,self.w2[i].shape[3]]),(1,2,0,3)))
        for i in range(5,7):
            w2.append(tf.reshape(tf.transpose(tf.nn.embedding_lookup(self.c[self.shared2_index[i]],self.i2[i]),[0,2,1]),[self.w2[i].shape[0],self.w2[i].shape[1]]))
        w2.append(tf.Variable(self.outlayer2,dtype=tf.float32))

        output=[]
        output.append(tf.nn.relu(conv2d_2(image,w2[0]) + self.b2[0]))
        output.append(max_pool_3(spatial_lrn(output[0],local_size=3,alpha=0.00005*9,beta=0.75)))

        output.append(tf.nn.relu(conv2d_2(output[1],w2[1]) + self.b2[1]))
        output.append(max_pool_3(spatial_lrn(output[2],local_size=3,alpha=0.00005*9,beta=0.75)))

        output.append(tf.nn.relu(conv2d(output[3],w2[2]) + self.b2[2]))

        output.append(tf.nn.relu(conv2d(output[4],w2[3]) + self.b2[3]))

        output.append(max_pool_v(tf.nn.relu(conv2d(output[5],w2[4]) + self.b2[4])))
        
        output.append(tf.nn.relu(tf.matmul(tf.contrib.layers.flatten(output[6]),w2[5]) +self.b2[5]))
        output.append(tf.nn.relu(tf.matmul(output[7],w2[6]) +self.b2[6]))
        output.append(tf.matmul(output[8],w2[7]) +self.b2[7])

        return output

    def get_imshape(self):
        imshape={
                'model1':[224,224,3],
                'model2':[227,227,3],
        }
        return imshape['model1'],imshape['model2']


def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_2(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def avg_pool(x):
	return tf.nn.avg_pool(x, ksize=[1, 14, 14, 1], strides=[1, 1, 1, 1], padding='VALID')

def max_pool_3(x):
	return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME')

def max_pool_v(x):
	return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='VALID')


def spatial_lrn(tensor, local_size=5, bias=1.0, alpha=1.0, beta=0.5):
    # Implement caffe Local Response Normalization(WITHIN_CHANNEL)
    squared = tf.square(tensor)
    in_channels = tensor.get_shape().as_list()[3]
    kernel = tf.constant(1.0, shape=[local_size, local_size, in_channels,1])
    squared_sum = tf.nn.depthwise_conv2d(squared, kernel, [1,1,1,1], padding='SAME')
    bias = tf.constant(bias, dtype=tf.float32)
    alpha = tf.constant(alpha, dtype=tf.float32)
    alpha = alpha/(local_size*local_size)
    beta = tf.constant(beta, dtype=tf.float32)

    return tensor / ((bias+alpha*squared_sum)**beta)