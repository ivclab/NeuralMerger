import tensorflow as tf
import numpy as np
import sys
import os


def embedding_idx(index,k_num):
    for i in range(index.shape[0]):
        index[i] = index[i] + k_num*i
    return index

class lenetsound_lenetfashion:
    def __init__(self,c,w1,b1,i1,outlayer1,w2,b2,i2,outlayer2):
        with tf.variable_scope("lenet_lenet"):
            self.c  = []
            codebook = []
            for i in range(3):
                codebook.append(tf.Variable(c[i],dtype=tf.float32))
                self.c.append(tf.reshape(codebook[i],[c[i].shape[0]*c[i].shape[1],c[i].shape[2]]))

            self.w1=[]
            self.b1=[]
            for i in range(4):
                self.w1.append(tf.constant(w1[i],tf.float32))
                self.b1.append(tf.constant(b1[i],tf.float32))

            self.i1         =[]
            for i in range(len(i1)):
                self.i1.append(tf.constant(embedding_idx(i1[i],c[i].shape[1]),dtype=tf.int32))

            self.w2=[]
            self.b2=[]
            for i in range(4):
                self.w2.append(tf.constant(w2[i],tf.float32))
                self.b2.append(tf.constant(b2[i],tf.float32))

            self.i2=[]
            for i in range(len(i2)):
                self.i2.append(tf.constant(embedding_idx(i2[i],c[i].shape[1]),dtype=tf.int32))

            self.outlayer1 = outlayer1
            self.outlayer2 = outlayer2


    def model1_target(self,image):
        output=[]
        output.append(max_pool_2x2(tf.nn.relu(conv2d(image, self.w1[0]) + self.b1[0])))
        output.append(max_pool_2x2(tf.nn.relu(conv2d(output[0], self.w1[1]) + self.b1[1])))
        output.append(tf.nn.relu(tf.matmul(tf.contrib.layers.flatten(output[1]), self.w1[2]) + self.b1[2]))
        output.append(tf.matmul(output[2], self.w1[3]) + self.b1[3])
        return output
    
    def model1(self,image):
        w1 = []
        for i in range(2):
            w1.append(tf.transpose(tf.reshape(tf.transpose(tf.nn.embedding_lookup(self.c[i],self.i1[i]),[0,2,1]),[self.w1[i].shape[2],self.w1[i].shape[0] ,self.w1[i].shape[1] ,self.w1[i].shape[3]]),(1,2,0,3)))
        for i in range(2,3):
            w1.append(tf.reshape(tf.transpose(tf.nn.embedding_lookup(self.c[i],self.i1[i]),[0,2,1]),[self.w1[i].shape[0],self.w1[i].shape[1]]))
        w1.append(tf.Variable(self.outlayer1,dtype=tf.float32))

        output=[]
        output.append(max_pool_2x2(tf.nn.relu(conv2d(image, w1[0]) + self.b1[0])))
        output.append(max_pool_2x2(tf.nn.relu(conv2d(output[0], w1[1]) + self.b1[1])))
        output.append(tf.nn.relu(tf.matmul(tf.contrib.layers.flatten(output[1]),w1[2]) + self.b1[2]))
        output.append(tf.nn.relu(tf.matmul(output[2], w1[3]) + self.b1[3]))

        return output


    def model2_target(self,image):
        output=[]
        output.append(max_pool_2x2(tf.nn.relu(conv2d(image, self.w2[0]) + self.b2[0])))
        output.append(max_pool_2x2(tf.nn.relu(conv2d(output[0], self.w2[1]) + self.b2[1])))
        output.append(tf.nn.relu(tf.matmul(tf.contrib.layers.flatten(output[1]), self.w2[2]) + self.b2[2]))
        output.append(tf.matmul(output[2], self.w2[3]) + self.b2[3])

        return output
    
    def model2(self,image):
        w2 = []
        for i in range(2):
            w2.append(tf.transpose(tf.reshape(tf.transpose(tf.nn.embedding_lookup(self.c[i],self.i2[i]),[0,2,1]),[self.w2[i].shape[2],self.w2[i].shape[0] ,self.w2[i].shape[1] ,self.w2[i].shape[3]]),(1,2,0,3)))
        for i in range(2,3):
            w2.append(tf.reshape(tf.transpose(tf.nn.embedding_lookup(self.c[i],self.i2[i]),[0,2,1]),[self.w2[i].shape[0],self.w2[i].shape[1]]))
        w2.append(tf.Variable(self.outlayer2,dtype=tf.float32))

        output=[]
        output.append(max_pool_2x2(tf.nn.relu(conv2d(image, w2[0]) + self.b2[0])))
        output.append(max_pool_2x2(tf.nn.relu(conv2d(output[0], w2[1]) + self.b2[1])))
        output.append(tf.nn.relu(tf.matmul(tf.contrib.layers.flatten(output[1]),w2[2]) + self.b2[2]))
        output.append(tf.nn.relu(tf.matmul(output[2], w2[3]) + self.b2[3]))

        return output
    def get_imshape(self):
        imshape={
                'model1':[32,32,1],
                'model2':[32,32,1],
        }
        return imshape['model1'],imshape['model2']
        
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

            

            





        





    