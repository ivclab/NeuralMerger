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
import tensorflow as tf
import numpy as np
from config import setting
from datasets import dataset_factory
from nets import nets_factory
from weight_loader import weight_factory
from loss import loss_function
from tqdm import trange
from saver import saver
import os
from datetime import datetime
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def train_test(config):
    ###############
    ## Task Name ##
    ###############
    DATA_PATH       = config.data_path
    MERGER_NAME     = config.net
    task1,task2     = MERGER_NAME.split("_")

    ######################
    ## Check Log Folder ##
    ######################
    today = datetime.now()
    folder_name = config.net + today.strftime('_%m%d') + str(today.hour) + str(today.minute)
    if  os.path.exists("logs/"+folder_name):
        raise ValueError('Exist folder name %s' % folder_name)

    ###############################################
    ## Load Well-Trained Weight and Merged Model ##
    ###############################################
    shared_codebook, w1, b1, index1,outlayer1, w2, b2, index2,outlayer2 = weight_factory.get_weight(config, MERGER_NAME)

    ####################
    ## Select Network ##
    ####################
    merged_net      = nets_factory.get_network(MERGER_NAME,shared_codebook,w1, b1,index1,outlayer1, w2, b2, index2,outlayer2)

    ##############################
    ## Load Train and Test Data ##
    ##############################
    train_x1, train_y1, _           = dataset_factory.get_dataset(config,task1,'train',DATA_PATH)
    train_x2, train_y2, _           = dataset_factory.get_dataset(config,task2,'train',DATA_PATH)
    test_x1, test_y1, data_num3     = dataset_factory.get_dataset(config,task1,'test', DATA_PATH)
    test_x2, test_y2, data_num4     = dataset_factory.get_dataset(config,task2,'test', DATA_PATH)

    ##############################
    ## Train and Test Operation ##
    ##############################
    trainer, test1_op, test2_op, input_op, loss_op = loss_function(config,merged_net)

    # Initialize
    init_op         = tf.group( tf.global_variables_initializer(),
                            tf.local_variables_initializer())
    variables       = tf.trainable_variables()
    
    with tf.Session()  as sess:
        loss_log    = tf.summary.merge_all()
        if config.save_model == True:
            writer      = tf.summary.FileWriter("logs/"+folder_name,sess.graph)
        log_step    = config.log_step
        sess.run(init_op)
        coord       = tf.train.Coordinator()
        threads     = tf.train.start_queue_runners(coord=coord)

        #######################
        ## Start Calibration ##
        #######################
        max_step  = config.max_step
        for i in trange(max_step):
            x1, y1, x2, y2 = sess.run([train_x1, train_y1, train_x2, train_y2])
            sess.run(trainer,feed_dict={input_op['x1']:x1,
                                        input_op['y1']:y1,
                                        input_op['x2']:x2,
                                        input_op['y2']:y2})
            if (i% log_step == 0):
                loss,log = sess.run([loss_op,loss_log],feed_dict={ input_op['x1']:x1,
                                                    input_op['y1']:y1,
                                                    input_op['x2']:x2,
                                                    input_op['y2']:y2})
                print('[{:6d}/{}]  M1 loss:{:.3f}  M2 Loss:{:.3f}'.format(i,max_step,loss['M1_loss'],loss['M2_loss']))
                if config.save_model == True:
                    writer.add_summary(log,i)
            
            if (i%config.save_step == config.save_step-1 and config.save_model == True):
                saver(config,sess.run(variables),folder_name)


        print('-------- Testing Result --------')
        print('Merged Model           : {}'.format(config.net))
        #####################
        ## Model 1 Testing ##
        #####################
        for i in range(data_num3):
            x1, y1    = sess.run([test_x1, test_y1])
            accuracy1 = sess.run(test1_op,feed_dict={   input_op['x1']:x1,
                                                        input_op['y1']:y1,})

        #####################
        ## Model 2 Testing ##
        #####################
        for i in range(data_num4):
            x2, y2    = sess.run([test_x2, test_y2])
            accuracy2 = sess.run(test2_op,feed_dict={   input_op['x2']:x2,
                                                        input_op['y2']:y2,})

        print('Model1 Origin Accuracy : %.4f'%accuracy1['acc1_op_'])
        print('NeuralMerger  Accuracy : %.4f'%accuracy1['acc1_op'])
        print('Model2 Origin Accuracy : %.4f'%accuracy2['acc2_op_'])
        print('NeuralMerger  Accuracy : %.4f'%accuracy2['acc2_op'])

        
        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    config  = setting()
    train_test(config)
