import tensorflow as tf



def loss_function(config,merged_net):
    ##################
    ## Define Input ##
    ##################
    im1_dim, im2_dim= merged_net.get_imshape()
    image1          = tf.placeholder(tf.float32,[None,im1_dim[0],im1_dim[1],im1_dim[2]])
    image2          = tf.placeholder(tf.float32,[None,im2_dim[0],im2_dim[1],im2_dim[2]])
    

    ####################
    ## Network Output ##
    ####################
    net1_out        = merged_net.model1(image1)
    net1_target     = merged_net.model1_target(image1)
    net2_out        = merged_net.model2(image2)
    net2_target     = merged_net.model2_target(image2)


    #################
    ## Define Loss ##
    #################
    # Model 1 loss
    label1          = tf.placeholder(tf.int64)
    layer_loss1     = 0
    for i in range(len(net1_out)):
        layer_loss1 += tf.reduce_mean(tf.abs(net1_out[i] - net1_target[i]))
    label_loss1     = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits (labels = label1,logits = net1_out[len(net1_out)-1]))

    # Model 2 loss
    label2          = tf.placeholder(tf.int64)
    layer_loss2     = 0
    for i in range(len(net2_out)):
        layer_loss2 += tf.reduce_mean(tf.abs(net2_out[i] - net2_target[i]))
    label_loss2     = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits (labels = label2,logits = net2_out[len(net2_out)-1]))

    gamma           = 10
    M1_loss         = layer_loss1 + gamma*label_loss1
    M2_loss         = layer_loss2 + gamma*label_loss2
    tf.summary.scalar('loss', M1_loss+M2_loss)

    ##############
    ## Training ##
    ##############
    global_step     = tf.Variable(0, trainable=False)
    add_global      = global_step.assign_add(1)
    initial_learning_rate = config.lr_rate
    learning_rate   = tf.train.exponential_decay( initial_learning_rate,
                                                global_step=global_step,
                                                decay_steps=config.decay_step ,decay_rate=0.1,staircase=True)

    with tf.control_dependencies([add_global]):
        trainer1    = tf.train.AdamOptimizer(learning_rate).minimize(M1_loss)
        trainer2    = tf.train.AdamOptimizer(learning_rate).minimize(M2_loss)


    #############
    ## Testing ##
    #############
    _, acc1_op     = tf.metrics.accuracy(label1, tf.argmax(tf.nn.softmax(net1_out[len(net1_out)-1]),1))
    _, acc1_op_    = tf.metrics.accuracy(label1, tf.argmax(tf.nn.softmax(net1_target[len(net1_out)-1]),1))
    _, acc2_op     = tf.metrics.accuracy(label2, tf.argmax(tf.nn.softmax(net2_out[len(net2_out)-1]),1))
    _, acc2_op_    = tf.metrics.accuracy(label2, tf.argmax(tf.nn.softmax(net2_target[len(net2_out)-1]),1))


    ###############
    ## Operation ##
    ###############
    input_op = {
        'x1':image1,
        'x2':image2,
        'y1':label1,
        'y2':label2,
    }

    trainer_op = {
        'trainer1' : trainer1,
        'trainer2' : trainer2,
    }

    loss_op = {
        'M1_loss':M1_loss,
        'M2_loss':M2_loss,
    }

    test1_op = {
        'acc1_op' :  acc1_op,
        'acc1_op_':  acc1_op_,
    }
    
    test2_op = {
        'acc2_op' :  acc2_op,
        'acc2_op_':  acc2_op_,
    }


    return trainer_op,test1_op,test2_op,input_op,loss_op