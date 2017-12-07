import tensorflow as tf
#import scipy

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

#x = tf.placeholder(tf.float32, shape=[None, 66, 200, 3])
#y_ = tf.placeholder(tf.float32, shape=[None, 1])
keep_prob = tf.placeholder(tf.float32)

def inference(x):
    x_image = x
    print('image shape : ' + str(x_image))

    # first convolutional layer
    with tf.name_scope('Conv_1'):
        W_conv1 = weight_variable([5, 5, 3, 24])
        b_conv1 = bias_variable([24])
    
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 2) + b_conv1)
    print('hidden layer 1 : ' + str(h_conv1))
    
    # second convolutional layer
    with tf.name_scope('Conv_2'):
        W_conv2 = weight_variable([5, 5, 24, 36])
        b_conv2 = bias_variable([36])
    
        h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)
    print('hidden layer 2 : ' + str(h_conv2))
    
    # third convolutional layer
    with tf.name_scope('Conv_3'):
        W_conv3 = weight_variable([5, 5, 36, 48])
        b_conv3 = bias_variable([48])
    
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 2) + b_conv3)
    print('hidden layer 3 : ' + str(h_conv3))
    
    # fourth convolutional layer
    with tf.name_scope('Conv_4'):
        W_conv4 = weight_variable([3, 3, 48, 64])
        b_conv4 = bias_variable([64])
    
        h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, 1) + b_conv4)
    print('hidden layer 4 : ' + str(h_conv4))
    
    # fifth convolutional layer
    with tf.name_scope('Conv_5'):
        W_conv5 = weight_variable([3, 3, 64, 64])
        b_conv5 = bias_variable([64])
    
        h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, 1) + b_conv5)
    
    print('h_conv5 : '+str(h_conv5))

    # Steering branch    
    with tf.name_scope('Task_1'):
        # FCL 1
        with tf.name_scope('Fc_1'):
            W_fc1 = weight_variable([1152, 1164])
            b_fc1 = bias_variable([1164])
        
            h_conv5_flat = tf.reshape(h_conv5, [-1, 1152])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)
        
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        
        with tf.name_scope('Fc_2'):
            # FCL 2
            W_fc2 = weight_variable([1164, 100])
            b_fc2 = bias_variable([100])
            
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
            
            h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
            
        with tf.name_scope('Fc_3'):
            # FCL 3
            W_fc3 = weight_variable([100, 50])
            b_fc3 = bias_variable([50])
            
            h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
            
            h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)
            
        with tf.name_scope('Fc_4'):
            # FCL 3
            W_fc4 = weight_variable([50, 10])
            b_fc4 = bias_variable([10])
            
            h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)
            
            h_fc4_s_drop = tf.nn.dropout(h_fc4, keep_prob)
            
        with tf.name_scope('Action_Steering'):
            # Output-Steering
            W_fc_s = weight_variable([10, 1])
            b_fc_s = bias_variable([1])

    with tf.name_scope('Task_2'):
        # FCL 1
        with tf.name_scope('Fc_1'):
            W_fc1 = weight_variable([1152, 1164])
            b_fc1 = bias_variable([1164])
        
            h_conv5_flat = tf.reshape(h_conv5, [-1, 1152])
            h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)
        
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        
        with tf.name_scope('Fc_2'):
            # FCL 2
            W_fc2 = weight_variable([1164, 100])
            b_fc2 = bias_variable([100])
            
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
            
            h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
            
        with tf.name_scope('Fc_3'):
            # FCL 3
            W_fc3 = weight_variable([100, 50])
            b_fc3 = bias_variable([50])
            
            h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
            
            h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)
            
        with tf.name_scope('Fc_4'):
            # FCL 3
            W_fc4 = weight_variable([50, 10])
            b_fc4 = bias_variable([10])
            
            h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)
            
            h_fc4_t_drop = tf.nn.dropout(h_fc4, keep_prob)
            
        with tf.name_scope('Action_Throttle'):
            # Output-Throttle
            W_fc_t = weight_variable([10, 1])
            b_fc_t = bias_variable([1])
    
    logits_steering = tf.matmul(h_fc4_s_drop, W_fc_s) + b_fc_s
    logits_throttle = tf.matmul(h_fc4_t_drop, W_fc_t) + b_fc_t

    return logits_steering, logits_throttle
