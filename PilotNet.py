import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
#import scipy

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
UPDATE_OPS_COLLECTION = 'pilotnet_update_ops'  # must be grouped with training op
PILOTNET_VARIABLES = 'pilotnet_variables'

def weight_variable(shape, is_train=True):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, trainable=is_train)

def bias_variable(shape, is_train=True):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, trainable=is_train)

def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

#x = tf.placeholder(tf.float32, shape=[None, 66, 200, 3])
#y_ = tf.placeholder(tf.float32, shape=[None, 1])
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)
Steering_train = True
Throttle_train = True

def inference(x):
    x_image = x
    print('image shape : ' + str(x_image))

    # first convolutional layer
    with tf.variable_scope('Conv_1'):
        W_conv1 = weight_variable([5, 5, 3, 24])
        b_conv1 = bias_variable([24])
        x = conv2d(x_image, W_conv1, 2) + b_conv1
        x = _bn(x, True, is_training)
        h_conv1 = tf.nn.relu(x)
    print('hidden layer 1 : ' + str(h_conv1))
    
    # second convolutional layer
    with tf.variable_scope('Conv_2'):
        W_conv2 = weight_variable([5, 5, 24, 36])
        b_conv2 = bias_variable([36])
        x = conv2d(h_conv1, W_conv2, 2) + b_conv2
        x = _bn(x, True, is_training)
        h_conv2 = tf.nn.relu(x)
    print('hidden layer 2 : ' + str(h_conv2))
    
    # third convolutional layer
    with tf.variable_scope('Conv_3'):
        W_conv3 = weight_variable([5, 5, 36, 48])
        b_conv3 = bias_variable([48])
        x = conv2d(h_conv2, W_conv3, 2) + b_conv3
        x = _bn(x, True, is_training)
        h_conv3 = tf.nn.relu(x)
    print('hidden layer 3 : ' + str(h_conv3))
    
    # fourth convolutional layer
    with tf.variable_scope('Conv_4'):
        W_conv4 = weight_variable([3, 3, 48, 64])
        b_conv4 = bias_variable([64])
        x = conv2d(h_conv3, W_conv4, 1) + b_conv4
        x = _bn(x, True, is_training)
        h_conv4 = tf.nn.relu(x)
    
    print('hidden layer 4 : ' + str(h_conv4))
    
    # fifth convolutional layer
    with tf.variable_scope('Conv_5'):
        W_conv5 = weight_variable([3, 3, 64, 64])
        b_conv5 = bias_variable([64])
        x = conv2d(h_conv4, W_conv5, 1) + b_conv5
        x = _bn(x, True, is_training)
        h_conv5 = tf.nn.relu(x)
    
    print('h_conv5 : '+str(h_conv5))

    # Steering branch    
    with tf.variable_scope('Task_1'):
        # FCL 1
        with tf.variable_scope('Fc_1'):
            s_W_fc1 = weight_variable([1152, 1164], Steering_train)
            s_b_fc1 = bias_variable([1164], Steering_train)
        
            s_h_conv5_flat = tf.reshape(h_conv5, [-1, 1152])
            x=tf.matmul(s_h_conv5_flat, s_W_fc1) + s_b_fc1
            x = _bn(x, False, is_training)
            s_h_fc1 = tf.nn.relu(x)
        
        
        with tf.variable_scope('Fc_2'):
            # FCL 2
            s_W_fc2 = weight_variable([1164, 100], Steering_train)
            s_b_fc2 = bias_variable([100], Steering_train)
            x=tf.matmul(s_h_fc1, s_W_fc2) + s_b_fc2
            x = _bn(x, False, is_training)
            s_h_fc2 = tf.nn.relu(x)
            
            
        with tf.variable_scope('Fc_3'):
            # FCL 3
            s_W_fc3 = weight_variable([100, 50], Steering_train)
            s_b_fc3 = bias_variable([50], Steering_train)
            x=tf.matmul(s_h_fc2, s_W_fc3) + s_b_fc3 
            x = _bn(x, False, is_training)
            s_h_fc3 = tf.nn.relu(x)
            
            
        with tf.variable_scope('Fc_4'):
            # FCL 3
            s_W_fc4 = weight_variable([50, 10], Steering_train)
            s_b_fc4 = bias_variable([10], Steering_train)
            x=tf.matmul(s_h_fc3, s_W_fc4) + s_b_fc4 
            x = _bn(x, False, is_training)
            s_h_fc4 = tf.nn.relu(x)
            
            
        with tf.variable_scope('Action_Steering'):
            # Output-Steering
            s_W_fc_s = weight_variable([10, 1], Steering_train)
            s_b_fc_s = bias_variable([1], Steering_train)
        
        logits_steering = tf.matmul(s_h_fc4, s_W_fc_s) + s_b_fc_s

    with tf.variable_scope('Task_2'):
        # FCL 1
        with tf.variable_scope('Fc_1'):
            t_W_fc1 = weight_variable([1152, 1164], Throttle_train)
            t_b_fc1 = bias_variable([1164], Throttle_train)
        
            t_h_conv5_flat = tf.reshape(h_conv5, [-1, 1152])
            x = tf.matmul(t_h_conv5_flat, t_W_fc1) + t_b_fc1
            x = _bn(x, False, is_training)
            t_h_fc1 = tf.nn.relu(x)
        
        
        with tf.variable_scope('Fc_2'):
            # FCL 2
            t_W_fc2 = weight_variable([1164, 100], Throttle_train)
            t_b_fc2 = bias_variable([100], Throttle_train)
            x = tf.matmul(t_h_fc1, t_W_fc2) + t_b_fc2 
            x = _bn(x, False, is_training)
            t_h_fc2 = tf.nn.relu(x)
            
            
        with tf.variable_scope('Fc_3'):
            # FCL 3
            t_W_fc3 = weight_variable([100, 50], Throttle_train)
            t_b_fc3 = bias_variable([50], Throttle_train)
            x = tf.matmul(t_h_fc2, t_W_fc3) + t_b_fc3
            x = _bn(x, False, is_training)
            t_h_fc3 = tf.nn.relu(x)
            
            
        with tf.variable_scope('Fc_4'):
            # FCL 3
            t_W_fc4 = weight_variable([50, 10], Throttle_train)
            t_b_fc4 = bias_variable([10], Throttle_train)
            x = tf.matmul(t_h_fc3, t_W_fc4) + t_b_fc4 
            x = _bn(x, False, is_training)
            t_h_fc4 = tf.nn.relu(x)
            
            
        with tf.variable_scope('Action_Throttle'):
            # Output-Throttle
            t_W_fc_t = weight_variable([10, 1], Throttle_train)
            t_b_fc_t = bias_variable([1], Throttle_train)
    
        logits_throttle = tf.matmul(t_h_fc4, t_W_fc_t) + t_b_fc_t

    return logits_steering, logits_throttle

def _bn(x, is_conv, is_training):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]
    #axis = list(range(len(x_shape) - 1))
    if is_conv:
        axis = [0, 1, 2]
    else:
        axis = [0]
    beta = _get_variable('beta',
                         params_shape,
                         initializer=tf.zeros_initializer())
    gamma = _get_variable('gamma',
                          params_shape,
                          initializer=tf.ones_initializer())

    moving_mean = _get_variable('moving_mean',
                                params_shape,
                                initializer=tf.zeros_initializer(),
                                trainable=False)
    moving_variance = _get_variable('moving_variance',
                                    params_shape,
                                    initializer=tf.ones_initializer(),
                                    trainable=False)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    print('mean : ' + str(mean))
    print('variance : ' + str(variance))
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = tf.cond( is_training, lambda: (mean, variance), lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)

    return x

def _get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, PILOTNET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)

