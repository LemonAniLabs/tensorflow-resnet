import tensorflow as tf
#import scipy

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

CNN_train = False
Steering_train = False
Throttle_train = True

def inference(x):
    x_image = x
    print('image shape : ' + str(x_image))
    #first convolutional layer
    W_conv1 = weight_variable([5, 5, 3, 24], CNN_train)
    b_conv1 = bias_variable([24], CNN_train)
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 2) + b_conv1)
    print('hidden layer 1 : ' + str(h_conv1))
    
    #second convolutional layer
    W_conv2 = weight_variable([5, 5, 24, 36], CNN_train)
    b_conv2 = bias_variable([36], CNN_train)
    
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 2) + b_conv2)
    print('hidden layer 2 : ' + str(h_conv2))
    
    #third convolutional layer
    W_conv3 = weight_variable([5, 5, 36, 48], CNN_train)
    b_conv3 = bias_variable([48], CNN_train)
    
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 2) + b_conv3)
    print('hidden layer 3 : ' + str(h_conv3))
    
    #fourth convolutional layer
    W_conv4 = weight_variable([3, 3, 48, 64], CNN_train)
    b_conv4 = bias_variable([64], CNN_train)
    
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4, 1) + b_conv4)
    print('hidden layer 4 : ' + str(h_conv4))
    
    #fifth convolutional layer
    W_conv5 = weight_variable([3, 3, 64, 64], CNN_train)
    b_conv5 = bias_variable([64], CNN_train)
    
    h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5, 1) + b_conv5)
    
    print('h_conv5 : '+str(h_conv5))
    
    # Steering
    #FCL 1
    W_fc1 = weight_variable([1152, 1164], Steering_train)
    b_fc1 = bias_variable([1164], Steering_train)
    
    h_conv5_flat = tf.reshape(h_conv5, [-1, 1152])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)
    
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    print('h_fc1_drop : '+str(h_fc1_drop))
    #FCL 2
    W_fc2 = weight_variable([1164, 100], Steering_train)
    b_fc2 = bias_variable([100], Steering_train)
    
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
    print('h_fc2_drop : '+str(h_fc2_drop))
    
    #FCL 3
    W_fc3 = weight_variable([100, 50], Steering_train)
    b_fc3 = bias_variable([50], Steering_train)
    
    h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
    
    h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)
    print('h_fc3_drop : '+str(h_fc3_drop))
    
    #FCL 3
    W_fc4 = weight_variable([50, 10], Steering_train)
    b_fc4 = bias_variable([10], Steering_train)
    
    h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)
    
    h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)
    print('h_fc4_drop : '+str(h_fc4_drop))
    
    #Output
    W_fc5 = weight_variable([10, 1], Steering_train)
    b_fc5 = bias_variable([1], Steering_train)
    
    logits = tf.matmul(h_fc4_drop, W_fc5) + b_fc5
    
    # Throttle
    #FCL 1
    T_W_fc1 = weight_variable([1152, 1164], Throttle_train)
    T_b_fc1 = bias_variable([1164], Throttle_train)
    
    T_h_conv5_flat = tf.reshape(h_conv5, [-1, 1152])
    T_h_fc1 = tf.nn.relu(tf.matmul(T_h_conv5_flat, T_W_fc1) + T_b_fc1)
    
    T_h_fc1_drop = tf.nn.dropout(T_h_fc1, keep_prob)
    
    print('h_fc1_drop : '+str(T_h_fc1_drop))
    #FCL 2
    T_W_fc2 = weight_variable([1164, 100], Throttle_train)
    T_b_fc2 = bias_variable([100], Throttle_train)
    
    T_h_fc2 = tf.nn.relu(tf.matmul(T_h_fc1_drop, T_W_fc2) + T_b_fc2)
    
    T_h_fc2_drop = tf.nn.dropout(T_h_fc2, keep_prob)
    print('h_fc2_drop : '+str(T_h_fc2_drop))
    
    #FCL 3
    T_W_fc3 = weight_variable([100, 50], Throttle_train)
    T_b_fc3 = bias_variable([50], Throttle_train)
    
    T_h_fc3 = tf.nn.relu(tf.matmul(T_h_fc2_drop, T_W_fc3) + T_b_fc3)
    
    T_h_fc3_drop = tf.nn.dropout(T_h_fc3, keep_prob)
    print('h_fc3_drop : '+str(T_h_fc3_drop))
    
    #FCL 3
    T_W_fc4 = weight_variable([50, 10], Throttle_train)
    T_b_fc4 = bias_variable([10], Throttle_train)
    
    T_h_fc4 = tf.nn.relu(tf.matmul(T_h_fc3_drop, T_W_fc4) + T_b_fc4)
    
    T_h_fc4_drop = tf.nn.dropout(T_h_fc4, keep_prob)
    print('h_fc4_drop : '+str(T_h_fc4_drop))
    
    #Output
    T_W_fc5 = weight_variable([10, 1], Throttle_train)
    T_b_fc5 = bias_variable([1], Throttle_train)
    
    T_logits = tf.matmul(T_h_fc4_drop, T_W_fc5) + T_b_fc5
    print("logits shape is ")
    print(logits)
    return T_logits
