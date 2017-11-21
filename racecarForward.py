import cv2 
from gtavResnet import _fc
import numpy as np
import tensorflow as tf
import gtavResnet
import os

def load_image(path, size=224):
    img = cv2.imread(path)
    res=cv2.resize(img,(size,size))
    res[0:60,:]=0
    return res

layers = 50

img = load_image("data/100.jpg",224)

sess = tf.Session()

filename = './pure-model/model.ckpt-101'

if layers == 50:
    num_blocks = [3, 4, 6, 3]
elif layers == 101:
    num_blocks = [3, 4, 23, 3]
elif layers == 152:
    num_blocks = [3, 8, 36, 3]

with tf.device('/gpu:0'):
    images = tf.placeholder("uint8", [None, 224, 224, 3], name="images")
    _x = gtavResnet.inference(images,
                              is_training=False,
                              num_blocks=num_blocks,
                              preprocess=False,
                              bottleneck=True)
    # post-net
    x = tf.reduce_mean(_x, axis=[1, 2], name="avg_pool")
    with tf.variable_scope('fc'):
        logits = _fc(x, num_units_out=6)
    #prob = tf.nn.softmax(logits, name='prob')


saver = tf.train.Saver()
saver.restore(sess, filename)

#graph = tf.get_default_graph()
#prob_tensor = graph.get_tensor_by_name("prob:0")
#for op in graph.get_operations():
#    print(op.name)

#print ("graph restored")

batch = img.reshape((1, 227, 227, 3))

feed_dict = {images: batch}

pred = sess.run(logits, feed_dict=feed_dict)
print(pred)
#print_prob(prob[0])
