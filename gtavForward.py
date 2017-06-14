import skimage.io
from gtavResnet import _fc
import numpy as np
import tensorflow as tf
import gtavResnet
import os

def load_image(path, size=224):
    img = skimage.io.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (size, size))
    return resized_img

layers = 50

img = load_image("data/100.jpg",227)

#sess = tf.Session()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

#filename = checkpoint_fn(layers)
#filename = os.path.realpath(filename)
filename = 'D:/Checkpoint/gtavResnet/model.ckpt-51001'

if layers == 50:
    num_blocks = [3, 4, 6, 3]
elif layers == 101:
    num_blocks = [3, 4, 23, 3]
elif layers == 152:
    num_blocks = [3, 8, 36, 3]

with tf.device('/gpu:0'):
    images = tf.placeholder("float32", [None, 227, 227, 3], name="images")
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
pred = sess.run(logits, feed_dict=feed_dict)
print(pred)

#print_prob(prob[0])
