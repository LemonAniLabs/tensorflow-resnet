from convert import print_prob, load_image, checkpoint_fn, meta_fn
import tensorflow as tf

import resnet

import os

layers = 50

img = load_image("data/cat.jpg")

sess = tf.Session()

filename = checkpoint_fn(layers)
filename = os.path.realpath(filename)

if layers == 50:
    num_blocks = [3, 4, 6, 3]
elif layers == 101:
    num_blocks = [3, 4, 23, 3]
elif layers == 152:
    num_blocks = [3, 8, 36, 3]

with tf.device('/cpu:0'):
    images = tf.placeholder("float32", [None, 224, 224, 3], name="images")
    logits = resnet.inference(images,
                              is_training=False,
                              num_blocks=num_blocks,
                              preprocess=True,
                              bottleneck=True)
    prob = tf.nn.softmax(logits, name='prob')


saver = tf.train.Saver()
saver.restore(sess, filename)

graph = tf.get_default_graph()
prob_tensor = graph.get_tensor_by_name("prob:0")
for op in graph.get_operations():
    print op.name

print "graph restored"

batch = img.reshape((1, 224, 224, 3))

feed_dict = {images: batch}

prob = sess.run(prob_tensor, feed_dict=feed_dict)

print_prob(prob[0])
