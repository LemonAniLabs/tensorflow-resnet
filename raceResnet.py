import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages
from gtav.data_utils import get_dataset
from gtav.tf_data_utils import readTF

import datetime
import numpy as np
import os
import sys
import time
from termcolor import colored, cprint

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1
FC_WEIGHT_DECAY = 0.00004
FC_WEIGHT_STDDEV = 0.01
MOMENTUM = 0.9
RESNET_VARIABLES = 'resnet_variables'
UPDATE_OPS_COLLECTION = 'resnet_update_ops'  # must be grouped with training op
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838, ]

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './train_log_racecar_2617',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './pure-model',
                           """Directory stored pretrain model""")
tf.app.flags.DEFINE_float('learning_rate', 0.001, "learning rate.")
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')
tf.app.flags.DEFINE_integer('batch_size', 64, "batch size")
tf.app.flags.DEFINE_integer('input_size', 224, "input image size")
tf.app.flags.DEFINE_boolean('continue', False,
                            'resume from latest saved state')

def guarantee_initialized_variables(session, list_of_variables = None):
    if list_of_variables is None:
        list_of_variables = tf.all_variables()
    #uninitialized_variables = list(tf.global_variables(name) for name in
    #                               session.run(tf.report_uninitialized_variables(list_of_variables)))
    #session.run(tf.initialize_variables(uninitialized_variables))
    print(session.run(tf.report_uninitialized_variables(list_of_variables)))
    #return unintialized_variables

def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    print([str(i.name) for i in not_initialized_vars]) # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

model_pareto=[]
def train():
    
#    labels = tf.placeholder("float", [None, 3], name="labels")
#    images = tf.placeholder("float",
#                            [None, FLAGS.input_size, FLAGS.input_size, 3],
#                            name="images")

    tfrecord_train = ['/mnt/s1/kr7830/Data/TX2/tfRecords/train/26XX/MiniCar_train_2617_1.tfrecords',
    '/mnt/s1/kr7830/Data/TX2/tfRecords/train/26XX/MiniCar_train_2617_3.tfrecords',
    '/mnt/s1/kr7830/Data/TX2/tfRecords/train/26XX/MiniCar_train_262223_1.tfrecords']
    tfrecord_val = '/mnt/s1/kr7830/Data/TX2/tfRecords/validation/MiniCar_val_4.tfrecords'
    img, targets = readTF(tfrecord_train, is_training=True)
    images, labels = tf.train.shuffle_batch([img, targets],
                                        batch_size=FLAGS.batch_size, capacity=2000,
                                        min_after_dequeue=1000
                                                   )
#    eval_img, eval_targets = readTF([tfrecord_val])
#    eval_images, eval_labels = tf.train.shuffle_batch([eval_img, eval_targets],
#                                                    batch_size=FLAGS.batch_size, capacity=2000,
#                                                    min_after_dequeue=1000
#                                                   )
    """ TODO: IMAGE PREPROCESSING """
    tf.summary.image('images', images)

    with tf.variable_scope("") as vs:
        logits = inference(images,
                           num_classes=1,
                           is_training=True,
                           preprocess=False,
                           bottleneck=True,
                           num_blocks=[3, 4, 6, 3])

#        vs.reuse_variables()
#
#        eval_logits = inference(eval_images,
#                                num_classes=1,
#                                is_training=False,
#                                preprocess=False,
#                                bottleneck=True,
#                                num_blocks=[3, 4, 6, 3])

    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    

    cprint('Construct Network Succes', 'yellow')
    cprint('Label shape' + str(labels), 'red')
    cprint('Logits shape' + str(logits), 'red')
    
    total_loss, l2_norn, lr_loss = new_loss(logits, labels, tf.to_float(tf.shape(images)[0]))

    # loss_avg
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([total_loss]))
    loss_avg = ema.average(total_loss)
    tf.summary.scalar('loss_avg', loss_avg)
    
    #Define Learning Rate Decay Function
    num_samples_per_epoch = 217872
    decay_steps = int(num_samples_per_epoch / FLAGS.batch_size * FLAGS.num_epochs_per_decay)
    learning_rate_ = tf.train.exponential_decay(FLAGS.learning_rate,
                                      global_step,
                                      decay_steps,
                                      FLAGS.learning_rate_decay_factor,
                                      staircase=True,
                                      name='exponential_decay_learning_rate')
    tf.summary.scalar('learning_rate', learning_rate_)


#    opt = tf.train.MomentumOptimizer(learning_rate_, MOMENTUM)
    opt = tf.train.AdamOptimizer(learning_rate_)
    grads = opt.compute_gradients(total_loss)
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)
    
    #SUMMARY
    summary_op = tf.summary.merge_all()

    init_op = tf.initialize_all_variables()
#    pretrained_var_map = {}
#    for v in tf.trainable_variables():
#        found = False
#        for bad_layer in ["fc"]:
#            if bad_layer in v.name:
#                found = True
#                cprint('Find layer ->' + bad_layer, 'red')
#        if found:
#            continue
#
#        pretrained_var_map[v.op.name[:]] = v
#        cprint(v.op.name[:],'yellow')
#        print(v.op.name, v.get_shape())
#    resnet_saver = tf.train.Saver(pretrained_var_map)
    def init_fn(ses):
        print("Initializing parameters.")
        ses.run(init_op)
#        pre_checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'ResNet-L50.ckpt')
#        resnet_saver.restore(ses, pre_checkpoint_path)

    saver = tf.train.Saver()
    sv = tf.train.Supervisor(is_chief=True,
                             logdir=FLAGS.train_dir + "/train",
                             summary_op=None,  # Automatic summaries don"t work with placeholders.
                             saver=saver,
                             global_step=global_step,
                             save_summaries_secs=30,
                             save_model_secs=60,
                             init_op=None,
                             init_fn=init_fn)


    config=tf.ConfigProto(log_device_placement=False)

################ HDF5 #########################################################
#    train_dataset = get_dataset(FLAGS.data_path)
#    train_data_provider = train_dataset.iterate_forever(FLAGS.batch_size)
#    eval_dataset = get_dataset(FLAGS.data_path, train=False)
###############################################################################



    with sv.managed_session(config=config) as sess, sess.as_default():
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        while True:
            start_time = time.time()

#            images_, targets_ = img_batch, label_batch 

            step = sess.run(global_step)
            i = [train_op, total_loss, l2_norn, lr_loss]

            write_summary = step % 100 and step > 1
            if write_summary:
                i.append(summary_op)

            #o = sess.run(i, {
            #    images: images_,
            #    labels: targets_,
            #})
            o = sess.run(i)

            loss_value = o[1]

            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

#            # Do evaluation
#            if step>0 and step % 100 ==0:
#                losses = []
#                for i in range(FLAGS.batch_size):
#                    preds, targets = sess.run([eval_logits, eval_labels])
#                    losses += [np.square(targets - preds)]
#                losses = np.concatenate(losses)
#                print("Eval: shape: {}".format(losses.shape))
#                summary = tf.Summary()
##                summary = summary_op
#                summary.value.add(tag="eval/loss", simple_value=float(0.5 * losses.sum() / losses.shape[0]))
##                names = ["steering", "throttle, ""speed"]
#                names = ["steering"]
#                for i in range(len(names)):
#                    summary.value.add(tag="eval/{}".format(names[i]), simple_value=float(0.5 * losses[:, i].mean()))
#                print(summary)
#                summary_writer.add_summary(summary, step)
#                #summary_writer.flush()


            if step % 100 == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                format_str = ('step %d, loss = %.6f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (step, loss_value, examples_per_sec, duration))

            if write_summary:
                summary_str = o[4]
                summary_writer.add_summary(summary_str, step)

            v_l2norn, v_lrloss = o[2], o[3]

            # Save the model checkpoint periodically.
#            if step > 1 and step % 100 == 0:
#                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
#                saver.save(sess, checkpoint_path, global_step=global_step)
        coord.request_stop()
        coord.join(threads)

#def keepModel(new_l2norn, new_lrloss):
#    
#    if len(model_pareto)==0:
#        model_pareto.append((global_step, new_l2norn, new_lrloss))
#        return
#    else:
#        keep_model=[]
#        for aModel in model_pareto:
#            old_step, old_l2norn, old_lrloss = aModel
#            if new_lrloss >= old_lrloss and  new_l2norn >= old_l2norn:
#                keep_model.append(aModel)
#
#
#        for aModel in model_pareto:
#            old_step, old_l2norn, old_lrloss = aModel
#            if new_lrloss < old_lrloss and  new_l2norn < old_l2norn:
#


def inference(x, is_training,
              num_classes=1000,
              num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
              preprocess=True,
              bottleneck=True):
    # if preprocess is True, input should be RGB [0,1], otherwise BGR with mean
    # subtracted
    if preprocess:
        x = _imagenet_preprocess(x)
        tf.summary.image('images', x)

    is_training = tf.convert_to_tensor(is_training,
                                       dtype='bool',
                                       name='is_training')

    with tf.variable_scope('scale1'):
        x = _conv(x, 64, ksize=7, stride=2)
        x = _bn(x, is_training)
        x = _relu(x)

    with tf.variable_scope('scale2'):
        x = _max_pool(x, ksize=3, stride=2)
        x = stack(x, num_blocks[0], 64, bottleneck, is_training, stride=1)

    with tf.variable_scope('scale3'):
        x = stack(x, num_blocks[1], 128, bottleneck, is_training, stride=2)

    with tf.variable_scope('scale4'):
        x = stack(x, num_blocks[2], 256, bottleneck, is_training, stride=2)

    with tf.variable_scope('scale5'):
        x = stack(x, num_blocks[3], 512, bottleneck, is_training, stride=2)

    # post-net
    x = tf.reduce_mean(x, axis=[1, 2], name="avg_pool")
    with tf.variable_scope('fc'):
        logits = _fc(x, num_units_out=1)

    return logits


# This is what they use for CIFAR-10 and 100.
# See Section 4.2 in http://arxiv.org/abs/1512.03385
def inference_small(x,
                    is_training,
                    num_classes=10,
                    num_blocks=3, # 6n+2 total weight layers will be used.
                    preprocess=True):
    # if preprocess is True, input should be RGB [0,1], otherwise BGR with mean
    # subtracted
    if preprocess:
        x = _imagenet_preprocess(x)

    bottleneck = False
    is_training = tf.convert_to_tensor(is_training,
                                       dtype='bool',
                                       name='is_training')

    with tf.variable_scope('scale1'):
        x = _conv(x, 16, ksize=3, stride=1)
        x = _bn(x, is_training)
        x = _relu(x)

        x = stack(x, num_blocks, 16, bottleneck, is_training, stride=1)

    with tf.variable_scope('scale2'):
        x = stack(x, num_blocks, 32, bottleneck, is_training, stride=2)

    with tf.variable_scope('scale3'):
        x = stack(x, num_blocks, 64, bottleneck, is_training, stride=2)

    # post-net
    x = tf.reduce_mean(x, axis=[1, 2], name="avg_pool")
    with tf.variable_scope('fc'):
        logits = _fc(x, num_units_out=num_classes)

    return logits


def _imagenet_preprocess(rgb):
    """Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb * 255.0)
    bgr = tf.concat(axis=3, values=[blue, green, red])
    bgr -= IMAGENET_MEAN_BGR
    return bgr


def new_loss(logits, labels, x_shape):

    l2_norm = tf.global_norm(tf.trainable_variables())
    loss_ = 0.5 * tf.reduce_sum(tf.square(logits - labels)) / x_shape
    tf.summary.scalar("model/loss", loss_)
    tf.summary.scalar("model/l2_norm", l2_norm)
    total_loss = loss_ + 5e-8 * l2_norm

    tf.summary.scalar('model/total_loss', total_loss)

    return total_loss, l2_norm, loss_

def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
 
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss_ = tf.add_n([cross_entropy_mean] + regularization_losses)
    tf.summary.scalar('loss', loss_)

    return loss_


def stack(x, num_blocks, filters_internal, bottleneck, is_training, stride):
    for n in range(num_blocks):
        s = stride if n == 0 else 1
        with tf.variable_scope('block%d' % (n + 1)):
            x = block(x,
                      filters_internal,
                      bottleneck=bottleneck,
                      is_training=is_training,
                      stride=s)
    return x


def block(x, filters_internal, is_training, stride, bottleneck):
    filters_in = x.get_shape()[-1]

    # Note: filters_out isn't how many filters are outputed. 
    # That is the case when bottleneck=False but when bottleneck is 
    # True, filters_internal*4 filters are outputted. filters_internal is how many filters
    # the 3x3 convs output internally.
    if bottleneck:
        filters_out = 4 * filters_internal
    else:
        filters_out = filters_internal

    shortcut = x  # branch 1

    if bottleneck:
        with tf.variable_scope('a'):
            x = _conv(x, filters_internal, ksize=1, stride=stride)
            x = _bn(x, is_training)
            x = _relu(x)

        with tf.variable_scope('b'):
            x = _conv(x, filters_internal, ksize=3, stride=1)
            x = _bn(x, is_training)
            x = _relu(x)

        with tf.variable_scope('c'):
            x = _conv(x, filters_out, ksize=1, stride=1)
            x = _bn(x, is_training)
    else:
        with tf.variable_scope('A'):
            x = _conv(x, filters_internal, ksize=3, stride=stride)
            x = _bn(x, is_training)
            x = _relu(x)

        with tf.variable_scope('B'):
            x = _conv(x, filters_out, ksize=3, stride=1)
            x = _bn(x, is_training)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or stride != 1:
            shortcut = _conv(shortcut, filters_out, ksize=1, stride=stride)
            shortcut = _bn(shortcut, is_training)

    return _relu(x + shortcut)


def _relu(x):
    return tf.nn.relu(x)


def _bn(x, is_training):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]
    axis = list(range(len(x_shape) - 1))

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
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(
        moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

    mean, variance = control_flow_ops.cond(
        is_training, lambda: (mean, variance),
        lambda: (moving_mean, moving_variance))

    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    #x.set_shape(inputs.get_shape()) ??

    return x


def _fc(x, num_units_out):
    num_units_in = x.get_shape()[1]
    weights_initializer = tf.truncated_normal_initializer(
        stddev=FC_WEIGHT_STDDEV)

    weights = _get_variable('weights',
                            shape=[num_units_in, num_units_out],
                            initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV)
    biases = _get_variable('biases',
                           shape=[num_units_out],
                           initializer=tf.zeros_initializer())
    x = tf.nn.xw_plus_b(x, weights, biases)
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
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, RESNET_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)


def _conv(x, filters_out, ksize=3, stride=1):
    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV)
    weights = _get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=CONV_WEIGHT_DECAY)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')


def _max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')
