import skimage.io  # bug. need to import this before tensorflow
import skimage.transform  # bug. need to import this before tensorflow
import gtavResnet
import tensorflow as tf
import time
import os
import sys
import re
import numpy as np
from termcolor import colored, cprint
from gtav.data_utils import get_dataset

from synset import *
from image_processing import image_preprocessing

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("data_path", "/home/lennon.lin/DataSet/DeepDrive/gtav_42", "Data path.")

class DataSet:
    def __init__(self, data_dir):
        self.subset = 'train'

    def data_files(self):
        tf_record_pattern = os.path.join(FLAGS.data_dir + '.tfrecords')
        return [ tf_record_pattern ]


def file_list(data_dir):
    dir_txt = data_dir + ".txt"
    print('dir_txt -> ', dir_txt)
    filenames = []
    with open(dir_txt, 'r') as f:
        for line in f:
            if line[0] == '.': continue
            line = line.rstrip()
            fn = os.path.join(data_dir, line)
            filenames.append(fn)
    return filenames


def load_data(data_dir):
    data = []
    i = 0

    print("listing files in", data_dir)
    start_time = time.time()
    files = file_list(data_dir)
    duration = time.time() - start_time
    print("took %f sec" % duration)

    for img_fn in files:
        f_name, ext = os.path.splitext(img_fn)
        ext = ext.split(' ')[0]
#        cprint('ext-> '+ str(ext), 'yellow')
#        cprint('f_name-> '+ str(f_name), 'yellow')
        if ext != '.JPEG': continue
#        cprint('pass JPEG check', 'red')
        #label_name = re.search(r'(n\d+)', f_name).group(1)
        label_name = os.path.split(os.path.split(f_name+ext)[0])[1]
        fn = os.path.join(data_dir, img_fn)
#        cprint('label_name -> ' + str(label_name), 'yellow')
#        cprint('filename -> ' + str(f_name+ext), 'red')
        label_index = synset_map[label_name]["index"]
#synset.synset_map['n02834397']['index'] = 443
        data.append({
            "filename": f_name+ext,
            "label_name": label_name,
            "label_index": label_index,
            "desc": synset[label_index],
        })

    return data


# Returns a numpy array of shape [size, size, 3]
def load_image(path, size):
    img = skimage.io.imread(path)

    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]

    img = skimage.transform.resize(crop_img, (size, size))

    # if it's a black and white photo, we need to change it to 3 channel
    # or raise an error if we're not allowing b&w (which we do during training)
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)

    assert img.shape == (size, size, 3)
    assert (0 <= img).all() and (img <= 1.0).all()

    return img

def distorted_inputs():
    data = load_data(FLAGS.data_dir)
    
    #cprint ('data -> ' + str(data), 'yellow')
    filenames = [ d['filename'] for d in data ]
    label_indexes = [ d['label_index'] for d in data ]
    #cprint ('filenames -> ' + str(filenames), 'yellow')
    filename, label_index = tf.train.slice_input_producer([filenames, label_indexes], shuffle=True)

    num_preprocess_threads = 4
    images_and_labels = []
    for thread_id in range(num_preprocess_threads):
        image_buffer = tf.read_file(filename)
        cprint('filename ->' + str(filename), 'yellow')
        bbox = []
        train = True
        cprint('image_buffer -> ' + str(image_buffer), 'yellow')
        image = image_preprocessing(image_buffer, bbox, train, thread_id)
        images_and_labels.append([image, label_index])

    images, label_index_batch = tf.train.batch_join(
        images_and_labels,
        batch_size=FLAGS.batch_size,
        capacity=2 * num_preprocess_threads * FLAGS.batch_size)

    height = FLAGS.input_size
    width = FLAGS.input_size
    depth = 3

    images = tf.cast(images, tf.float32)
    images = tf.reshape(images, shape=[FLAGS.batch_size, height, width, depth])

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, tf.reshape(label_index_batch, [FLAGS.batch_size])


def main(_):
    #dataset = DataSet(FLAGS.data_dir)
    #images, labels = distorted_inputs()
    gtavResnet.train()


if __name__ == '__main__':
    tf.app.run()
