import tensorflow as tf
import numpy as np
import image_processing as img_pro

def readTF(filename, is_training=False):

    filename_queue = tf.train.string_input_producer(filename)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'car_info/steering': tf.FixedLenFeature([], tf.float32),
                                           'car_info/throttle': tf.FixedLenFeature([], tf.float32),
                                           'car_info/speed':tf.FixedLenFeature([], tf.float32),
                                           'image/encoded' : tf.FixedLenFeature([], tf.string),
                                       })
    en_image = features['image/encoded']
    image = tf.image.decode_jpeg(en_image,3)
    image.set_shape([66, 200, 3])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
#    image = tf.image.crop_to_bounding_box(image, 100, 0, 66, 200)
#    image = tf.image.resize_images(image, size=[66,200])

    if is_training:
        image = img_pro.distort_color(image)
        print(image.shape)
    
    steering = features['car_info/steering']
    throttle = features['car_info/throttle']
    speed = features['car_info/speed']

#    return image, [steering, throttle, speed]
    return image, [steering]

def load_batch(dataset, batch_size=32, height=224, width=224, is_training=False):
    """Loads a single batch of data.
    
    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.
    
    Returns:
      images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
      images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
      labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
    """
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=32,
        common_queue_min=8)
    image_raw, label = data_provider.get(['image', 'label'])
    
    # Preprocess image for usage by Inception.
    image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=True)
    
    # Preprocess the image for display purposes.
    image_raw = tf.expand_dims(image_raw, 0)
    image_raw = tf.image.resize_images(image_raw, [height, width])
    image_raw = tf.squeeze(image_raw)

    # Batch it up.
    images, images_raw, labels = tf.train.batch(
          [image, image_raw, label],
          batch_size=batch_size,
          num_threads=1,
          capacity=2 * batch_size)
    
    return images, images_raw, labels
