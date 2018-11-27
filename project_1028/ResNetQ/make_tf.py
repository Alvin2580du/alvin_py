# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

import common

flags = tf.app.flags

flags.DEFINE_string('images_path', './data/train', 'path of images.')
flags.DEFINE_string('record_path', './train_record', 'path of tf record file.')
FLAGS = flags.FLAGS

INPUT_WIDTH = 224
INPUT_HEIGHT = 224
INPUT_CHANNEL = 3


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_image_binary(filename):
    image = Image.open(filename)
    image = np.asarray(image, np.uint8)
    shape = np.array(image.shape, np.int32)
    return shape, image.tobytes()


def read_and_decode(tfrecord_file, batch_size):
    filename_queue = tf.train.string_input_producer([tfrecord_file])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'image/label': tf.FixedLenFeature([], tf.int64),
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/channel': tf.FixedLenFeature([], tf.int64),
            'image/encoded': tf.FixedLenFeature([], tf.string),
        })

    h = tf.cast(img_features['image/height'], tf.int32)
    w = tf.cast(img_features['image/width'], tf.int32)
    c = tf.cast(img_features['image/channel'], tf.int32)

    image = tf.decode_raw(img_features['image/encoded'], tf.uint8)
    image = tf.reshape(image, [h, w, c])

    label = tf.cast(img_features['image/label'], tf.int32)
    label = tf.reshape(label, [1])

    image = tf.image.resize_images(image, (INPUT_HEIGHT, INPUT_WIDTH))
    image = tf.reshape(image, [INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL])
    # image, label = tf.train.batch([image, label],  batch_size= batch_size)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=8,
                                              capacity=2000)
    return image_batch, tf.reshape(label_batch, [batch_size])


def read_tfrecord2(tfrecord_file, batch_size):
    train_batch, train_label_batch = read_and_decode(tfrecord_file, batch_size)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        train_batch, train_batch_label = sess.run([train_batch, train_label_batch])
        coord.request_stop()
        coord.join(threads)
    return train_batch, train_batch_label


def generate_record(images_path, output_path):
    writer = tf.python_io.TFRecordWriter(output_path)
    files = common.get_files(images_path, '*.jpg')
    for f in files:
        print(os.path.basename(f))
        if f.lower().count('true') > 0:
            label = 1
        else:
            label = 0
        shape, binary_image = get_image_binary(f)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/label': int64_feature(label),
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channel': int64_feature(shape[2]),
            'image/encoded': bytes_feature(binary_image)
        }))
        writer.write(example.SerializeToString())
    writer.close()


def main():
    images_path = os.path.join(FLAGS.images_path)
    images_record_path = FLAGS.record_path
    generate_record(images_path, images_record_path)


def main_test():
    batch_size = 2
    train_batch, train_label_batch = read_tfrecord2(FLAGS.record_path, batch_size)
    print(train_batch.shape)
    print(train_label_batch)

    plt.figure()
    plt.imshow(train_batch[0, :, :, 2])
    plt.show()

    plt.figure()
    plt.imshow(train_batch[0, :, :, 1])
    plt.show()

    train_batch1 = train_batch[0, :, :, :]
    print(train_batch.shape)
    print(train_batch1.dtype)
    im = Image.fromarray(np.uint8(train_batch1))
    im.show()


if __name__ == '__main__':
    main()
    # main_test()
