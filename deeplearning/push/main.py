# -*- coding: UTF-8 -*-

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
# 0 - debug
# 1 - info (still a LOT of outputs)
# 2 - warnings
# 3 - errors

import tensorflow as tf
import model
import preprocess

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('train_record', './train_record', 'path of training record file.')
flags.DEFINE_string('val_record', './val_record', 'path of val record file.')
flags.DEFINE_string('resnet50_model_path', None, 'path of pre-trained ResNet-50 model.')
flags.DEFINE_string('checkpoint_prefix', None, 'path of checkpoint')
flags.DEFINE_string('log_dir', './logs', 'path of log .')
FLAGS = flags.FLAGS

CLASS_NUM = 2
INPUT_WIDTH = 224
INPUT_HEIGHT = 224
INPUT_CHANNEL = 3
BATCH_SIZE = 8


def read_and_decode(tfrecord_file, batch_size, is_training=False):
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

    label = tf.cast(img_features['image/label'], tf.int64)
    label = tf.reshape(label, [1])

    image = tf.image.resize_images(image, (INPUT_HEIGHT, INPUT_WIDTH))
    image = tf.reshape(image, [INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL])

    # data augmentation
    image = preprocess.preprocess_image(image, INPUT_HEIGHT, INPUT_WIDTH, is_training=is_training)

    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=batch_size,
                                                      num_threads=4,
                                                      min_after_dequeue=100,
                                                      capacity=200)
    return image_batch, tf.reshape(label_batch, [batch_size])


def main(_):
    inputs, labels = read_and_decode(FLAGS.train_record, BATCH_SIZE, is_training=True)

    cls_model = model.Model(is_training=True, num_classes=CLASS_NUM)
    prediction_dict = cls_model.predict(inputs)
    loss_dict = cls_model.loss(prediction_dict, labels)
    loss = loss_dict['loss']
    postprocessed_dict = cls_model.postprocess(prediction_dict)
    acc = cls_model.accuracy(postprocessed_dict, labels)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', acc)

    # optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.99)
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=.001)
    train_op = slim.learning.create_train_op(loss, optimizer,
                                             summarize_gradients=True)

    variables_to_restore = slim.get_variables_to_restore()

    if FLAGS.resnet50_model_path is not None:
        init_fn = slim.assign_from_checkpoint_fn(FLAGS.resnet50_model_path,
                                                 variables_to_restore,
                                                 ignore_missing_vars=True)
    else:
        print('restoring model from checkpoint ....')
        init_fn = slim.assign_from_checkpoint_fn(FLAGS.checkpoint_prefix,
                                                 variables_to_restore,
                                                 ignore_missing_vars=True)

    slim.learning.train(train_op=train_op, logdir=FLAGS.log_dir,
                        init_fn=init_fn,
                        number_of_steps=1000,
                        save_summaries_secs=20,
                        save_interval_secs=300)


if __name__ == '__main__':
    tf.app.run()
