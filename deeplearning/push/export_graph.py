# -*- coding: utf-8 -*-
import tensorflow as tf

import exporter
import model

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('input_type', 'image_tensor', 'Type of input node. Can '
                                                  "be one of ['image_tensor', 'encoded_image_string_tensor'"
                                                  ", 'tf_example']")
flags.DEFINE_string('input_shape', None, "specified as '[None, None, None, 3]'.")
flags.DEFINE_string('checkpoint_prefix', None, 'path to trained checkpoint')
flags.DEFINE_string('output_dir', None, 'path to write outputs')
tf.app.flags.mark_flag_as_required('checkpoint_prefix')
tf.app.flags.mark_flag_as_required('output_dir')
FLAGS = flags.FLAGS

CLASS_NUM = 2
INPUT_WIDTH = 224
INPUT_HEIGHT = 224
INPUT_CHANNEL = 3


def main(_):
    cls_model = model.Model(is_training=False, num_classes=CLASS_NUM)
    if FLAGS.input_shape:
        input_shape = [
            int(dim) if dim != -1 else None
            for dim in FLAGS.input_shape.split(',')
            ]
    else:
        input_shape = [None, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNEL]
    exporter.export_inference_graph(FLAGS.input_type,
                                    cls_model,
                                    FLAGS.checkpoint_prefix,
                                    FLAGS.output_dir,
                                    input_shape)


if __name__ == '__main__':
    tf.app.run()
