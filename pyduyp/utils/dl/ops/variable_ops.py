import math
import tensorflow as tf
from tensorflow.contrib import slim


def get_stddev(x, kernel_height, kernel_width):
    return 1 / math.sqrt(kernel_width * kernel_height * x.get_shape()[-1])


def show_all_variables():
    model_vars = tf.trainable_variables()  # A list of Variable objects.
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)
    # total size of the variables, total bytes of the variables


def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)


def conv_cond_concat(x, y):
    """
    Concatenate conditioning vector on feature map axis.
    """
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return concat([x, y * tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))
