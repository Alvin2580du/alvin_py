import tensorflow as tf
import os
import scipy.misc
import pandas as pd
import numpy as np
from pyduyp.logger.log import log
from pyduyp.utils.dl.ops.dlops import *


def VGG19(x, is_training, reuse=False):
    with tf.variable_scope('vgg19', reuse=reuse):
        phi = []
        with tf.variable_scope('conv1a'):
            x = conv_layer(x, [3, 3, 3, 64], 1)
            x = batch_normalize(x, is_training)
            x = lrelu(x)
        with tf.variable_scope('conv1b'):
            x = conv_layer(x, [3, 3, 64, 64], 1)
            x = batch_normalize(x, is_training)
            x = lrelu(x)
        phi.append(x)

        x = max_pooling_layer(x, 2, 2)
        with tf.variable_scope('conv2a'):
            x = conv_layer(x, [3, 3, 64, 128], 1)
            x = batch_normalize(x, is_training)
            x = lrelu(x)
        with tf.variable_scope('conv2b'):
            x = conv_layer(x, [3, 3, 128, 128], 1)
            x = batch_normalize(x, is_training)
            x = lrelu(x)
        phi.append(x)

        x = max_pooling_layer(x, 2, 2)
        with tf.variable_scope('conv3a'):
            x = conv_layer(x, [3, 3, 128, 256], 1)
            x = batch_normalize(x, is_training)
            x = lrelu(x)
        with tf.variable_scope('conv3b'):
            x = conv_layer(x, [3, 3, 256, 256], 1)
            x = batch_normalize(x, is_training)
            x = lrelu(x)
        with tf.variable_scope('conv3c'):
            x = conv_layer(x, [3, 3, 256, 256], 1)
            x = batch_normalize(x, is_training)
            x = lrelu(x)
        with tf.variable_scope('conv3d'):
            x = conv_layer(x, [3, 3, 256, 256], 1)
            x = batch_normalize(x, is_training)
            x = lrelu(x)
        phi.append(x)

        x = max_pooling_layer(x, 2, 2)
        with tf.variable_scope('conv4a'):
            x = conv_layer(x, [3, 3, 256, 512], 1)
            x = batch_normalize(x, is_training)
            x = lrelu(x)
        with tf.variable_scope('conv4b'):
            x = conv_layer(x, [3, 3, 512, 512], 1)
            x = batch_normalize(x, is_training)
            x = lrelu(x)
        with tf.variable_scope('conv4c'):
            x = conv_layer(x, [3, 3, 512, 512], 1)
            x = batch_normalize(x, is_training)
            x = lrelu(x)
        with tf.variable_scope('conv4d'):
            x = conv_layer(x, [3, 3, 512, 512], 1)
            x = batch_normalize(x, is_training)
            x = lrelu(x)
        phi.append(x)

        x = max_pooling_layer(x, 2, 2)
        with tf.variable_scope('conv5a'):
            x = conv_layer(x, [3, 3, 512, 512], 1)
            x = batch_normalize(x, is_training)
            x = lrelu(x)
        with tf.variable_scope('conv5b'):
            x = conv_layer(x, [3, 3, 512, 512], 1)
            x = batch_normalize(x, is_training)
            x = lrelu(x)
        with tf.variable_scope('conv5c'):
            x = conv_layer(x, [3, 3, 512, 512], 1)
            x = batch_normalize(x, is_training)
            x = lrelu(x)
        with tf.variable_scope('conv5d'):
            x = conv_layer(x, [3, 3, 512, 512], 1)
            x = batch_normalize(x, is_training)
            x = lrelu(x)
        phi.append(x)

        x = max_pooling_layer(x, 2, 2)
        x = flatten_layer(x)
        with tf.variable_scope('fc1'):
            x = full_connection_layer(x, 4096)
            x = lrelu(x)
        with tf.variable_scope('fc2'):
            x = full_connection_layer(x, 4096)
            x = lrelu(x)
        with tf.variable_scope('softmax'):
            x = full_connection_layer(x, 1000)

        return x


is_training = tf.placeholder(tf.bool, [])

image_holder = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name='input')
vgg = VGG19(image_holder, is_training)
with tf.Session() as sess:
    batch_size = 1
    for image in os.listdir("D:\\results"):
        imagename = os.path.join("D:\\results", image)
        image = scipy.misc.imread(imagename)
        image = image.reshape((batch_size, 256, 256, 3))
        init = tf.global_variables_initializer()
        sess.run(init)
        res = sess.run(vgg, feed_dict={image_holder: image, is_training: True})
        df = pd.DataFrame(res)
        df.to_csv("D:\\tmp\\{}.csv".format(str(imagename.split("\\")[-1]).replace(".png", "")))
