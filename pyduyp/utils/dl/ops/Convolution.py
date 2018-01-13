import tensorflow as tf


def conv2d(inputs, output_dim, ksize, stride, stddev=0.02, scope="conv2d", biases=True):

    with tf.variable_scope(scope):
        w_shape = [ksize, ksize, inputs.get_shape()[-1], output_dim]
        w_init = tf.truncated_normal_initializer(stddev=stddev)
        w = tf.get_variable(name='weights', shape=w_shape, initializer=w_init)
        conv_stride = [1, stride, stride, 1]
        conv = tf.nn.conv2d(input=inputs, filter=w, strides=conv_stride, padding='SAME', data_format='NHWC', name='conv')
        if biases:
            b = tf.get_variable(name='biases', shape=[output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())
        return conv


def deconv2d(input_, output_shape, kernel_height=5, kernel_width=5, jump_row_number=2, jump_column_number=2,
             stddev=0.02, name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [kernel_height, kernel_width, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, jump_row_number,
                                                                                       jump_column_number, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


if __name__ == "__main__":

    inputs = tf.Variable(tf.random_normal([1, 64, 64, 3]))
    op = conv2d(inputs=inputs, output_dim=64, ksize=3, stride=1, scope='conv2d_1', biases=False)
    print(op, op.name)
    op = conv2d(inputs=op, output_dim=128, ksize=3, stride=2, scope='conv2d_2')
    print(op, op.name)

