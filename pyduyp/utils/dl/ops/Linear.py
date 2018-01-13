import tensorflow as tf


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        weight = tf.get_variable("weight", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, weight) + bias, weight, bias
        else:
            return tf.matmul(input_, weight) + bias

if __name__ == "__main__":

    inputs = tf.Variable(tf.random_normal([1, 64, 64, 3]))
    op = linear(inputs, output_size=3)
    print(op)



