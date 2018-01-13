import tensorflow as tf
import os
from pyduyp.logger.log import log

from tensorflow.examples.tutorials.mnist import input_data

learning_rate = 0.001
decay = 0.9
training_epochs = 10
batch_size = 100
p_keep_conv_value = 0.8
p_keep_hidden_value = 0.5

# set variables
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

p_keep_conv = tf.placeholder(tf.float32)
p_keep_hidden = tf.placeholder(tf.float32)

script_dir = os.path.dirname(os.path.abspath(__file__))
mnist = input_data.read_data_sets(script_dir, one_hot=True)

W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
W4 = tf.Variable(tf.random_normal([128 * 4 * 4, 625], stddev=0.01))
W5 = tf.Variable(tf.random_normal([625, 10], stddev=0.01))


def model(inputs):
    with tf.variable_scope("layer") as vs:
        L1 = tf.nn.relu(tf.nn.conv2d(inputs, W1, strides=[1, 1, 1, 1], padding='SAME'))
        L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        L1 = tf.nn.dropout(L1, p_keep_conv)

        L2 = tf.nn.relu(tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME'))
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        L2 = tf.nn.dropout(L2, p_keep_conv)
        L3 = tf.nn.relu(tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME'))
        L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        L3 = tf.reshape(L3, [-1, W4.get_shape().as_list()[0]])
        L3 = tf.nn.dropout(L3, p_keep_conv)
        L4 = tf.nn.relu(tf.matmul(L3, W4))
        L4 = tf.nn.dropout(L4, p_keep_hidden)
        model = tf.matmul(L4, W5)
    return model


def build_model():
    model_out = model(X)
    # build training operation
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=model_out, logits=Y))
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    return train_op


def train():
    with tf.Session() as sess:

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        test_feed_dict = {
            X: mnist.test.images.reshape(-1, 28, 28, 1),
            Y: mnist.test.labels,
            p_keep_conv: p_keep_conv_value,
            p_keep_hidden: p_keep_hidden_value
        }

        for epoch in range(training_epochs):
            total_batch = int(mnist.train.num_examples / batch_size)

            for step in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)

                feed_dict = {
                    X: batch_xs.reshape(-1, 28, 28, 1),
                    Y: batch_ys,
                    p_keep_conv: p_keep_conv_value,
                    p_keep_hidden: p_keep_hidden_value
                }
                train_op = build_model()
                sess.run(train_op, feed_dict=feed_dict)

            check_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
            accuracy_rates = sess.run(accuracy, feed_dict=test_feed_dict)

            log.info("{}".format('Epoch:', '%04d' % (epoch + 1), '/ Accuracy =', accuracy_rates))


if __name__ == "__main__":
    train()
