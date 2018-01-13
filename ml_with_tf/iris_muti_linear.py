import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import datasets
import os


def make_iris():
    iris = datasets.load_iris()
    x = pd.DataFrame(iris.data)
    y = pd.DataFrame(iris.target).values
    y_onehot = tf.one_hot(y, 3)
    sess = tf.InteractiveSession()
    y_onehot_value = sess.run(y_onehot).reshape((150, 3))
    y_onehot_value = pd.DataFrame(y_onehot_value)
    x.to_csv("iris_x.csv", sep=',', header=None, index=None)
    y_onehot_value.to_csv("iris_y.csv", sep=',', header=None, index=None)


def model(inputs, in_size, out_size):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]))
    outputs = tf.nn.softmax(tf.matmul(inputs, Weights) + biases)
    return outputs


def train():
    x_data = pd.read_csv("iris_x.csv", header=None).values
    y_data = pd.read_csv("iris_y.csv", header=None).values
    train_x = x_data[0:120, :]
    train_y = y_data[0:120, :]

    test_x = x_data[120:151, :]
    test_y = y_data[120:151, :]


    x_data_holder = tf.placeholder(tf.float32, [None, 4])
    y_data_holder = tf.placeholder(tf.float32, [None, 3])

    y_prediction = model(x_data_holder, 4, 3)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_data_holder * tf.log(y_prediction), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        epoch = 2000
        for e in range(epoch):
            sess.run(train_step, feed_dict={x_data_holder: train_x, y_data_holder: train_y})
            if e % 50 == 0:
                train_loss = sess.run(cross_entropy, feed_dict={x_data_holder: train_x, y_data_holder: train_y})

                y_pre = sess.run(y_prediction, feed_dict={x_data_holder: test_x})
                correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(test_y, 1))
                print(correct_prediction.eval(session=sess))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                test_acc = sess.run(accuracy, feed_dict={x_data_holder: test_x, y_data_holder: test_y})
                print("acc: {}; loss: {}".format(test_acc, train_loss))

        training_cost = sess.run(cross_entropy, feed_dict={x_data_holder: train_x, y_data_holder: train_y})
        print("Training cost={}".format(training_cost))

if __name__ == "__main__":
    train()
