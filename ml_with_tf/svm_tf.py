import numpy as np
import tensorflow as tf
from sklearn import datasets


x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)


def gen_data(batch_size):
    iris = datasets.load_iris()
    iris_X = np.array([[x[0], x[3]] for x in iris.data])
    iris_y = np.array([1 if y == 0 else -1 for y in iris.target])
    train_indices = np.random.choice(len(iris_X),
                int(round(len(iris_X) * 0.8)), replace=False)
    train_x = iris_X[train_indices]
    train_y = iris_y[train_indices]
    rand_index = np.random.choice(len(train_x), size=batch_size)
    batch_train_x = train_x[rand_index]
    batch_train_y = np.transpose([train_y[rand_index]])
    test_indices = np.array(
        list(set(range(len(iris_X))) - set(train_indices)))
    test_x = iris_X[test_indices]
    test_y = iris_y[test_indices]
    return batch_train_x, batch_train_y, test_x, test_y


def svm():
    A = tf.Variable(tf.random_normal(shape=[2, 1]))
    b = tf.Variable(tf.random_normal(shape=[1, 1]))
    model_output = tf.subtract(tf.matmul(x_data, A), b)
    l2_norm = tf.reduce_sum(tf.square(A))
    alpha = tf.constant([0.01])
    classification_term = tf.reduce_mean(tf.maximum(0.,
         tf.subtract(1., tf.multiply(model_output, y_target))))
    loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))
    my_opt = tf.train.GradientDescentOptimizer(0.01)
    train_step = my_opt.minimize(loss)
    return model_output, loss, train_step


def train(sess, batch_size):
    print("# Training loop")

    for i in range(100):
        x_vals_train, y_vals_train,\
        x_vals_test, y_vals_test = gen_data(batch_size)
        model_output, loss, train_step = svm()

        init = tf.global_variables_initializer()
        sess.run(init)

        prediction = tf.sign(model_output)
        accuracy = tf.reduce_mean(tf.cast(
            tf.equal(prediction, y_target), tf.float32))
        sess.run(train_step, feed_dict={x_data: x_vals_train,
                                        y_target: y_vals_train})

        train_loss = sess.run(loss, feed_dict={x_data: x_vals_train,
                                               y_target: y_vals_train})
        train_acc = sess.run(accuracy, feed_dict={x_data: x_vals_train,
                                                  y_target: y_vals_train})

        test_acc = sess.run(accuracy, feed_dict={x_data: x_vals_test,
                                                 y_target: np.transpose([y_vals_test])})

        if i % 10 == 1:
            print("train loss: {:.6f}, train accuracy : {:.6f}".
                  format(train_loss[0], train_acc))
            print
            print("test accuracy : {:.6f}".format(test_acc))
            print("- * - "*15)


def main(_):
    with tf.Session() as sess:
        train(sess, batch_size=16)


if __name__ == "__main__":
    tf.app.run()
