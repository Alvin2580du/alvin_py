import tensorflow as tf

x = tf.constant(2)
y = tf.constant(5)


def f1():
    return tf.multiply(x, 17)


def f2():
    return tf.add(y, 23)


r = tf.cond(tf.less(x, y), f1, f2)  # 2小于5， 所以返回f1.

sess = tf.Session()
print(sess.run(r))

