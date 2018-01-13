import tensorflow as tf
import os

from pyduyp.logger.log import log

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def fc_variable():
    v1 = tf.Variable(initial_value=tf.random_normal(shape=[2, 3], mean=0., stddev=1.), dtype=tf.float32,
                     name='variable_1')
    return v1


def variable_value(variables):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        """
        tensorflow.python.framework.errors_impl.FailedPreconditionError: Attempting to use uninitialized value variable_1
        """
        print(sess.run(variables))
        """
        [[ 0.00556329  0.20311342 -0.79569227]
         [ 0.1700473   0.9499892  -0.46801034]]
        """


def fc_variable_scope():
    with tf.variable_scope("foo"):
        v = tf.get_variable("v", [1])
        w = tf.get_variable("w", [1])
        log.debug("{}, {}".format(v, w))
    with tf.variable_scope("foo", reuse=True):
        v1 = tf.get_variable("v")
        log.debug("{}".format(v1))


"""
foo/v:0
foo/w:0
foo/v:0
"""


def fc_variable_scope_v2():
    with tf.variable_scope("foo"):
        v = tf.get_variable("v", [1])
        w = tf.get_variable("w", [1])
        log.debug("{}, {}".format(v, w))

    with tf.variable_scope("foo", reuse=False):
        v1 = tf.get_variable("v")


"""
ValueError: Variable foo/v already exists, disallowed. 
Did you mean to set reuse=True in VarScope? Originally defined at:
"""


def fc_variable_scope_v3():
    with tf.variable_scope("foo"):
        v = tf.get_variable("v", [1])
        w = tf.get_variable("w", [1])
        log.debug("{}, {}".format(v, w))

    with tf.variable_scope("foo", reuse=True):
        v1 = tf.get_variable("u")


"""
ValueError: Variable foo/u does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=None in VarScope?


"""


def fc_variable_scope_v4():
    with tf.variable_scope("foo"):
        v = tf.get_variable("v", [1])
        w = tf.get_variable("w", [1])
        log.debug("{}, {}".format(v, w))

    with tf.variable_scope("foo", reuse=False):
        v1 = tf.get_variable("u")


"""
ValueError: Shape of a new variable (foo/u) must be fully defined, but instead was <unknown>.

"""


def fc_variable_scope_v5():
    with tf.variable_scope("foo"):
        v = tf.get_variable("v", [1])
        w = tf.get_variable("w", [1])
        log.debug("{}, {}".format(v, w))

    with tf.variable_scope("foo", reuse=False):
        v1 = tf.get_variable("u", [1])


"""
foo/v:0
foo/w:0
foo/u:0
"""


def fc_variable_scope_v6():
    with tf.variable_scope("foo"):
        v1 = tf.Variable(tf.random_normal(shape=[2, 3], mean=0., stddev=1.), dtype=tf.float32, name='v1')
        v2 = tf.get_variable("v2", [1])
        log.debug("{}, {}".format(v1, v2))

    with tf.variable_scope("foo", reuse=True):
        v3 = tf.get_variable('v2')
        v4 = tf.get_variable('v1')
        log.debug("{}, {}".format(v3, v4))


"""
ValueError: Variable foo/v1 does not exist, or was not created with tf.get_variable(). Did you mean to set reuse=None in VarScope?

"""


def compare_name_and_variable_scope():
    with tf.name_scope("hello") as ns:
        arr1 = tf.get_variable("arr1", shape=[2, 10], dtype=tf.float32)
        print(arr1.name)
    with tf.variable_scope("hello") as vs:
        arr1 = tf.get_variable("arr1", shape=[2, 10], dtype=tf.float32)
        print(arr1.name)


if __name__ == "__main__":
    fc_variable_scope_v6()
