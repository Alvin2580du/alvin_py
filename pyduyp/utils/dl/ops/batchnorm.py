import tensorflow as tf
import numpy as np


class ConvolutionalBatchNormalizer(object):
    """Helper class that groups the normalization logic and variables.        

  Use:                                                                      
      ewma = tf.train.ExponentialMovingAverage(decay=0.99)                  
      bn = ConvolutionalBatchNormalizer(depth, 0.001, ewma, True)           
      update_assignments = bn.get_assigner()                                
      x = bn.normalize(y, train=training?)                                  
      (the output x will be batch-normalized).                              
  """

    def __init__(self, depth, epsilon, ewma_trainer, scale_after_norm):
        self.mean = tf.Variable(tf.constant(0.0, shape=[depth]),
                                trainable=False)
        self.variance = tf.Variable(tf.constant(1.0, shape=[depth]),
                                    trainable=False)
        self.beta = tf.Variable(tf.constant(0.0, shape=[depth]))
        self.gamma = tf.Variable(tf.constant(1.0, shape=[depth]))
        self.ewma_trainer = ewma_trainer
        self.epsilon = epsilon
        self.scale_after_norm = scale_after_norm

    def get_assigner(self):
        """Returns an EWMA apply op that must be invoked after optimization."""
        return self.ewma_trainer.apply([self.mean, self.variance])

    def normalize(self, x, train=True):
        """Returns a batch-normalized version of x."""
        if train:
            mean, variance = tf.nn.moments(x, [0, 1, 2])
            assign_mean = self.mean.assign(mean)
            assign_variance = self.variance.assign(variance)
            with tf.control_dependencies([assign_mean, assign_variance]):
                return tf.nn.batch_norm_with_global_normalization(
                    x, mean, variance, self.beta, self.gamma,
                    self.epsilon, self.scale_after_norm)
        else:
            mean = self.ewma_trainer.average(self.mean)
            variance = self.ewma_trainer.average(self.variance)
            local_beta = tf.identity(self.beta)
            local_gamma = tf.identity(self.gamma)
            return tf.nn.batch_norm_with_global_normalization(
                x, mean, variance, local_beta, local_gamma,
                self.epsilon, self.scale_after_norm)


def batch_norm(x, n_out, phase_train, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


class Contrib_batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None,
                                            epsilon=self.epsilon, scale=True, is_training=train,
                                            scope=self.name)


def contrib_batch_norm(input, train=True, epsilon=1e-5, momentum=0.9, name='batch_norm'):
    with tf.variable_scope(name):
        return tf.contrib.layers.batch_norm(input, decay=momentum, updates_collections=None,
                                            epsilon=epsilon, scale=True, is_training=train,
                                            scope=name)


if __name__ == "__main__":
    import math

    n_in, n_out = 3, 16
    ksize = 3
    stride = 1
    phase_train = tf.placeholder(tf.bool, name='phase_train')
    input_image = tf.placeholder(tf.float32, name='input_image')
    kernel = tf.Variable(
        tf.truncated_normal([ksize, ksize, n_in, n_out], stddev=math.sqrt(2.0 / (ksize * ksize * n_out))),
        name='kernel')
    conv = tf.nn.conv2d(input_image, kernel, [1, stride, stride, 1], padding='SAME')
    conv_bn = batch_norm(conv, n_out, phase_train)
    relu = tf.nn.relu(conv_bn)

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        for i in range(20):
            test_image = np.random.rand(4, 32, 32, 3)
            sess_outputs = session.run([relu],
                                       {input_image.name: test_image, phase_train.name: True})
            print(np.array(sess_outputs).shape)
