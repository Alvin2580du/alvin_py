import tensorflow as tf
import time
import os
import pprint

from pyduyp.utils.dl.utils import input_setup, read_data, merge, imsave

flags = tf.app.flags
flags.DEFINE_integer("epoch", 15000, "Number of epoch [15000]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 33, "The size of image to use [33]")
flags.DEFINE_integer("label_size", 21, "The size of label to produce [21]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", 14, "The size of stride to apply input image [14]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "Train", "Name of sample directory [sample]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [True]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()


class SRCNN(object):
    def __init__(self, sess, image_size=33, label_size=21, batch_size=128,
                 c_dim=1, checkpoint_dir=None, sample_dir=None):

        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size

        self.c_dim = c_dim

        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.build_model()
        input_setup(self.sess, FLAGS)

    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')

        self.weights = {
            'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=1e-3), name='w1'),
            'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=1e-3), name='w2'),
            'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=1e-3), name='w3')
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros([64]), name='b1'),
            'b2': tf.Variable(tf.zeros([32]), name='b2'),
            'b3': tf.Variable(tf.zeros([1]), name='b3')
        }
        self.pred = self.model()
        # Loss function (MSE)
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        self.saver = tf.train.Saver()

    def train(self, config):
        data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")

        train_data, train_label = read_data(data_dir)
        self.train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)
        tf.initialize_all_variables().run()
        counter = 0
        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        print("Training...")
        for ep in range(config.epoch):
            batch_idxs = len(train_data) // config.batch_size
            for idx in range(0, batch_idxs):
                batch_images = train_data[idx * config.batch_size: (idx + 1) * config.batch_size]
                batch_labels = train_label[idx * config.batch_size: (idx + 1) * config.batch_size]

                counter += 1
                feed_dict = {self.images: batch_images, self.labels: batch_labels}
                _, err = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)

                if counter % 10 == 0:
                    print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" %
                          ((ep + 1), counter, time.time() - start_time, err))

                if counter % 500 == 0:
                    self.save(config.checkpoint_dir, counter)

    def test(self, config):
        print("Testing...")
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")
        train_data, train_label = read_data(data_dir)
        result = self.pred.eval({self.images: train_data, self.labels: train_label})
        nx, ny = input_setup(self.sess, config)

        result = merge(result, [nx, ny])
        result = result.squeeze()
        image_path = os.path.join(os.getcwd(), config.sample_dir)
        image_path = os.path.join(image_path, "test_image.png")
        print(image_path)
        imsave(result, image_path)

    def model(self):
        conv1 = tf.nn.conv2d(self.images, self.weights['w1'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b1']
        conv1 = tf.nn.relu(conv1)
        conv2 = tf.nn.conv2d(conv1, self.weights['w2'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b2']
        conv2 = tf.nn.relu(conv2)
        conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b3']
        return conv3

    def save(self, checkpoint_dir, step):
        model_name = "SRCNN.model"
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % ("srcnn", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    with tf.Session() as sess:

        srcnn = SRCNN(sess,
                      image_size=FLAGS.image_size,
                      label_size=FLAGS.label_size,
                      batch_size=FLAGS.batch_size,
                      c_dim=FLAGS.c_dim,
                      checkpoint_dir=FLAGS.checkpoint_dir,
                      sample_dir=FLAGS.sample_dir)
        if FLAGS.is_train:
            srcnn.train(FLAGS)
        else:
            srcnn.test(FLAGS)


if __name__ == '__main__':
    tf.app.run()
