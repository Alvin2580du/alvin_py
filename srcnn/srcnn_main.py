import tensorflow as tf
import time
import pprint
import os
import glob
import h5py
import scipy.misc
import scipy.ndimage
import numpy as np

from pyduyp.logger.log import log


flags = tf.app.flags
flags.DEFINE_integer("epoch", 15000, "Number of epoch [15000]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
flags.DEFINE_integer("image_size", 64, "The size of image to use [33]")
flags.DEFINE_integer("label_size", 64, "The size of label to produce [21]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
flags.DEFINE_integer("scale", 4, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", 14, "The size of stride to apply input image [14]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "result", "Name of sample directory [sample]")
flags.DEFINE_string("train_data", "yaogan", "Name of sample directory [sample]")
flags.DEFINE_string("test_data", "yaogan", "Name of sample directory [sample]")

flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()


def read_data(path):
    """
  Read h5 format data file

  Args:
    path: file path of desired file
    data: '.h5' file format that contains train data values
    label: '.h5' file format that contains train label values
  """
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        label = np.array(hf.get('label'))
        return data, label


def preprocess(path, scale=4):
    """
  Preprocess single image file 
    (1) Read original image as YCbCr format (and grayscale as default)
    (2) Normalize
    (3) Apply image file with bicubic interpolation

  Args:
    path: file path of desired file
    input_: image applied bicubic interpolation (low-resolution)
    label_: image with original resolution (high-resolution)
  """
    image = imread(path, is_grayscale=False)
    log.debug("image shape:{}".format(image.shape))
    label_ = modcrop(image, scale)
    log.debug("label_ shape: {}".format(label_.shape))
    # Must be normalized
    image = image / 255.
    label_ = label_ / 255.
    input_ = scipy.ndimage.interpolation.zoom(label_, (1. / scale), prefilter=False)
    log.debug("inputs shape: {}".format(input_.shape))
    input_ = scipy.ndimage.interpolation.zoom(input_, (scale / 1.), prefilter=False)
    log.debug("inputs shape: {}".format(input_.shape))
    return input_, label_


def prepare_data(dataset):
    """
  Args:
    dataset: choose train dataset or test dataset

    For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
  """
    if FLAGS.is_train:
        data_dir = os.path.join(os.getcwd(), dataset)
        data = glob.glob(os.path.join(data_dir, "*.png"))
    else:
        data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)), "Set5")
        log.debug("65: data dir:{}".format(data_dir))
        data = glob.glob(os.path.join(data_dir, "*.png"))

    return data


def make_data(data, label):
    """
  Make input data as h5 file format
  Depending on 'is_train' (flag value), savepath would be changed.
  """
    if FLAGS.is_train:
        savepath = os.path.join(os.getcwd(), 'checkpoint/train.h5')
    else:
        savepath = os.path.join(os.getcwd(), 'checkpoint/test.h5')

    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=label)


def imread(path, is_grayscale=True):
    """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
    if is_grayscale:
        return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
    else:
        return scipy.misc.imread(path, mode='YCbCr').astype(np.float)


def modcrop(image, scale=4):
    """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.

  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image


def input_setup(config):
    if config.is_train:
        data = prepare_data(dataset="Train\\{}".format(config.train_data))
    else:
        data = prepare_data(dataset="Test\\{}".format(config.test_data))

    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(config.image_size - config.label_size) / 2  # 6

    if config.is_train:
        for i in range(len(data)):
            input_, label_ = preprocess(data[i], config.scale)

            if len(input_.shape) == 3:
                h, w, _ = input_.shape
            else:
                h, w = input_.shape
            log.debug("{}, {}".format(h, w))
            for x in range(0, h - config.image_size + 1, config.stride):
                for y in range(0, w - config.image_size + 1, config.stride):
                    sub_input = input_[x:x + config.image_size, y:y + config.image_size]
                    sub_label = label_[x + int(padding):x + int(padding) + config.label_size,
                                y + int(padding):y + int(padding) + config.label_size]

                    # Make channel value
                    sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
                    sub_label = sub_label.reshape([config.label_size, config.label_size, 1])

                    sub_input_sequence.append(sub_input)
                    sub_label_sequence.append(sub_label)

    else:
        input_, label_ = preprocess(data[2], config.scale)

        if len(input_.shape) == 3:
            h, w, _ = input_.shape
        else:
            h, w = input_.shape
        nx = ny = 0
        for x in range(0, h - config.image_size + 1, config.stride):
            nx += 1
            ny = 0
            for y in range(0, w - config.image_size + 1, config.stride):
                ny += 1
                sub_input = input_[x:x + config.image_size, y:y + config.image_size]  # [33 x 33]
                sub_label = label_[x + int(padding):x + int(padding) + config.label_size,
                            y + int(padding):y + int(padding) + config.label_size]  # [21 x 21]

                sub_input = sub_input.reshape([config.image_size, config.image_size, 1])
                sub_label = sub_label.reshape([config.label_size, config.label_size, 1])

                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)

    arrdata = np.asarray(sub_input_sequence)
    arrlabel = np.asarray(sub_label_sequence)
    log.info("{}, {}".format(arrdata.shape, arrlabel.shape))
    make_data(arrdata, arrlabel)

    if not config.is_train:
        return nx, ny


def imsave(image, path):
    return scipy.misc.imsave(path, image)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 1))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


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
        print("  build modeel success ")
        input_setup(FLAGS)

    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')
        log.debug("{}".format(self.labels.get_shape().as_list()))
        self.weights = {
            'w1': tf.Variable(tf.random_normal([5, 5, 1, 64], stddev=1e-3), name='w1'),
            'w2': tf.Variable(tf.random_normal([3, 3, 64, 32], stddev=1e-3), name='w2'),
            'w3': tf.Variable(tf.random_normal([3, 3, 32, 1], stddev=1e-3), name='w3'),
            'w4': tf.Variable(tf.random_normal([3, 3, 64, 32], stddev=1e-3), name='w4'),
            'w5': tf.Variable(tf.random_normal([3, 3, 32, 1], stddev=1e-3), name='w5')
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros([64]), name='b1'),
            'b2': tf.Variable(tf.zeros([32]), name='b2'),
            'b3': tf.Variable(tf.zeros([1]), name='b3'),
            'b4': tf.Variable(tf.zeros([32]), name='b4'),
            'b5': tf.Variable(tf.zeros([1]), name='b5')
        }
        """value,
                     filter,  # pylint: disable=redefined-builtin
                     output_shape,
                     strides,
                     padding="SAME",
                     data_format="NHWC",
                     name=None"""
        # self.pred = self.model()
        conv1 = tf.nn.conv2d(self.images, self.weights['w1'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b1']
        conv1 = tf.nn.relu(conv1)
        log.debug("{}".format(conv1))
        conv2 = tf.nn.conv2d(conv1, self.weights['w2'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b2']
        conv2 = tf.nn.relu(conv2)
        log.debug("{}".format(conv2))
        conv3 = tf.nn.conv2d_transpose(conv2, output_shape=64, filter=self.weights['w3'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b3']
        conv3 = tf.nn.relu(conv3)
        log.debug("{}".format(conv3))
        conv4 = tf.nn.conv2d_transpose(conv3, output_shape=64, filter=self.weights['w4'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b4']
        conv4 = tf.nn.relu(conv4)
        log.debug("{}".format(conv4))
        self.pred = tf.nn.conv2d_transpose(conv4, output_shape=64, filter=self.weights['w5'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b5']
        log.debug("{}".format(self.pred))
        exit(1)
        # Loss function (MSE)
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))
        self.saver = tf.train.Saver()

    def train(self, config):
        data_dir = os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")

        train_data, train_label = read_data(data_dir)
        log.debug("{}, {}".format(type(train_data), type(train_label)))
        self.train_op = tf.train.GradientDescentOptimizer(config.learning_rate).minimize(self.loss)
        tf.global_variables_initializer().run()
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
                log.debug(" batch_images: {}".format(batch_images.shape))
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
        nx, ny = input_setup(config)

        result = merge(result, [nx, ny])
        result = result.squeeze()
        image_path = os.path.join(os.getcwd(), config.sample_dir)
        image_path = os.path.join(image_path, "test_image.png")
        print(image_path)
        imsave(result, image_path)

    # def model(self):
    #     conv1 = tf.nn.conv2d(self.images, self.weights['w1'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b1']
    #     conv1 = tf.nn.relu(conv1)
    #     conv2 = tf.nn.conv2d(conv1, self.weights['w2'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b2']
    #     conv2 = tf.nn.relu(conv2)
    #     conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1, 1, 1, 1], padding='VALID') + self.biases['b3']
    #     return conv3

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
