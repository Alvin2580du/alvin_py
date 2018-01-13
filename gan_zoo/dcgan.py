from __future__ import division
import os
import time
from glob import glob
from six.moves import xrange
import pprint
import numpy as np
import tensorflow as tf

from pyduyp.utils.dl.ops.Convolution import conv2d as conv2d
from pyduyp.utils.dl.ops.Convolution import deconv2d as deconv2d
from pyduyp.utils.dl.ops.batchnorm import Contrib_batch_norm as batch_norm
from pyduyp.utils.image_utils import save_images, get_image
from pyduyp.utils.dl.ops.variable_ops import lrelu, conv_out_size_same, show_all_variables
from pyduyp.utils.dl.ops.Linear import linear as linear


from pyduyp.logger.log import log
log.info("================= DCGAN Runing =================")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")

flags.DEFINE_integer("epoch", 15, "Epoch to train [15]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")

flags.DEFINE_integer("sample_num", 64, "The size of batch images [64]")


flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped).")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce.")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")

flags.DEFINE_string("dataset", "celeba", "The name of dataset [celebA, mnist, lsun]")
# data_path: change to your datasets path
flags.DEFINE_string("data_path", "D:\\alvin_data", "The path of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("loss_dir", "loss", "Directory name to save the image loss [loss]")

flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", True, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

"""
Total size of variables: 6696324
Total bytes of variables: 26785296
"""
pp = pprint.PrettyPrinter()


class DCGAN(object):
    def __init__(self, sess, input_height=108, input_width=108, is_crop=True,
           batch_size=64, sample_num=100, output_height=64, output_width=64,
           y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
           gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
           input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):

        self.sess = sess
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)

        self.batch_size = batch_size
        self.sample_num = sample_num
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.build_model()

    def build_model(self):
        if self.y_dim:
            self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

        if self.is_crop:
            image_dims = [self.output_height, self.output_width, self.c_dim]
        else:
            image_dims = [self.input_height, self.input_width, self.c_dim]

        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
        # self.sample_inputs = tf.placeholder(tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

        inputs = self.inputs
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        if self.y_dim:
            self.G = self.generator(self.z, self.y)
            self.D, self.D_logits = self.discriminator(inputs, self.y, reuse=False)

            self.sampler = self.sampler(self.z, self.y)
            self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
        else:
            self.G = self.generator(self.z)
            self.D, self.D_logits = self.discriminator(inputs)

            self.sampler = self.sampler(self.z)
            self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        def sigmoid_cross_entropy_with_logits(x, y):
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))

        self.g_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss = self.d_loss_real + self.d_loss_fake

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=100)

    def train(self, dcgan_config):
        """Train DCGAN"""

        d_optim = tf.train.AdamOptimizer(dcgan_config.learning_rate, beta1=dcgan_config.beta1).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(dcgan_config.learning_rate, beta1=dcgan_config.beta1).minimize(self.g_loss, var_list=self.g_vars)
        tf.global_variables_initializer().run()

        counter = 1
        start_time = time.time()
        for epoch in xrange(dcgan_config.epoch):
            data = glob(os.path.join(dcgan_config.data_path, dcgan_config.dataset, self.input_fname_pattern))
            np.random.shuffle(data)

            batch_count = min(len(data), dcgan_config.train_size) // dcgan_config.batch_size
            print("batch_count is ", batch_count)
            for bc in xrange(0, batch_count):

                batch_files = data[bc * dcgan_config.batch_size: (bc+1) * dcgan_config.batch_size]

                batch = [get_image(batch_file, input_height=self.input_height, input_width=self.input_width,
                                   resize_height=self.output_height, resize_width=self.output_width,
                                   is_crop=self.is_crop, is_grayscale=self.is_grayscale) for batch_file in batch_files]
                if (self.is_grayscale):
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [dcgan_config.batch_size, self.z_dim]).astype(np.float32)
                self.sess.run(d_optim, feed_dict={self.inputs: batch_images, self.z: batch_z})

                # Update G network
                self.sess.run(g_optim, feed_dict={self.z: batch_z})
                self.sess.run(g_optim, feed_dict={self.z: batch_z})

                fake_loss = self.d_loss_fake.eval({self.z: batch_z})
                real_loss = self.d_loss_real.eval({self.inputs: batch_images})
                g_loss = self.g_loss.eval({self.z: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"
                      % (epoch, bc, batch_count, time.time() - start_time, fake_loss+real_loss, g_loss))

                if np.mod(counter, 200) == 0:
                    sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

                    # sample_z = np.random.normal(-1, 1, size=(self.sample_num, self.z_dim))

                    samples, d_loss, g_loss = self.sess.run([self.sampler, self.d_loss, self.g_loss],
                                                            feed_dict={self.z: sample_z, self.inputs: batch_images})

                    save_images(samples, [8, 8], './{}/train_{:02d}_{:04d}.png'.format(dcgan_config.sample_dir, epoch, bc))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter, 10) == 0:
                    self.save(self.checkpoint_dir)

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            if not self.y_dim:
                h0 = lrelu(conv2d(image, self.df_dim, ksize=3, stride=2, scope='d_h0_conv'))
                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, ksize=3, stride=2, scope='d_h1_conv')))
                h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, ksize=3, stride=2, scope='d_h2_conv')))
                h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, ksize=3, stride=2, scope='d_h3_conv')))
                h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
                out = tf.nn.sigmoid(h4)
                return out, h4

    def generator(self, z):
        # with tf.variable_scope("generator") as scope:
        with tf.variable_scope("generator"):
            if not self.y_dim:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

                # project `z` and reshape
                self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)
                self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(self.h0))
                self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
                h1 = tf.nn.relu(self.g_bn1(self.h1))
                h2, self.h2_w, self.h2_b = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
                h2 = tf.nn.relu(self.g_bn2(h2))
                h3, self.h3_w, self.h3_b = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
                h3 = tf.nn.relu(self.g_bn3(h3))
                h4, self.h4_w, self.h4_b = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)
                return tf.nn.tanh(h4)

    def sampler(self, z):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            if not self.y_dim:
                s_h, s_w = self.output_height, self.output_width
                s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
                s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
                s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
                s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
                h0 = tf.reshape(linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'), [-1, s_h16, s_w16, self.gf_dim * 8])
                h0 = tf.nn.relu(self.g_bn0(h0, train=False))
                h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
                h1 = tf.nn.relu(self.g_bn1(h1, train=False))
                h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
                h2 = tf.nn.relu(self.g_bn2(h2, train=False))
                h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
                h3 = tf.nn.relu(self.g_bn3(h3, train=False))
                h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

                return tf.nn.tanh(h4)

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir):
        model_name = "DCGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
          os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name))


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    if not os.path.exists(FLAGS.loss_dir):
        os.makedirs(FLAGS.loss_dir)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True

    with tf.Session(config=run_config) as sess:
        dcgan = DCGAN(sess,
                      input_height=FLAGS.input_height,
                      output_height=FLAGS.output_height,
                      batch_size=FLAGS.batch_size,
                      sample_num=FLAGS.sample_num,
                      c_dim=FLAGS.c_dim,
                      dataset_name=FLAGS.dataset,
                      input_fname_pattern=FLAGS.input_fname_pattern,
                      is_crop=FLAGS.is_crop,
                      checkpoint_dir=FLAGS.checkpoint_dir,
                      sample_dir=FLAGS.sample_dir)

        show_all_variables()
        dcgan.train(FLAGS)


if __name__ == '__main__':
    tf.app.run()
