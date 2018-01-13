import os
import sys
import time
import re
import functools
import numpy as np
import tensorflow as tf
from tqdm import trange
from glob import glob
from datetime import datetime
import pprint
import cv2
import argparse
import json

from wgan_ops import linear, conv2d, batchnorm, deconv2d, lib, layernorm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

pp = pprint.PrettyPrinter()
now_time = datetime.now().strftime('%m_%d_%H:%M')
# log_file = open("log_{}".format(now_time), "w")
# sys.stdout = log_file


parser = argparse.ArgumentParser()

parser.add_argument('--MODE', type=str, default="improved_wgan",
                    choices=["dcgan", "wgan", "improved_wgan", "lsgan"], help='height')
parser.add_argument('--epochs', type=int, default=100000, help='batch size')
parser.add_argument('--batch_size', type=int, default=25, help='batch size')
parser.add_argument('--learning_rate', type=np.float32, default=0.0005, help='learning rate')
parser.add_argument('--dis_epochs', type=int, default=5, help='discrimator epoch')
parser.add_argument('--N_GPUS', type=int, default=1, help='GPU numbers')
parser.add_argument('--LAMBDA', type=int, default=10, help='lambda in paper')
parser.add_argument('--OUTPUT_DIM', type=int, default=64 * 64 * 3, help='output dim')
parser.add_argument('--DIM', type=int, default=64, help='dim')
parser.add_argument('--model_dir', type=str, default='wgan_models')
parser.add_argument('--data_dir', type=str, default='/home/dms/alvin_data/celeba_1/')
parser.add_argument('--data_format', type=str, default='*.jpg')

config, unparsed = parser.parse_known_args()

DEVICES = ['/gpu:{}'.format(i) for i in range(config.N_GPUS)]


def save_config(config):
    param_path = os.path.join(config.model_dir, "params.json")

    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)

    print("[*] MODEL dir: %s" % config.model_dir)
    print("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def GeneratorAndDiscriminator():
    # 1
    # Baseline (G: DCGAN, D: DCGAN)
    # return DCGANGenerator, DCGANDiscriminator

    # 2
    # No BN and constant number of filts in G
    # return WGANPaper_CrippledDCGANGenerator, DCGANDiscriminator

    # 3
    # 512-dim 4-layer ReLU MLP G
    # return FCGenerator, DCGANDiscriminator

    # 4
    # No normalization anywhere
    # return functools.partial(DCGANGenerator, bn=False), functools.partial(DCGANDiscriminator, bn=False)

    # 5
    # Gated multiplicative nonlinearities everywhere
    # return MultiplicativeDCGANGenerator, MultiplicativeDCGANDiscriminator

    # 6
    # tanh nonlinearities everywhere
    # return functools.partial(DCGANGenerator, bn=True, nonlinearity=tf.tanh), \
    #        functools.partial(DCGANDiscriminator, bn=True, nonlinearity=tf.tanh)

    # 7
    # 101-layer ResNet G and D
    return ResnetGenerator, ResnetDiscriminator

    # raise Exception('You must choose an architecture!')


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


def ReLULayer(name, n_in, n_out, inputs):
    output = linear.Linear(name + '.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)


def LeakyReLULayer(name, n_in, n_out, inputs):
    output = linear.Linear(name + '.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)


def Batchnorm(name, axes, inputs):
    if ('Discriminator' in name) and (config.MODE == 'improved_wgan'):
        if axes != [0, 2, 3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return layernorm.Layernorm(name, [1, 2, 3], inputs)
    else:
        return batchnorm.Batchnorm(name, axes, inputs, fused=True)


def pixcnn_gated_nonlinearity(a, b):
    return tf.sigmoid(a) * tf.tanh(b)


def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4 * kwargs['output_dim']
    output = conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0, 3, 1, 2])
    return output


def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample == 'down':
        conv_shortcut = functools.partial(conv2d.Conv2D, stride=2)
        conv_1 = functools.partial(conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim / 2)
        conv_1b = functools.partial(conv2d.Conv2D, input_dim=input_dim / 2, output_dim=output_dim / 2, stride=2)
        conv_2 = functools.partial(conv2d.Conv2D, input_dim=output_dim / 2, output_dim=output_dim)

    elif resample == 'up':
        conv_shortcut = SubpixelConv2D
        conv_1 = functools.partial(conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim / 2)
        conv_1b = functools.partial(deconv2d.Deconv2D, input_dim=input_dim / 2, output_dim=output_dim / 2)
        conv_2 = functools.partial(conv2d.Conv2D, input_dim=output_dim / 2, output_dim=output_dim)

    elif resample == None:
        conv_shortcut = conv2d.Conv2D
        conv_1 = functools.partial(conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim / 2)
        conv_1b = functools.partial(conv2d.Conv2D, input_dim=input_dim / 2, output_dim=output_dim / 2)
        conv_2 = functools.partial(conv2d.Conv2D, input_dim=input_dim / 2, output_dim=output_dim)

    else:
        raise Exception('invalid resample value')

    if output_dim == input_dim and resample == None:
        shortcut = inputs  # Identity skip-connection
    else:
        shortcut = conv_shortcut(name + '.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = tf.nn.relu(output)
    output = conv_1(name + '.Conv1', filter_size=1, inputs=output, he_init=he_init, weightnorm=False)
    output = tf.nn.relu(output)
    output = conv_1b(name + '.Conv1B', filter_size=filter_size, inputs=output, he_init=he_init, weightnorm=False)
    output = tf.nn.relu(output)
    output = conv_2(name + '.Conv2', filter_size=1, inputs=output, he_init=he_init, weightnorm=False, biases=False)
    output = Batchnorm(name + '.BN', [0, 2, 3], output)

    return shortcut + (0.3 * output)


# ! Generators
def FCGenerator(n_samples, noise=None, FC_DIM=512):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = ReLULayer('Generator.1', 128, FC_DIM, noise)
    output = ReLULayer('Generator.2', FC_DIM, FC_DIM, output)
    output = ReLULayer('Generator.3', FC_DIM, FC_DIM, output)
    output = ReLULayer('Generator.4', FC_DIM, FC_DIM, output)
    output = linear.Linear('Generator.Out', FC_DIM, config.OUTPUT_DIM, output)
    output = tf.tanh(output)
    return output


def DCGANGenerator(n_samples, noise=None, dim=config.DIM, bn=True, nonlinearity=tf.nn.relu):
    conv2d.set_weights_stdev(0.02)
    deconv2d.set_weights_stdev(0.02)
    linear.set_weights_stdev(0.02)

    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = linear.Linear('Generator.Input', 128, 4 * 4 * 8 * dim, noise)  # 128 --> 8192
    output = tf.reshape(output, [-1, 8 * dim, 4, 4])  # 8192 --> 512

    if bn:
        output = Batchnorm('Generator.BN1', [0, 2, 3], output)
    output = nonlinearity(output)

    output = deconv2d.Deconv2D('Generator.2', 8 * dim, 4 * dim, 5, output)  # 512 --> 256

    if bn:
        output = Batchnorm('Generator.BN2', [0, 2, 3], output)
    output = nonlinearity(output)

    output = deconv2d.Deconv2D('Generator.3', 4 * dim, 2 * dim, 5, output)  # 256 --> 128

    if bn:
        output = Batchnorm('Generator.BN3', [0, 2, 3], output)
    output = nonlinearity(output)

    output = deconv2d.Deconv2D('Generator.4', 2 * dim, dim, 5, output)  # 128 --> 64

    if bn:
        output = Batchnorm('Generator.BN4', [0, 2, 3], output)
    output = nonlinearity(output)

    output = deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)  # 64 --> 3
    output = tf.tanh(output)

    conv2d.unset_weights_stdev()
    deconv2d.unset_weights_stdev()
    linear.unset_weights_stdev()

    return tf.reshape(output, [-1, config.OUTPUT_DIM])  # 64 * 64 * 3


def WGANPaper_CrippledDCGANGenerator(n_samples, noise=None, dim=config.DIM):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = linear.Linear('Generator.Input', 128, 4 * 4 * dim, noise)  # 128 --> 4*4*64
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, dim, 4, 4])  # 4*4*64 --> 64*4*4

    output = deconv2d.Deconv2D('Generator.2', dim, dim, 5, output)
    output = tf.nn.relu(output)

    output = deconv2d.Deconv2D('Generator.3', dim, dim, 5, output)
    output = tf.nn.relu(output)

    output = deconv2d.Deconv2D('Generator.4', dim, dim, 5, output)
    output = tf.nn.relu(output)

    output = deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, config.OUTPUT_DIM])


def ResnetGenerator(n_samples, noise=None, dim=config.DIM):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = linear.Linear('Generator.Input', 128, 4 * 4 * 8 * dim, noise)
    output = tf.reshape(output, [-1, 8 * dim, 4, 4])  # 4*4*8*64 --> 8*64

    for i in range(6):
        output = ResidualBlock('Generator.4x4_{}'.format(i), 8 * dim, 8 * dim, 3, output, resample=None)
    output = ResidualBlock('Generator.Up1', 8 * dim, 4 * dim, 3, output, resample='up')  # 8*64 --> 4*64
    for i in range(6):
        output = ResidualBlock('Generator.8x8_{}'.format(i), 4 * dim, 4 * dim, 3, output, resample=None)
    output = ResidualBlock('Generator.Up2', 4 * dim, 2 * dim, 3, output, resample='up')  # 4*64 --> 2*64
    for i in range(6):
        output = ResidualBlock('Generator.16x16_{}'.format(i), 2 * dim, 2 * dim, 3, output, resample=None)
    output = ResidualBlock('Generator.Up3', 2 * dim, 1 * dim, 3, output, resample='up')  # 2*64 --> 1*64
    for i in range(6):
        output = ResidualBlock('Generator.32x32_{}'.format(i), 1 * dim, 1 * dim, 3, output, resample=None)
    output = ResidualBlock('Generator.Up4', 1 * dim, dim / 2, 3, output, resample='up')  # 1*64 --> 1/2*64
    for i in range(5):
        output = ResidualBlock('Generator.64x64_{}'.format(i), dim / 2, dim / 2, 3, output, resample=None)

    output = conv2d.Conv2D('Generator.Out', dim / 2, 3, 1, output, he_init=False)  # 1/2*64 --> 3
    output = tf.tanh(output / 5.)

    return tf.reshape(output, [-1, config.OUTPUT_DIM])  # 3 --> 64*64*3


def MultiplicativeDCGANGenerator(n_samples, noise=None, dim=config.DIM, bn=True):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = linear.Linear('Generator.Input', 128, 4 * 4 * 8 * dim * 2, noise)
    output = tf.reshape(output, [-1, 8 * dim * 2, 4, 4])
    if bn:
        output = Batchnorm('Generator.BN1', [0, 2, 3], output)
    output = pixcnn_gated_nonlinearity(output[:, ::2], output[:, 1::2])

    output = deconv2d.Deconv2D('Generator.2', 8 * dim, 4 * dim * 2, 5, output)
    if bn:
        output = Batchnorm('Generator.BN2', [0, 2, 3], output)
    output = pixcnn_gated_nonlinearity(output[:, ::2], output[:, 1::2])

    output = deconv2d.Deconv2D('Generator.3', 4 * dim, 2 * dim * 2, 5, output)
    if bn:
        output = Batchnorm('Generator.BN3', [0, 2, 3], output)
    output = pixcnn_gated_nonlinearity(output[:, ::2], output[:, 1::2])

    output = deconv2d.Deconv2D('Generator.4', 2 * dim, dim * 2, 5, output)
    if bn:
        output = Batchnorm('Generator.BN4', [0, 2, 3], output)
    output = pixcnn_gated_nonlinearity(output[:, ::2], output[:, 1::2])

    output = deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
    output = tf.tanh(output)

    return tf.reshape(output, [-1, config.OUTPUT_DIM])


# ! Discriminators
def MultiplicativeDCGANDiscriminator(inputs, dim=config.DIM, bn=True):
    output = tf.reshape(inputs, [-1, 3, 64, 64])

    output = conv2d.Conv2D('Discriminator.1', 3, dim * 2, 5, output, stride=2)  # 3 --> 128
    output = pixcnn_gated_nonlinearity(output[:, ::2], output[:, 1::2])

    output = conv2d.Conv2D('Discriminator.2', dim, 2 * dim * 2, 5, output, stride=2)  # 64 --> 256
    if bn:
        output = Batchnorm('Discriminator.BN2', [0, 2, 3], output)
    output = pixcnn_gated_nonlinearity(output[:, ::2], output[:, 1::2])

    output = conv2d.Conv2D('Discriminator.3', 2 * dim, 4 * dim * 2, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN3', [0, 2, 3], output)
    output = pixcnn_gated_nonlinearity(output[:, ::2], output[:, 1::2])

    output = conv2d.Conv2D('Discriminator.4', 4 * dim, 8 * dim * 2, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN4', [0, 2, 3], output)
    output = pixcnn_gated_nonlinearity(output[:, ::2], output[:, 1::2])

    output = tf.reshape(output, [-1, 4 * 4 * 8 * dim])
    output = linear.Linear('Discriminator.Output', 4 * 4 * 8 * dim, 1, output)

    return tf.reshape(output, [-1])


def ResnetDiscriminator(inputs, dim=config.DIM):
    output = tf.reshape(inputs, [-1, 3, 64, 64])
    output = conv2d.Conv2D('Discriminator.In', 3, dim / 2, 1, output, he_init=False)

    for i in xrange(5):
        output = ResidualBlock('Discriminator.64x64_{}'.format(i), dim / 2, dim / 2, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down1', dim / 2, dim * 1, 3, output, resample='down')
    for i in xrange(6):
        output = ResidualBlock('Discriminator.32x32_{}'.format(i), dim * 1, dim * 1, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down2', dim * 1, dim * 2, 3, output, resample='down')
    for i in xrange(6):
        output = ResidualBlock('Discriminator.16x16_{}'.format(i), dim * 2, dim * 2, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down3', dim * 2, dim * 4, 3, output, resample='down')
    for i in xrange(6):
        output = ResidualBlock('Discriminator.8x8_{}'.format(i), dim * 4, dim * 4, 3, output, resample=None)
    output = ResidualBlock('Discriminator.Down4', dim * 4, dim * 8, 3, output, resample='down')
    for i in xrange(6):
        output = ResidualBlock('Discriminator.4x4_{}'.format(i), dim * 8, dim * 8, 3, output, resample=None)

    output = tf.reshape(output, [-1, 4 * 4 * 8 * dim])
    output = linear.Linear('Discriminator.Output', 4 * 4 * 8 * dim, 1, output)

    return tf.reshape(output / 5., [-1])


def FCDiscriminator(inputs, FC_DIM=512, n_layers=3):
    output = LeakyReLULayer('Discriminator.Input', config.OUTPUT_DIM, FC_DIM, inputs)
    for i in xrange(n_layers):
        output = LeakyReLULayer('Discriminator.{}'.format(i), FC_DIM, FC_DIM, output)
    output = linear.Linear('Discriminator.Out', FC_DIM, 1, output)

    return tf.reshape(output, [-1])


def DCGANDiscriminator(inputs, dim=config.DIM, bn=True, nonlinearity=LeakyReLU):
    output = tf.reshape(inputs, [-1, 3, 64, 64])  # 3 * 64 * 64

    conv2d.set_weights_stdev(0.02)
    deconv2d.set_weights_stdev(0.02)
    linear.set_weights_stdev(0.02)

    output = conv2d.Conv2D('Discriminator.1', 3, dim, 5, output, stride=2)  # 3 --> 64
    output = nonlinearity(output)

    output = conv2d.Conv2D('Discriminator.2', dim, 2 * dim, 5, output, stride=2)  # 64 --> 128
    if bn:
        output = Batchnorm('Discriminator.BN2', [0, 2, 3], output)
    output = nonlinearity(output)

    output = conv2d.Conv2D('Discriminator.3', 2 * dim, 4 * dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN3', [0, 2, 3], output)
    output = nonlinearity(output)

    output = conv2d.Conv2D('Discriminator.4', 4 * dim, 8 * dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN4', [0, 2, 3], output)
    output = nonlinearity(output)

    output = tf.reshape(output, [-1, 4 * 4 * 8 * dim])
    output = linear.Linear('Discriminator.Output', 4 * 4 * 8 * dim, 1, output)

    conv2d.unset_weights_stdev()
    deconv2d.unset_weights_stdev()
    linear.unset_weights_stdev()

    return tf.reshape(output, [-1])


def make_generator(path, n_files, batch_size):
    epoch_count = [1]

    def get_epoch():
        images = np.zeros((batch_size, 3, 64, 64), dtype='int32')
        files = range(n_files)
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        epoch_count[0] += 1

        for n, i in enumerate(files):
            image_path = "%s/%06d.jpg" % (path, int(str(i + 1).zfill(len(str(n_files)))))
            image = cv2.imread(image_path)
            image = cv2.resize(image, (64, 64))
            images[n % batch_size] = image.transpose(2, 0, 1)
            if n > 0 and n % batch_size == 0:
                yield (images,)
    return get_epoch


def save_images(X, save_path):
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99 * X).astype('uint8')
    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1
    nh, nw = rows, n_samples / rows
    if X.ndim == 4:
        X = X.transpose(0, 2, 3, 1)
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh, w * nw, 3))
    else:
        raise IndexError
    for n, x in enumerate(X):
        j = n / nw
        i = n % nw
        img[j * h: j * h + h, i * w: i * w + w] = x

    cv2.imwrite(save_path, img)


Generator, Discriminator = GeneratorAndDiscriminator()


def main(config):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
        all_real_data_conv = tf.placeholder(tf.int32, shape=[config.batch_size, 3, 64, 64])
        noise = tf.placeholder(tf.float32, shape=[128, 3, 64, 64])

        split_real_data_conv = tf.split(all_real_data_conv, len(DEVICES))
        gen_costs, disc_costs = [], []

        for device_index, (device, real_data_conv) in enumerate(zip(DEVICES, split_real_data_conv)):
            with tf.device(device):
                real_data = tf.reshape(2 * ((tf.cast(real_data_conv, tf.float32) / 255.) - .5),
                                       [config.batch_size / len(DEVICES), config.OUTPUT_DIM])
                fake_data = Generator(config.batch_size / len(DEVICES))
                # dis_inputs = np.concatenate((real_data, fake_data))
                disc_real = Discriminator(real_data)
                disc_fake = Discriminator(fake_data)

                # 1. wgan
                if config.MODE == 'wgan':
                    gen_cost = -tf.reduce_mean(disc_fake)
                    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
                # 2. improved_wgan
                elif config.MODE == 'improved_wgan':
                    gen_cost = -tf.reduce_mean(disc_fake)  # compute the mean of total value about disc_fake
                    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

                    alpha = tf.random_uniform(
                        shape=[config.batch_size / len(DEVICES), 1], minval=0., maxval=1.)  # batch_size * 1

                    interpolates = alpha * fake_data + (1 - alpha) * real_data

                    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
                    # tf.reduce_sum: compute the sum of total value about tf.square(gradients), reduction_indices=[1])
                    # slopes is the second gradients
                    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))

                    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                    disc_cost += config.LAMBDA * gradient_penalty

                # 3. dcgan
                elif config.MODE == 'dcgan':

                    gen_cost = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                labels=tf.ones_like(disc_fake)))
                    disc_cost = tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                labels=tf.zeros_like(disc_fake)))
                    disc_cost += tf.reduce_mean(
                        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real,
                                                                labels=tf.ones_like(disc_real)))
                    disc_cost /= 2.

                # 4. lsgan
                elif config.MODE == 'lsgan':
                    gen_cost = tf.reduce_mean((disc_fake - 1) ** 2)
                    disc_cost = (tf.reduce_mean((disc_real - 1) ** 2) + tf.reduce_mean((disc_fake - 0) ** 2)) / 2.


                else:
                    raise Exception()

                gen_costs.append(gen_cost)
                disc_costs.append(disc_cost)

        gen_cost = tf.add_n(gen_costs) / len(DEVICES)
        disc_cost = tf.add_n(disc_costs) / len(DEVICES)

        if config.MODE == 'wgan':
            gen_train_op = tf.train.RMSPropOptimizer(
                learning_rate=5e-5).minimize(gen_cost, var_list=lib.params_with_name('Generator'),
                                             colocate_gradients_with_ops=True)
            disc_train_op = tf.train.RMSPropOptimizer(
                learning_rate=5e-5).minimize(disc_cost, var_list=lib.params_with_name('Discriminator.'),
                                             colocate_gradients_with_ops=True)

            clip_ops = []
            for var in lib.params_with_name('Discriminator'):
                clip_bounds = [-.01, .01]
                clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
            clip_disc_weights = tf.group(*clip_ops)
        elif config.MODE == 'improved_wgan':
            gen_train_op = tf.train.AdamOptimizer(
                learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost,
                                                                   var_list=lib.params_with_name('Generator'),
                                                                   colocate_gradients_with_ops=True)

            disc_train_op = tf.train.AdamOptimizer(
                learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost,
                                                                   var_list=lib.params_with_name('Discriminator.'),
                                                                   colocate_gradients_with_ops=True)
        elif config.MODE == 'dcgan':
            gen_train_op = tf.train.AdamOptimizer(
                learning_rate=2e-4, beta1=0.5).minimize(gen_cost,
                                                        var_list=lib.params_with_name('Generator'),
                                                        colocate_gradients_with_ops=True)
            disc_train_op = tf.train.AdamOptimizer(
                learning_rate=2e-4, beta1=0.5).minimize(disc_cost,
                                                        var_list=lib.params_with_name('Discriminator.'),
                                                        colocate_gradients_with_ops=True)
        elif config.MODE == 'lsgan':
            gen_train_op = tf.train.RMSPropOptimizer(
                learning_rate=1e-4).minimize(gen_cost,
                                             var_list=lib.params_with_name('Generator'),
                                             colocate_gradients_with_ops=True)
            disc_train_op = tf.train.RMSPropOptimizer(
                learning_rate=1e-4).minimize(disc_cost,
                                             var_list=lib.params_with_name('Discriminator.'),
                                             colocate_gradients_with_ops=True)
        else:
            raise Exception()

        # For generating samples
        fixed_noise = tf.constant(np.random.normal(size=(config.batch_size, 128)).astype('float32'))
        all_fixed_noise_samples = []
        for device_index, device in enumerate(DEVICES):
            n_samples = config.batch_size / len(DEVICES)
            all_fixed_noise_samples.append(
                Generator(n_samples, noise=fixed_noise[device_index * n_samples: (device_index + 1) * n_samples]))
        all_fixed_noise_samples = tf.concat(all_fixed_noise_samples, axis=0)

        def generate_image(iteration):
            samples = session.run(all_fixed_noise_samples)
            samples = ((samples + 1.) * (255.99 / 2)).astype('int32')
            save_images(samples.reshape((config.batch_size, 3, 64, 64)),
                        './results/samples/samples_{}.png'.format(iteration))

        # Dataset iterator

        train_size = len(glob(os.path.join(config.data_dir, config.data_format)))
        train_data = make_generator(config.data_dir, train_size, batch_size=config.batch_size)

        def inf_train_gen():
            while True:
                for (images,) in train_data():
                    yield images

        # Save a batch of ground-truth samples
        _x = inf_train_gen().next()
        _x_r = session.run(real_data, feed_dict={real_data_conv: _x})
        _x_r = ((_x_r + 1.) * (255.99 / 2)).astype('int32')
        save_images(_x_r.reshape((config.batch_size, 3, 64, 64)), 'samples_groundtruth.png')
        session.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=10)
        counter = 1
        ckpt = tf.train.get_checkpoint_state(config.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(session, os.path.join(config.model_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            pp.pprint("[*] Load success ...,start the counter:{}")
            pp.pprint(counter)
        else:
            pp.pprint("[*] Load Filed ...")

        # Train loop
        for iteration in trange(config.epochs + 1):
            start_time = time.time()
            gen_loss, _ = session.run([gen_cost, gen_train_op])
            for i in range(config.dis_epochs):
                gen = inf_train_gen()
                _data = gen.next()
                _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={all_real_data_conv: _data})

                print("dis_loss: {} gen_loss: {} time_cost:{}".
                      format(_disc_cost / config.batch_size, gen_loss / config.batch_size, time.time() - start_time))

            if iteration % 100 == 99:
                save_path = config.model_dir + config.MODE
                saver.save(session, save_path, global_step=100)

            if iteration % 100 == 0:
                generate_image(iteration)

            if iteration % 1000 == 1:
                pp.pprint("iteration:{}".format(iteration))

        counter += 1


if __name__ == "__main__":
    config, unparsed = parser.parse_known_args()
    save_config(config)
    main(config)