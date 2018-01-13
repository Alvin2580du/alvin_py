import argparse
import os
import math
import json
import logging
import numpy as np
from PIL import Image
import tensorflow as tf
from glob import glob
from tensorflow.contrib import slim
from tqdm import trange
from collections import deque
from datetime import datetime
import time
import shutil
import pandas as pd
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# now_time = datetime.now().strftime('%m_%d_%H:%M')
# log_file = open("log_{}".format(now_time), "w")
# sys.stdout = log_file


def str2bool(v):
    return v.lower() in ('true', '1')


arg = argparse.ArgumentParser()
arg.add_argument('--dataset', type=str, default='celeba_1')
arg.add_argument('--batch_size', type=int, default=16)
arg.add_argument('--is_train', type=str2bool, default=True)

arg.add_argument('--max_step', type=int, default=500000)
arg.add_argument('--model_name', type=str, default='model_7_22')
arg.add_argument('--input_scale_size', type=int, default=64, help='input image will be resized with given value')

arg.add_argument('--conv_hidden_num', type=int, default=128, choices=[64, 128], help='n in the paper')
arg.add_argument('--z_num', type=int, default=128, choices=[64, 128])
arg.add_argument('--grayscale', type=str2bool, default=False)
arg.add_argument('--num_worker', type=int, default=4)

# Training / test parameters
arg.add_argument('--optimizer', type=str, default='adam')
arg.add_argument('--lr_update_step', type=int, default=100000, choices=[100000, 75000])
arg.add_argument('--d_lr', type=float, default=0.00008)
arg.add_argument('--g_lr', type=float, default=0.00008)
arg.add_argument('--lr_lower_boundary', type=float, default=0.00002)
arg.add_argument('--beta1', type=float, default=0.5)
arg.add_argument('--beta2', type=float, default=0.999)
arg.add_argument('--gamma', type=float, default=0.5)
arg.add_argument('--lambda_k', type=float, default=0.001)
arg.add_argument('--use_gpu', type=str2bool, default=True)

# Misc
arg.add_argument('--log_step', type=int, default=50)
arg.add_argument('--save_step', type=int, default=1000)
arg.add_argument('--num_log_samples', type=int, default=3)
arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
arg.add_argument('--log_dir', type=str, default='began_logs')
arg.add_argument('--data_dir', type=str, default='/home/dms/alvin_data/')
arg.add_argument('--test_data_path', type=str, default=None,
                 help='directory with images which will be used in test sample generation')
arg.add_argument('--sample_per_image', type=int, default=64,
                 help='# of sample per image during test sample generation')
arg.add_argument('--random_seed', type=int, default=123)

# is visiuliztion
arg.add_argument('--is_visiuliztion', type=str2bool, default=False)
arg.add_argument('--visu_epoch', type=int, default=2)
arg.add_argument('--visu_path', type=str, default='/home/dms/began_visu')


def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]


def get_conv_shape(tensor, data_format):
    shape = int_shape(tensor)
    # always return [N, H, W, C]
    if data_format == 'NCHW':
        return [shape[0], shape[2], shape[3], shape[1]]
    elif data_format == 'NHWC':
        return shape


def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])


def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])


def reshape(x, h, w, c, data_format):
    if data_format == 'NCHW':
        x = tf.reshape(x, [-1, c, h, w])
    else:
        x = tf.reshape(x, [-1, h, w, c])
    return x


def resize_nearest_neighbor(x, new_size, data_format):
    if data_format == 'NCHW':
        x = nchw_to_nhwc(x)
        x = tf.image.resize_nearest_neighbor(x, new_size)
        x = nhwc_to_nchw(x)
    else:
        x = tf.image.resize_nearest_neighbor(x, new_size)
    return x


def upscale_1(x, scale):
    _, h, w, _ = x.get_shape().as_list()
    return tf.image.resize_nearest_neighbor(x, (h * scale, w * scale))


def upscale(x, scale, data_format):
    _, h, w, _ = get_conv_shape(x, data_format)
    return resize_nearest_neighbor(x, (h * scale, w * scale), data_format)


def prepare_dirs_and_logger(config):
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    config.data_path = os.path.join(config.data_dir, config.dataset)
    config.model_dir = os.path.join(config.log_dir, config.model_name)

    for path in [config.log_dir, config.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)


def get_time():
    return datetime.now().strftime("%m%d_%H%M%S")


def save_config(config):
    param_path = os.path.join(config.model_dir, "params.json")
    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def rank(array):
    return len(array.shape)


def make_grid(tensor, nrow=4, padding=2, normalize=False, scale_each=False):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h:h + h_width, w:w + w_width] = tensor[k]
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=4, padding=2, normalize=False, scale_each=False):
    ndarr = make_grid(tensor, nrow=nrow, padding=padding, normalize=normalize, scale_each=scale_each)
    im = Image.fromarray(ndarr)
    im.save(filename)


def my_save_image(tensor, filename, nrow=4, padding=2, normalize=False, scale_each=False):
    ndarr = make_grid(tensor, nrow=nrow, padding=padding, normalize=normalize, scale_each=scale_each)
    im = Image.fromarray(ndarr)
    im.save(filename)


def to_nhwc(image, data_format):
    if data_format == 'NCHW':
        new_image = nchw_to_nhwc(image)
    else:
        new_image = image
    return new_image


def to_nchw_numpy(image):
    if image.shape[3] in [1, 3]:
        new_image = image.transpose([0, 3, 1, 2])
    else:
        new_image = image
    return new_image


def norm_img(image, data_format=None):
    image = image / 127.5 - 1.
    if data_format:
        image = to_nhwc(image, data_format)
    return image


def denorm_img(norm, data_format):
    return tf.clip_by_value(to_nhwc((norm + 1) * 127.5, data_format), 0, 255)


def slerp(val, low, high):
    """Code from https://github.com/soumith/dcgan.torch/issues/14"""
    omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high  # L'Hopital's rule/LERP
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def get_loader(root, batch_size, scale_size, data_format, split=None, is_grayscale=False, seed=None):
    dataset_name = os.path.basename(root)

    for ext in ["jpg", "png"]:
        paths = glob("{}/*.{}".format(root, ext))

        if ext == "jpg":
            tf_decode = tf.image.decode_jpeg
        elif ext == "png":
            tf_decode = tf.image.decode_png

        if len(paths) != 0:
            break
    with Image.open(paths[0]) as img:
        w, h = img.size
        shape = [h, w, 3]

    filename_queue = tf.train.string_input_producer(list(paths), shuffle=False, seed=seed)
    reader = tf.WholeFileReader()
    filename, data = reader.read(filename_queue)
    image = tf_decode(data, channels=3)

    if is_grayscale:
        image = tf.image.rgb_to_grayscale(image)
    image.set_shape(shape)

    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size

    queue = tf.train.shuffle_batch(
        [image], batch_size=batch_size,
        num_threads=4, capacity=capacity,
        min_after_dequeue=min_after_dequeue, name='synthetic_inputs')

    if dataset_name in ['CelebA']:
        queue = tf.image.crop_to_bounding_box(queue, 50, 25, 128, 128)
        queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])
    elif dataset_name in ['celeba_1']:
        queue = tf.image.crop_to_bounding_box(queue, 50, 25, 128, 128)
        queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])
    else:
        queue = tf.image.resize_nearest_neighbor(queue, [scale_size, scale_size])

    if data_format == 'NCHW':
        queue = tf.transpose(queue, [0, 3, 1, 2])
    elif data_format == 'NHWC':
        pass
    else:
        raise Exception("[!] Unkown data_format: {}".format(data_format))

    return tf.to_float(queue)


def GeneratorCNN(z, hidden_num, data_format, reuse):
    with tf.variable_scope("G", reuse=reuse) as vs:
        num_output = np.prod([8, 8, hidden_num])
        x = slim.fully_connected(z, num_output, activation_fn=None)
        x = reshape(x, 8, 8, hidden_num, data_format)
        x1 = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x2 = slim.conv2d(x1, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x3 = upscale(x2, 2, data_format)
        x4 = slim.conv2d(x3, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x5 = slim.conv2d(x4, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x6 = upscale(x5, 2, data_format)
        x7 = slim.conv2d(x6, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x8 = slim.conv2d(x7, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x9 = upscale(x8, 2, data_format)
        x10 = slim.conv2d(x9, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)

        layer_split = tf.split(x10, 128, axis=1)
        layer_split_selected = layer_split[0]
        for i in range(1, 127):
            feature_map = layer_split[i]
            layer_split_selected = tf.concat([layer_split_selected, feature_map], axis=1)
        layer_split2change = layer_split[127]
        layer_split2change = tf.subtract(layer_split2change, layer_split2change)
        new_inputs = tf.concat([layer_split_selected, layer_split2change], axis=1)

        x11 = slim.conv2d(new_inputs, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        out = slim.conv2d(x11, 3, 3, 1, activation_fn=None, data_format=data_format)
        visiu = [x1, x2, x4, x5, x7, x8, x10, x11, out]
    variables = tf.contrib.framework.get_variables(vs)
    return out, variables, visiu


def DiscriminatorCNN(x, input_channel, z_num, hidden_num, data_format):
    with tf.variable_scope("D") as vs:
        # Encoder
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x = slim.conv2d(x, 128, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x = slim.conv2d(x, 128, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x = slim.conv2d(x, 128, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
        x = slim.conv2d(x, 256, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x = slim.conv2d(x, 256, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x = slim.conv2d(x, 256, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
        x = slim.conv2d(x, 384, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x = slim.conv2d(x, 384, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x = slim.conv2d(x, 384, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
        x = slim.conv2d(x, 512, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x = slim.conv2d(x, 512, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x = tf.reshape(x, [-1, np.prod([8, 8, 512])])

        z_out = slim.fully_connected(x, z_num, activation_fn=None)
        x = slim.fully_connected(x, z_num, activation_fn=None)

        # Decoder
        num_output = np.prod([8, 8, hidden_num])
        x = slim.fully_connected(x, num_output, activation_fn=None)

        x = reshape(x, 8, 8, hidden_num, data_format)

        # 1
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x = upscale(x, 2, data_format)

        # 2
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x = upscale(x, 2, data_format)

        # 3
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x = upscale(x, 2, data_format)

        # 4
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
        out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None, data_format=data_format)
    variables = tf.contrib.framework.get_variables(vs)
    return out, z_out, variables


class Trainer(object):
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader
        self.dataset = config.dataset

        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size

        self.step = tf.Variable(0, name='step', trainable=False)

        self.g_lr = tf.Variable(config.g_lr, name='g_lr')
        self.d_lr = tf.Variable(config.d_lr, name='d_lr')

        self.g_lr_update = tf.assign(self.g_lr, tf.maximum(self.g_lr * 0.5, config.lr_lower_boundary),
                                     name='g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, tf.maximum(self.d_lr * 0.5, config.lr_lower_boundary),
                                     name='d_lr_update')

        self.gamma = config.gamma
        self.lambda_k = config.lambda_k

        self.z_num = config.z_num
        self.conv_hidden_num = config.conv_hidden_num
        self.input_scale_size = config.input_scale_size

        self.model_dir = config.model_dir
        self.model_name = config.model_name

        self.use_gpu = config.use_gpu
        self.data_format = config.data_format

        _, height, width, self.channel = \
            get_conv_shape(self.data_loader, self.data_format)
        self.repeat_num = int(np.log2(height)) - 2

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step
        self.lr_update_step = config.lr_update_step

        self.is_train = config.is_train
        self.build_model()

        self.saver = tf.train.Saver(max_to_keep=100)
        self.summary_writer = tf.summary.FileWriter(self.model_dir)

        sv = tf.train.Supervisor(logdir=self.model_dir,
                                 is_chief=True,
                                 saver=self.saver,
                                 summary_op=None,
                                 summary_writer=self.summary_writer,
                                 save_model_secs=3600,
                                 global_step=self.step,
                                 ready_for_local_init_op=None)

        gpu_options = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     gpu_options=gpu_options)

        self.sess = sv.prepare_or_wait_for_session(config=sess_config)

        if not self.is_train:
            # dirty way to bypass graph finilization error
            g = tf.get_default_graph()
            g._finalized = False
            self.build_test_model()
        self.visu_path = config.visu_path
        self.visu_epoch = config.visu_epoch

    def train(self):
        z_fixed = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))

        x_fixed = self.get_image_from_loader()
        save_image(x_fixed, '{}/x_fixed.png'.format(self.model_dir))

        prev_measure = 1
        measure_history = deque([0] * self.lr_update_step, self.lr_update_step)

        for step in trange(self.start_step, self.max_step):
            fetch_dict = {
                "k_update": self.k_update,
                "measure": self.measure,
            }
            if step % self.log_step == 0:
                fetch_dict.update({
                    "summary": self.summary_op,
                    "g_loss": self.g_loss,
                    "d_loss": self.d_loss,
                    "k_t": self.k_t,
                })
            result = self.sess.run(fetch_dict)

            measure = result['measure']
            measure_history.append(measure)

            if step % self.log_step == 0:
                self.summary_writer.add_summary(result['summary'], step)
                self.summary_writer.flush()

                g_loss = result['g_loss']
                d_loss = result['d_loss']
                k_t = result['k_t']

                print("[{}/{}] Loss_D: {:.6f} Loss_G: {:.6f} measure: {:.4f}, k_t: {:.4f}".
                      format(step, self.max_step, d_loss, g_loss, measure, k_t))

            if step % (self.log_step * 10) == 0:
                x_fake = self.generate(z_fixed, self.model_dir, idx=step)
                self.autoencode(x_fixed, self.model_dir, idx=step, x_fake=x_fake)

            if step % self.lr_update_step == self.lr_update_step - 1:
                self.sess.run([self.g_lr_update, self.d_lr_update])
                # cur_measure = np.mean(measure_history)
                # if cur_measure > prev_measure * 0.99:
                # prev_measure = cur_measure

    def build_model(self):
        self.x = self.data_loader
        x = norm_img(self.x)

        self.z = tf.random_uniform(
            (tf.shape(x)[0], self.z_num), minval=-1.0, maxval=1.0)
        self.k_t = tf.Variable(0., trainable=False, name='k_t')

        G, self.G_var, self.visulize = GeneratorCNN(self.z, self.conv_hidden_num, self.data_format, reuse=False)

        d_out, self.D_z, self.D_var = DiscriminatorCNN(tf.concat([G, x], 0), self.channel, self.z_num,
                                                       self.conv_hidden_num, self.data_format)
        AE_G, AE_x = tf.split(d_out, 2)

        self.G = denorm_img(G, self.data_format)
        self.AE_G, self.AE_x = denorm_img(AE_G, self.data_format), denorm_img(AE_x, self.data_format)

        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))

        g_optimizer, d_optimizer = optimizer(self.g_lr), optimizer(self.d_lr)

        self.d_loss_real = tf.reduce_mean(tf.abs(AE_x - x))
        self.d_loss_fake = tf.reduce_mean(tf.abs(AE_G - G))

        self.d_loss = self.d_loss_real - self.k_t * self.d_loss_fake
        self.g_loss = tf.reduce_mean(tf.abs(AE_G - G))

        d_optim = d_optimizer.minimize(self.d_loss, var_list=self.D_var)
        g_optim = g_optimizer.minimize(self.g_loss, global_step=self.step, var_list=self.G_var)

        self.balance = self.gamma * self.d_loss_real - self.g_loss
        self.measure = self.d_loss_real + tf.abs(self.balance)

        with tf.control_dependencies([d_optim, g_optim]):
            self.k_update = tf.assign(
                self.k_t, tf.clip_by_value(self.k_t + self.lambda_k * self.balance, 0, 1))

        self.summary_op = tf.summary.merge([
            tf.summary.image("G", self.G),
            tf.summary.image("AE_G", self.AE_G),
            tf.summary.image("AE_x", self.AE_x),

            tf.summary.scalar("loss/d_loss", self.d_loss),
            tf.summary.scalar("loss/d_loss_real", self.d_loss_real),
            tf.summary.scalar("loss/d_loss_fake", self.d_loss_fake),
            tf.summary.scalar("loss/g_loss", self.g_loss),
            tf.summary.scalar("misc/measure", self.measure),
            tf.summary.scalar("misc/k_t", self.k_t),
            tf.summary.scalar("misc/d_lr", self.d_lr),
            tf.summary.scalar("misc/g_lr", self.g_lr),
            tf.summary.scalar("misc/balance", self.balance),
        ])

    def build_test_model(self):
        with tf.variable_scope("test") as vs:
            # Extra wgan_ops for interpolation
            z_optimizer = tf.train.AdamOptimizer(0.0001)

            self.z_r = tf.get_variable("z_r", [self.batch_size, self.z_num], tf.float32)
            self.z_r_update = tf.assign(self.z_r, self.z)

        G_z_r, _, self.visulize_2 = GeneratorCNN(self.z_r, self.conv_hidden_num, self.data_format, reuse=True)

        with tf.variable_scope("test") as vs:
            self.z_r_loss = tf.reduce_mean(tf.abs(self.x - G_z_r))
            self.z_r_optim = z_optimizer.minimize(self.z_r_loss, var_list=[self.z_r])

        test_variables = tf.contrib.framework.get_variables(vs)
        self.sess.run(tf.variables_initializer(test_variables))

    def generate(self, inputs, root_path=None, path=None, idx=None, save=True):
        x = self.sess.run(self.G, {self.z: inputs})
        if path is None and save:
            path = os.path.join(root_path, '{}_G.png'.format(idx))
            save_image(x, path)
            print("[*] Samples saved: {}".format(path))
        return x

    def autoencode(self, inputs, path, idx=None, x_fake=None):
        items = {
            'real': inputs,
            'fake': x_fake,
        }
        for key, img in items.items():
            if img is None:
                continue
            if img.shape[3] in [1, 3]:
                img = img.transpose([0, 3, 1, 2])

            x_path = os.path.join(path, '{}_D_{}.png'.format(idx, key))
            x = self.sess.run(self.AE_x, {self.x: img})
            save_image(x, x_path)
            print("[*] Samples saved: {}".format(x_path))

    def encode(self, inputs):
        if inputs.shape[3] in [1, 3]:
            inputs = inputs.transpose([0, 3, 1, 2])
        return self.sess.run(self.D_z, {self.x: inputs})

    def decode(self, z):
        return self.sess.run(self.AE_x, {self.D_z: z})

    def interpolate_G(self, real_batch, step=0, root_path='.', train_epoch=0):
        batch_size = len(real_batch)
        half_batch_size = int(batch_size / 2)

        self.sess.run(self.z_r_update)
        tf_real_batch = to_nchw_numpy(real_batch)
        for i in trange(train_epoch):
            z_r_loss, _ = self.sess.run([self.z_r_loss, self.z_r_optim], {self.x: tf_real_batch})
        z = self.sess.run(self.z_r)

        z1, z2 = z[:half_batch_size], z[half_batch_size:]
        real1_batch, real2_batch = real_batch[:half_batch_size], real_batch[half_batch_size:]

        generated = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(z1, z2)])
            z_decode = self.generate(z, save=False)
            generated.append(z_decode)

        generated = np.stack(generated).transpose([1, 0, 2, 3, 4])
        for idx, img in enumerate(generated):
            save_image(img, os.path.join(root_path, 'test{}_interp_G_{}.png'.format(step, idx)), nrow=10)

        all_img_num = np.prod(generated.shape[:2])
        batch_generated = np.reshape(generated, [all_img_num] + list(generated.shape[2:]))
        save_image(batch_generated, os.path.join(root_path, 'test{}_interp_G.png'.format(step)), nrow=10)

    def interpolate_D(self, real1_batch, real2_batch, step=0, root_path="."):
        real1_encode = self.encode(real1_batch)
        real2_encode = self.encode(real2_batch)

        decodes = []
        for idx, ratio in enumerate(np.linspace(0, 1, 10)):
            z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(real1_encode, real2_encode)])
            z_decode = self.decode(z)
            decodes.append(z_decode)

        decodes = np.stack(decodes).transpose([1, 0, 2, 3, 4])
        for idx, img in enumerate(decodes):
            img = np.concatenate([[real1_batch[idx]], img, [real2_batch[idx]]], 0)
            save_image(img, os.path.join(root_path, 'test{}_interp_D_{}.png'.format(step, idx)), nrow=10 + 2)

    def test(self):
        now_time = datetime.now().strftime('%m.%d_%H%M%S')
        root_path = "./began_test/{}_{}".format(self.config.model_name, now_time)  # self.model_dir
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        all_G_z = None
        test_epoch = 1
        for step in trange(test_epoch):
            z_fixed = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))
            G_z = self.generate(z_fixed, path=os.path.join(root_path, "G_z_{}.png".format(step)))

            if all_G_z is None:
                all_G_z = G_z
                my_save_image(all_G_z, '{}/all_G_z.png'.format(root_path), nrow=16)
            else:
                all_G_z = np.concatenate([all_G_z, G_z])
                my_save_image(all_G_z, '{}/all_G_z.png'.format(root_path), nrow=16)

    def visualization(self, task, layer_number=9):
        # 1
        samples_dir = os.path.join(self.visu_path, "sample_{}".format(self.model_name))
        z_fixed_save_dir = os.path.join(self.visu_path, "z_csv_{}".format(self.model_name))
        saved_csv_dir = os.path.join(self.visu_path, "all_csv_{}".format(self.model_name))
        # 2
        sample_contact_dir = os.path.join(self.visu_path, "contact_csv_{}".format(self.model_name))
        # 3
        compute_contact_dir = os.path.join(self.visu_path, "compute_contact_{}".format(self.model_name))
        # 4
        top_nine_show_save_dir = os.path.join(self.visu_path, "show_top9_{}".format(self.model_name))

        dir_list = [samples_dir, z_fixed_save_dir, saved_csv_dir, sample_contact_dir, compute_contact_dir,
                    top_nine_show_save_dir]
        for dir in dir_list:
            if not os.path.exists(dir):
                os.makedirs(dir)

        def count_feature_map_number(layer_number):
            if layer_number < 8:
                return 128
            elif layer_number == 8:
                return 3
            else:
                raise NotImplementedError

        if task == "first":
            for step in trange(self.visu_epoch):
                z_fixed = np.random.uniform(-1, 1, size=(self.batch_size, self.z_num))
                z_fixed_pd = pd.DataFrame(z_fixed)

                z_fixed_pd.to_csv(os.path.join(z_fixed_save_dir, "{}_epoch_z.csv".format(step)))
                G_z = self.G.eval(session=self.sess, feed_dict={self.z: z_fixed})
                save_image(G_z, '{}/{}_samples.png'.format(samples_dir, step))
                for layers_num in range(layer_number):
                    layers = self.visulize[layers_num].eval(session=self.sess, feed_dict={self.z: z_fixed})
                    for layer in layers:
                        feature_map_id = 0
                        for feature_map in layer:
                            feature_map_reshape = feature_map.reshape((1, feature_map.shape[0] * feature_map.shape[1]))
                            feature_map_pd = pd.DataFrame(feature_map_reshape)
                            feature_map_pd.insert(0, 'row_max', feature_map_pd.apply(lambda x: x.max(), axis=1))
                            feature_map_pd.insert(1, 'step', step)
                            new = feature_map_pd.sort_values(by='row_max', ascending=False)
                            layers_path = os.path.join(saved_csv_dir,
                                                       "layer_{}/fm_{}".format(layers_num, feature_map_id))

                            if not os.path.exists(layers_path):
                                os.makedirs(layers_path)
                            # only save step and row max
                            # step_and_rowmax = new.iloc[0:2, 0:2]
                            new.to_csv(os.path.join(layers_path, "{}_step.csv".format(step)), index=None)
                            feature_map_id += 1

        elif task == "second":
            start_time = time.time()
            for ln in trange(layer_number):
                layer_dir = os.path.join(saved_csv_dir, "layer_{}".format(ln))
                feature_map_number = count_feature_map_number(ln)
                for fm in range(feature_map_number):
                    feature_map_dir = os.path.join(layer_dir, "fm_{}".format(fm))
                    csv_list = glob((os.path.join(feature_map_dir, "*.csv")))
                    for csv in csv_list:
                        new_dir = os.path.join(sample_contact_dir, "layer_{}/fm_{}".format(ln, fm))
                        if not os.path.exists(new_dir):
                            os.makedirs(new_dir)
                        contact_path = os.path.join(new_dir, '{}_fm.csv'.format(fm))
                        fr = open(csv, 'r').read()
                        with open(contact_path, 'a+') as f:
                            f.write(fr)

        elif task == "third":
            start_time = time.time()
            for ln in trange(layer_number):
                layer_dir = os.path.join(sample_contact_dir, "layer_{}".format(ln))
                feature_map_number = count_feature_map_number(ln)
                for fm in range(feature_map_number):
                    feature_map_dir = os.path.join(layer_dir, "fm_{}".format(fm))

                    csv_list = glob((os.path.join(feature_map_dir, "*.csv")))
                    for csv in csv_list:
                        skip_rows = range(2, self.visu_epoch * 2, 2)
                        feature_map = pd.read_csv(csv, skiprows=skip_rows)
                        new_feature_map = feature_map.sort_values(by='row_max', ascending=False)
                        top_nine = new_feature_map.head(9)
                        epoch = top_nine.ix[:, 1]

                        compute_save_path = os.path.join(compute_contact_dir, "layer_{}/fm_{}".format(ln, fm))
                        if not os.path.exists(compute_save_path):
                            os.makedirs(compute_save_path)

                        counter = 0
                        epoch_list = []
                        for j in range(9):
                            current_epoch = list(epoch.values)[j]
                            epoch_list.append(current_epoch)
                            image = top_nine.iloc[counter:j + 1, 0:2]
                            save_path = os.path.join(compute_save_path, "{}_epoch_top_{}.xlsx".format(current_epoch, j))
                            image.to_excel(save_path, sheet_name="Sheet1", index=None)
                            counter += 1

                        for e in epoch_list:
                            samples = glob(os.path.join(z_fixed_save_dir, "*.csv"))
                            for sample in samples:
                                sample_name = sample.split("/")[-1].split(".")[0].split("_")[0]
                                if int(sample_name) == int(e):
                                    shutil.copy(sample, compute_save_path)

        elif task == "four":
            start_time = time.time()
            for ln in trange(layer_number):
                layers_path = os.path.join(compute_contact_dir, "layer_{}".format(ln))
                feature_map_number = count_feature_map_number(ln)
                for fm in range(feature_map_number):
                    feature_map_dir = os.path.join(layers_path, "fm_{}".format(fm))
                    z_fixed_list = glob(os.path.join(feature_map_dir, "*.csv"))

                    second_z_name_list = []
                    save_top9_path = os.path.join(top_nine_show_save_dir, "layer_{}/fm_{}".format(ln, fm))

                    if not os.path.exists(save_top9_path):
                        os.makedirs(save_top9_path)
                    for second_z in z_fixed_list:
                        second_z_name = second_z.split("/")[-1].split("_")[0]
                        second_z_name_list.append(second_z_name)

                    for e in second_z_name_list:
                        show_z_read_path = os.path.join(z_fixed_save_dir, "{}_epoch_z.csv".format(e))
                        z_fixed = pd.read_csv(show_z_read_path)
                        z_fixed = z_fixed.iloc[0:1, 1:129]
                        layers = self.visulize[ln].eval(session=self.sess, feed_dict={self.z: z_fixed.values})
                        for idx1, layer in enumerate(layers):
                            for idx2, feature_map in enumerate(layer):
                                image_size = feature_map.shape[0]
                                image2plot = feature_map.reshape((image_size, image_size))
                                image2plot_re = (image2plot - image2plot.min()) / (
                                    image2plot.max() - image2plot.min()) * 255.
                                image2plot_re = cv2.resize(image2plot_re, (128, 128))
                                save_top9_image_path = os.path.join(save_top9_path, "{}_epoch_fm.jpg".format(e))
                                cv2.imwrite(save_top9_image_path, image2plot_re)

                    for e in second_z_name_list:
                        samples = glob(os.path.join(samples_dir, "*.png"))
                        for sample in samples:
                            sample_name = sample.split("/")[-1].split("_")[0]
                            if int(sample_name) == int(e):
                                shutil.copy(sample, save_top9_path)
        else:
            raise NotImplementedError

    def get_image_from_loader(self):
        x = self.data_loader.eval(session=self.sess)
        if self.data_format == 'NCHW':
            x = x.transpose([0, 2, 3, 1])
        return x


def get_config():
    config, unparsed = arg.parse_known_args()

    if config.use_gpu:
        data_format = 'NCHW'
    else:
        data_format = 'NHWC'
    setattr(config, 'data_format', data_format)
    return config, unparsed


def main(config):
    prepare_dirs_and_logger(config)
    rng = np.random.RandomState(config.random_seed)
    tf.set_random_seed(config.random_seed)

    if config.is_train:
        data_path = config.data_path
    else:
        setattr(config, 'batch_size', 1)
        if config.test_data_path is None:
            data_path = config.data_path
        else:
            data_path = config.test_data_path

    data_loader = get_loader(data_path, config.batch_size, config.input_scale_size, config.data_format)
    trainer = Trainer(config, data_loader)

    if config.is_train:
        save_config(config)
        trainer.train()
    elif config.is_visiuliztion:
        if not os.path.exists(config.visu_path):
            os.makedirs(config.visu_path)
        if not config.model_name:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        step_list = ['first', 'second', 'third', 'four']
        # step_list = ['second']
        start_time = time.time()
        for s in step_list:
            trainer.visualization(task=s)

    else:
        trainer.test()


if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
