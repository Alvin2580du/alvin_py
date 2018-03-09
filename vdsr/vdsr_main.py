import tensorflow as tf
import math
import numpy as np
import os
from tqdm import trange
from pyduyp.logger.log import log
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMG_SIZE = (64, 64)
batch_size = 16
base_lr = 0.0001
lr_rate = 0.1
lr_step_size = 120
max_epoch = 120


def psnr(target, ref, scale):
    target_data = np.array(target)
    target_data = target_data[scale:-scale, scale:-scale]

    ref_data = np.array(ref)
    ref_data = ref_data[scale:-scale, scale:-scale]

    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(1.0 / rmse)


def model(input_tensor):
    with tf.device("/cpu:0"):
        weights = []
        conv_00_w = tf.get_variable("conv_00_w", [3, 3, 1, 64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9)))
        log.info("conv_00_w: {}".format(conv_00_w))
        conv_00_b = tf.get_variable("conv_00_b", [64], initializer=tf.constant_initializer(0))
        weights.append(conv_00_w)
        weights.append(conv_00_b)
        tensor = tf.nn.relu(
            tf.nn.bias_add(tf.nn.conv2d(input_tensor, conv_00_w, strides=[1, 1, 1, 1], padding='SAME'), conv_00_b))

        for i in range(18):
            conv_w = tf.get_variable("conv_%02d_w" % (i + 1), [3, 3, 64, 64],
                                     initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
            conv_b = tf.get_variable("conv_%02d_b" % (i + 1), [64], initializer=tf.constant_initializer(0))
            weights.append(conv_w)
            weights.append(conv_b)
            tensor = tf.nn.relu(
                tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b))

        conv_w = tf.get_variable("conv_20_w", [3, 3, 64, 1],
                                 initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
        conv_b = tf.get_variable("conv_20_b", [1], initializer=tf.constant_initializer(0))
        weights.append(conv_w)
        weights.append(conv_b)
        tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b)
        log.info("tensor:{}".format(tensor))
        tensor = tf.add(tensor, input_tensor)
        log.info("tensor:{}".format(tensor))
        return tensor, weights


def get_image_batch_forpng(start_idx, batch_size):
    hr_paths = []
    lr_paths = []
    with open("yaogan.txt", 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            lr, hr = line.split('|')[1].strip(), line.split('|')[0].strip()
            hr_paths.append(hr)
            lr_paths.append(lr)

    excerpt = slice(start_idx, start_idx + batch_size)
    yield hr_paths[excerpt], lr_paths[excerpt]


def read_data2arr(inputs_list):
    out = []
    for image in inputs_list:
        image = cv2.imread(image)
        out.append(image)
    out2arr = np.array(out)
    log.info("{}".format(out2arr.shape))
    return out2arr


if __name__ == '__main__':
    train_list_length = 2700
    train_input = tf.placeholder(tf.float32, shape=(batch_size, IMG_SIZE[0], IMG_SIZE[1], 1))
    train_gt = tf.placeholder(tf.float32, shape=(batch_size, IMG_SIZE[0], IMG_SIZE[1], 1))

    shared_model = tf.make_template('shared_model', model)
    train_output, weights = shared_model(train_input)
    loss = tf.reduce_sum(tf.nn.l2_loss(tf.subtract(train_output, train_gt)))
    for w in weights:
        loss += tf.nn.l2_loss(w) * 1e-4
    tf.summary.scalar("loss", loss)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(base_lr, global_step * batch_size, train_list_length * lr_step_size,
                                               lr_rate, staircase=True)
    tf.summary.scalar("learning rate", learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    opt = optimizer.minimize(loss, global_step=global_step)
    saver = tf.train.Saver(weights, max_to_keep=0)

    config = tf.ConfigProto()

    with tf.Session(config=config) as sess:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        merged = tf.summary.merge_all()
        file_writer = tf.summary.FileWriter('logs', sess.graph)
        tf.initialize_all_variables().run()
        for epoch in trange(0, max_epoch):
            log.info(" ................... Start Training ...................")
            batch_count = train_list_length // batch_size
            for bc in range(batch_count):
                offset = bc * batch_size
                for hr, lr in get_image_batch_forpng(bc, batch_size):
                    input_data, gt_data = read_data2arr(lr), read_data2arr(hr)

                    feed_dict = {train_input: input_data, train_gt: gt_data}
                    _, l, output, lr, g_step = sess.run([opt, loss, train_output, learning_rate, global_step],
                                                        feed_dict=feed_dict)
                    loginfo = "epoch/bc:{}/{}, loss: {},lr: {}".format(epoch, bc, np.sum(l) / batch_size, lr)
                    log.info("{}".format(loginfo))

                if epoch % 50 == 0:
                    saver.save(sess, "./checkpoints/vdsr_{}_{}.ckpt".format(epoch, bc), global_step=epoch)
