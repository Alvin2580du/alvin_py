import tensorflow as tf
import argparse
import math
import numpy as np
import os
import re
import glob
import scipy.io
from sklearn.utils import shuffle
import signal
import sys
import threading

from pyduyp.logger.log import log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_PATH = "D:\\alvin_py\\srcnn\\Train\\yaogan"
IMG_SIZE = (64, 64)
BATCH_SIZE = 16
BASE_LR = 0.0001
LR_RATE = 0.1
LR_STEP_SIZE = 120
MAX_EPOCH = 120

USE_QUEUE_LOADING = True

model_path = "./models"


def psnr(target, ref, scale):
    # assume RGB image
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
        tensor = None

        # conv_00_w = tf.get_variable("conv_00_w", [3,3,1,64], initializer=tf.contrib.layers.xavier_initializer())
        conv_00_w = tf.get_variable("conv_00_w", [3, 3, 1, 64],
                                    initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9)))
        conv_00_b = tf.get_variable("conv_00_b", [64], initializer=tf.constant_initializer(0))
        weights.append(conv_00_w)
        weights.append(conv_00_b)
        tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_tensor, conv_00_w, strides=[1, 1, 1, 1], padding='SAME'), conv_00_b))

        for i in range(18):
            conv_w = tf.get_variable("conv_%02d_w" % (i + 1), [3, 3, 64, 64], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
            conv_b = tf.get_variable("conv_%02d_b" % (i + 1), [64], initializer=tf.constant_initializer(0))
            weights.append(conv_w)
            weights.append(conv_b)
            tensor = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b))

        conv_w = tf.get_variable("conv_20_w", [3, 3, 64, 1], initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / 64)))
        conv_b = tf.get_variable("conv_20_b", [1], initializer=tf.constant_initializer(0))
        weights.append(conv_w)
        weights.append(conv_b)
        tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1, 1, 1, 1], padding='SAME'), conv_b)

        tensor = tf.add(tensor, input_tensor)
        return tensor, weights


def get_train_list(data_path):
    l = glob.glob(os.path.join(data_path, "*"))
    l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
    train_list = []
    for f in l:
        if os.path.exists(f):
            if os.path.exists(f[:-4] + "_2.mat"): train_list.append([f, f[:-4] + "_2.mat"])
            if os.path.exists(f[:-4] + "_3.mat"): train_list.append([f, f[:-4] + "_3.mat"])
            if os.path.exists(f[:-4] + "_4.mat"): train_list.append([f, f[:-4] + "_4.mat"])
    return train_list


def get_train_list_for_png(data_path):
    l = glob.glob(os.path.join(data_path, "*.png"))

    train_list = []
    for f in l:
        train_list.append(f)
    return train_list


def get_image_batch(train_list, offset, batch_size):
    target_list = train_list[offset:offset + batch_size]
    input_list = []
    gt_list = []
    cbcr_list = []
    for pair in target_list:
        input_img = scipy.io.loadmat(pair[1])['patch']
        gt_img = scipy.io.loadmat(pair[0])['patch']
        input_list.append(input_img)
        gt_list.append(gt_img)
    input_list = np.array(input_list)
    input_list.resize([BATCH_SIZE, IMG_SIZE[1], IMG_SIZE[0], 1])
    gt_list = np.array(gt_list)
    gt_list.resize([BATCH_SIZE, IMG_SIZE[1], IMG_SIZE[0], 1])
    return input_list, gt_list, np.array(cbcr_list)


def get_test_image(test_list, offset, batch_size):
    target_list = test_list[offset:offset + batch_size]
    input_list = []
    gt_list = []
    for pair in target_list:
        mat_dict = scipy.io.loadmat(pair[1])
        input_img = None
        if mat_dict.has_key("img_2"):
            input_img = mat_dict["img_2"]
        elif mat_dict.has_key("img_3"):
            input_img = mat_dict["img_3"]
        elif mat_dict.has_key("img_4"):
            input_img = mat_dict["img_4"]
        else:
            continue
        gt_img = scipy.io.loadmat(pair[0])['img_raw']
        input_list.append(input_img[:, :, 0])
        gt_list.append(gt_img[:, :, 0])
    return input_list, gt_list


if __name__ == '__main__':
    train_list = get_train_list_for_png(DATA_PATH)
    log.debug("{}".format(train_list))

    if not USE_QUEUE_LOADING:
        log.debug("not use queue loading, just sequential loading...")
        ### WITHOUT ASYNCHRONOUS DATA LOADING ###
        train_input = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1))
        train_gt = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 1))

    ### WITHOUT ASYNCHRONOUS DATA LOADING ###
    else:
        ### WITH ASYNCHRONOUS DATA LOADING ###
        train_input_single = tf.placeholder(tf.float32, shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
        train_gt_single = tf.placeholder(tf.float32, shape=(IMG_SIZE[0], IMG_SIZE[1], 1))
        q = tf.FIFOQueue(10000, [tf.float32, tf.float32], [[IMG_SIZE[0], IMG_SIZE[1], 1], [IMG_SIZE[0], IMG_SIZE[1], 1]])
        enqueue_op = q.enqueue([train_input_single, train_gt_single])
        train_input, train_gt = q.dequeue_many(BATCH_SIZE)
        log.debug("{}, {}".format(train_input, train_gt))
    ### WITH ASYNCHRONOUS DATA LOADING ###

    shared_model = tf.make_template('shared_model', model)
    log.debug("{}".format(shared_model))
    train_output, weights = shared_model(train_input)
    loss = tf.reduce_sum(tf.nn.l2_loss(tf.subtract(train_output, train_gt)))
    for w in weights:
        loss += tf.nn.l2_loss(w) * 1e-4
    tf.summary.scalar("loss", loss)

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(BASE_LR, global_step * BATCH_SIZE, len(train_list) * LR_STEP_SIZE, LR_RATE, staircase=True)
    tf.summary.scalar("learning rate", learning_rate)

    optimizer = tf.train.AdamOptimizer(learning_rate)  # tf.train.MomentumOptimizer(learning_rate, 0.9)
    opt = optimizer.minimize(loss, global_step=global_step)

    saver = tf.train.Saver(weights, max_to_keep=0)

    shuffle(train_list)
    config = tf.ConfigProto()
    # config.operation_timeout_in_ms=10000

    with tf.Session(config=config) as sess:
        # TensorBoard open log with "tensorboard --logdir=logs"
        if not os.path.exists('logs'):
            os.mkdir('logs')
        merged = tf.summary.merge_all()
        file_writer = tf.summary.FileWriter('logs', sess.graph)

        tf.initialize_all_variables().run()

        if model_path:
            saver.restore(sess, model_path)

        ### WITH ASYNCHRONOUS DATA LOADING ###
        def load_and_enqueue(coord, file_list, enqueue_op, train_input_single, train_gt_single, idx=0, num_thread=1):
            count = 0
            length = len(file_list)
            try:
                while not coord.should_stop():
                    i = count % length
                    input_img = scipy.io.loadmat(file_list[i][1])['patch'].reshape([IMG_SIZE[0], IMG_SIZE[1], 1])
                    gt_img = scipy.io.loadmat(file_list[i][0])['patch'].reshape([IMG_SIZE[0], IMG_SIZE[1], 1])
                    sess.run(enqueue_op, feed_dict={train_input_single: input_img, train_gt_single: gt_img})
                    count += 1
            except Exception as e:
                log.warning("{}".format("stopping...", idx, e))

        ### with asynchronous data loading  ###
        threads = []

        def signal_handler(signum, frame):
            sess.run(q.close(cancel_pending_enqueues=True))
            coord.request_stop()
            coord.join(threads)
            log.info("{}".format("Done"))
            sys.exit(1)


        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal_handler)

        if USE_QUEUE_LOADING:
            # create threads
            num_thread = 20
            coord = tf.train.Coordinator()
            for i in range(num_thread):
                length = len(train_list) / num_thread
                t = threading.Thread(target=load_and_enqueue, args=(
                    coord, train_list[i * length:(i + 1) * length], enqueue_op, train_input_single, train_gt_single, i,
                    num_thread))
                threads.append(t)
                t.start()
            print("num thread:", len(threads))

            for epoch in range(0, MAX_EPOCH):
                max_step = len(train_list) // BATCH_SIZE
                for step in range(max_step):
                    _, l, output, lr, g_step, summary = sess.run(
                        [opt, loss, train_output, learning_rate, global_step, merged])
                    log.info("[epoch %2.4f] loss %.4f\t lr %.5f" % (
                    epoch + (float(step) * BATCH_SIZE / len(train_list)), np.sum(l) / BATCH_SIZE, lr))
                    file_writer.add_summary(summary, step + epoch * max_step)
                # print "[epoch %2.4f] loss %.4f\t lr %.5f\t norm %.2f"%(epoch+(float(step)*BATCH_SIZE/len(train_list)), np.sum(l)/BATCH_SIZE, lr, norm)
                saver.save(sess, "./checkpoints/VDSR_adam_epoch_%03d.ckpt" % epoch, global_step=global_step)
        else:
            for epoch in range(0, MAX_EPOCH):
                for step in range(len(train_list) // BATCH_SIZE):
                    offset = step * BATCH_SIZE
                    input_data, gt_data, cbcr_data = get_image_batch(train_list, offset, BATCH_SIZE)
                    feed_dict = {train_input: input_data, train_gt: gt_data}
                    _, l, output, lr, g_step = sess.run([opt, loss, train_output, learning_rate, global_step],
                                                        feed_dict=feed_dict)
                    print("[epoch %2.4f] loss %.4f\t lr %.5f" % (
                    epoch + (float(step) * BATCH_SIZE / len(train_list)), np.sum(l) / BATCH_SIZE, lr))
                    del input_data, gt_data, cbcr_data

                saver.save(sess, "./checkpoints/VDSR_const_clip_0.01_epoch_%03d.ckpt" % epoch, global_step=global_step)
