import numpy as np
import tensorflow as tf
import glob, os, re
import scipy.io
import pickle
import time
import math

from pyduyp.logger.log import log

DATA_PATH = "./data/test/"

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


#
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

        tensor = tf.add(tensor, input_tensor)
        return tensor, weights


def get_img_list(data_path):
    l = glob.glob(os.path.join(data_path, "*"))
    l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
    train_list = []
    for f in l:
        if os.path.exists(f):
            if os.path.exists(f[:-4] + "_2.mat"): train_list.append([f, f[:-4] + "_2.mat", 2])
            if os.path.exists(f[:-4] + "_3.mat"): train_list.append([f, f[:-4] + "_3.mat", 3])
            if os.path.exists(f[:-4] + "_4.mat"): train_list.append([f, f[:-4] + "_4.mat", 4])
    return train_list


def get_train_list_for_png(data_path):
    l = glob.glob(os.path.join(data_path, "*.bmp"))

    train_list = []
    for f in l:
        train_list.append(f)
    return train_list


def get_test_image(test_list, offset, batch_size):
    target_list = test_list[offset:offset + batch_size]
    input_list = []
    gt_list = []
    scale_list = []
    for pair in target_list:
        print(pair[1])
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
        input_list.append(input_img)
        gt_list.append(gt_img)
        scale_list.append(pair[2])
    return input_list, gt_list, scale_list


def test_VDSR_with_sess(epoch, ckpt_path, data_path, sess):
    log.debug("{}".format(data_path))
    folder_list = glob.glob(os.path.join(data_path, 'Set*'))
    log.debug("folder_list: {}".format(folder_list))
    log.debug("{}".format(ckpt_path))
    saver.restore(sess, ckpt_path)

    psnr_dict = {}
    for folder_path in folder_list:
        psnr_list = []
        img_list = get_train_list_for_png(folder_path)
        log.debug("{}".format(img_list))
        for i in range(len(img_list)):
            input_list, gt_list, scale_list = get_test_image(img_list, i, 1)
            input_y = input_list[0]
            gt_y = gt_list[0]
            start_t = time.time()
            img_vdsr_y = sess.run([output_tensor], feed_dict={
                input_tensor: np.resize(input_y, (1, input_y.shape[0], input_y.shape[1], 1))})
            img_vdsr_y = np.resize(img_vdsr_y, (input_y.shape[0], input_y.shape[1]))
            end_t = time.time()
            log.debug("{}".format("end_t", end_t, "start_t", start_t))
            log.debug("{}".format("time consumption", end_t - start_t))
            log.debug("{}".format("image_size", input_y.shape))
            psnr_bicub = psnr(input_y, gt_y, scale_list[0])
            psnr_vdsr = psnr(img_vdsr_y, gt_y, scale_list[0])
            log.debug("{}".format("PSNR: bicubic %f\t VDSR %f" % (psnr_bicub, psnr_vdsr)))
            psnr_list.append([psnr_bicub, psnr_vdsr, scale_list[0]])
        psnr_dict[os.path.basename(folder_path)] = psnr_list
    with open('psnr/%s' % os.path.basename(ckpt_path), 'wb') as f:
        pickle.dump(psnr_dict, f)


def test_VDSR(epoch, ckpt_path, data_path):
    with tf.Session() as sess:
        test_VDSR_with_sess(epoch, ckpt_path, data_path, sess)


if __name__ == '__main__':
    model_list = sorted(glob.glob("./checkpoints/VDSR_adam_epoch_*"))
    # model_list = [fn for fn in model_list if not os.path.basename(fn).endswith("meta")]
    log.info("{}".format(model_list))
    with tf.Session() as sess:
        input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 1))
        shared_model = tf.make_template('shared_model', model)
        output_tensor, weights = shared_model(input_tensor)

        saver = tf.train.Saver(weights)
        tf.initialize_all_variables().run()
        for model_ckpt in model_list:
            log.debug("{}".format(model_ckpt))
            epoch = int(model_ckpt.split('epoch_')[-1].split('.ckpt')[0])

            test_VDSR_with_sess(80, model_ckpt, DATA_PATH, sess)
