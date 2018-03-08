import os
import glob
import h5py
import scipy.misc
import scipy.ndimage
import numpy as np
import tensorflow as tf

from pyduyp.logger.log import log

FLAGS = tf.app.flags.FLAGS


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


def prepare_data(sess, dataset):
    """
  Args:
    dataset: choose train dataset or test dataset

    For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
  """
    if FLAGS.is_train:
        filenames = os.listdir(dataset)
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


def input_setup(sess, config):
    if config.is_train:
        data = prepare_data(sess, dataset="Train\\{}".format(config.train_data))
    else:
        data = prepare_data(sess, dataset="Test\\{}".format(config.test_data))

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
                    sub_label = label_[x + int(padding):x + int(padding) + config.label_size, y + int(padding):y + int(padding) + config.label_size]

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
