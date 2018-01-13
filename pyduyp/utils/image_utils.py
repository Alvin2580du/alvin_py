import scipy.misc

import numpy as np


def imread(path, is_grayscale=False):
    if is_grayscale:
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(x[j:j + crop_h, i:i + crop_w], [resize_h, resize_w])


def transform(image, input_height, input_width, resize_height=64, resize_width=64, is_crop=True):
    if is_crop:
        cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image) / 127.5 - 1.


def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, is_crop=True,
              is_grayscale=False):
    image = imread(image_path, is_grayscale)
    return transform(image, input_height, input_width, resize_height, resize_width, is_crop)


def inverse_transform(images):
    return (images + 1.) / 2.


def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for bc, image in enumerate(images):
        i = bc % size[1]
        j = bc // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
