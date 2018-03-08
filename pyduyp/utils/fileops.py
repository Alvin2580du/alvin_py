import codecs
import pickle
import hashlib
from functools import cmp_to_key
import locale
import pandas as pd
import numpy as np
from glob import glob
import cv2
import tensorflow as tf
import scipy.misc
import scipy
from scipy.misc import imread, imresize, imsave
import os
from PIL import Image

from pyduyp.logger.log import log
from pyduyp.utils.utils import replace_symbol


def checkdir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)


def checkexist(file):
    return os.path.exists(file)


def readListFromTxt(filePath):
    resultList = []
    fr = codecs.open(filePath, 'r', encoding='utf-8')
    while True:
        line = fr.readline()
        if line:
            resultList.append(line.strip())
        else:
            break
    fr.close()
    return resultList


def dump(path, data):
    output = open(path, 'wb')
    pickle.dump(data, output)
    output.close()


def load(path):
    pkl_file = open(path, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data


def writeList2csv(filePath, infoList):
    fw = open(filePath, 'w', encoding='utf-8')
    for itemList in infoList:
        line = ''
        if isinstance(itemList, list):
            line = ','.join(itemList)
        if isinstance(itemList, str):
            line = itemList.strip()
        if line:
            fw.write(line + '\n')
    fw.close()


def writeContent(filePath, content):
    fw = open(filePath, 'w', encoding='utf-8')
    fw.write(content)
    fw.close()


def file2list(path, file):
    res = []
    out_f = open(os.path.join(path, "{}".format(file.split(".")[0] + "_new.txt")), 'w')

    with open(os.path.join(path, file), 'r') as f:
        lines = f.readlines()
        for line in lines:
            res.append(line.strip("\n"))
    print(res, sep='', end='\n', file=out_f)


def dele_word(file, word, include_self=True, path='./results'):
    out_f = open(os.path.join(path, "{}".format(file.split(".")[0] + "_new.txt")), 'w')
    with open(os.path.join(path, file), 'r') as f:
        lines = f.readlines()
        for line in lines:
            line2list = list(line)
            if word not in line2list:
                if include_self:
                    print(line.strip(), sep='', end="\n", file=out_f)
                else:
                    if len(line) > len(file.split(".")[0]):
                        print(line.strip(), sep='', end="\n", file=out_f)


def file2set(path, file, delete=False):
    with open(os.path.join(path, file), 'r') as f:
        lines = f.readlines()
        res = []
        for line in lines:
            res.append(line.strip())
        out = set(res)
        out_f = open(os.path.join(path, "new_" + file), 'w')
        for w in out:
            if delete:
                if len(w) > 5:
                    out_f.writelines(w + "\n")
            else:
                out_f.writelines(w + "\n")


def sortfilebylength(in_path, out_path, delete=False):
    line_l = open(out_path, "a+", encoding="utf-8")
    with open(in_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        lines.sort(key=lambda x: len(x))
        for line in lines:
            if delete:
                if len(line) > 5:
                    line_l.writelines(line.strip() + "\n")
            else:
                line_l.writelines(line.strip() + "\n")


def curlmd5(src):
    m = hashlib.md5()
    m.update(src.encode('UTF-8'))
    return m.hexdigest()


def sort_file_by_dict(data_dir, input_filename, output_filename, delete=True):
    """
    输出文件和输入文件保存在同一目录下
    :param data_dir: 数据根目录
    :param input_filename: 要排序文件的名字
    :param output_filename: 输出文件的名字
    :param delete: 是否删除标点符号
    :return: 0
    """
    locale.setlocale(locale.LC_ALL, locale='zh_CN.UTF-8')
    files = []
    line_number = 0
    inputs_dir = os.path.join(data_dir, input_filename)
    with open(inputs_dir, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            if delete:
                line_new = replace_symbol(line.replace("\n", '').lstrip().rstrip().strip())
                if len(line_new) > 1:
                    files.append(line_new)
                    line_number += 1
                    # TODO 或者可以隔500000保存一次,加快保存速度.
                    if line_number % 10000 == 0:
                        log.info("=============== process : {} ===============".format(line_number))
            else:
                line_new = line.replace("\n", '').lstrip().rstrip().strip()
                files.append(line_new)
                line_number += 1
                if line_number % 10000 == 0:
                    pass
    log.info(" Total lines : {}".format(line_number))
    b = sorted(files, key=cmp_to_key(locale.strcoll))
    df = pd.DataFrame(b)
    df.columns = ['message']
    output_dir = os.path.join(data_dir, output_filename)
    log.info("Save file : {}".format(output_dir))
    df.to_csv(output_dir, index=None)


# crop
def crop(x, wrg, hrg, is_random=False, row_index=0, col_index=1):
    h, w = x.shape[row_index], x.shape[col_index]
    if is_random:
        h_offset = int(np.random.uniform(0, h - hrg) - 1)
        w_offset = int(np.random.uniform(0, w - wrg) - 1)
        return x[h_offset: hrg + h_offset, w_offset: wrg + w_offset]
    else:
        h_offset = int(np.floor((h - hrg) / 2.))
        w_offset = int(np.floor((w - wrg) / 2.))
        h_end = h_offset + hrg
        w_end = w_offset + wrg
        return x[h_offset: h_end, w_offset: w_end]


def get_loader(root, batch_size, ext="png", seed=None):
    paths = glob("{}/*.{}".format(root, ext))
    tf_decode = tf.image.decode_png
    img = Image.open(paths[0])
    w, h = img.size
    shape = [h, w, 3]
    img.close()
    filename_queue = tf.train.string_input_producer(list(paths), shuffle=False, seed=seed)
    reader = tf.WholeFileReader()
    filename, data = reader.read(filename_queue)
    image = tf_decode(data, channels=3)
    image.set_shape(shape)

    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size

    queue = tf.train.shuffle_batch([image], batch_size=batch_size, num_threads=4, capacity=capacity,
                                   min_after_dequeue=min_after_dequeue, name='synthetic_inputs')
    return tf.to_float(queue)


def read_batch_data(batch_size=32, shuffle=False):
    data_dir = "D:\\SRGAN-master\\asset\\data\\yaogan\\test\\"
    assert os.path.exists(data_dir)
    files_dir = glob(os.path.join(data_dir, "*.png"))
    if len(files_dir) == 0:
        raise Exception("No Data ")

    files_dir2arr = np.array(files_dir)
    global indices
    indices = np.arange(len(files_dir2arr))

    for start_idx in range(0, len(files_dir2arr) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield files_dir2arr[excerpt]


def read_images(inputs_list):
    results = []
    for image in inputs_list:
        data = cv2.imread(image)
        results.append(data)

    data2arr = np.array(results, dtype=np.float32)
    return data2arr


def opencv_resize_small_bicubic(inputs, times=2):
    results = []
    for arr in inputs:
        shape = arr.shape
        new_shape = (int(shape[0] / times), int(shape[1] / times))
        res = cv2.resize(arr, new_shape, interpolation=cv2.INTER_CUBIC)
        results.append(res)
    return np.array(results)


def opencv_resize_larger_images(inputs, times=2):
    results = []
    for arr in inputs:
        shape = arr.shape
        new_shape = (int(shape[0] * times), int(shape[1] * times))
        res = cv2.resize(arr, new_shape, interpolation=cv2.INTER_NEAREST)
        results.append(res)
    return np.array(results)


def resize_images2same_size(path):
    data_dir = "D:\\Other\\yaoganshuju\\yaogan\\{}".format(path)
    save_path = "D:\\SRGAN-master\\asset\\data\\yaogan\\{}\\".format(path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    files_dir = glob(os.path.join(data_dir, "*.png"))
    for file in files_dir:
        file_name = os.path.basename(file)
        data = cv2.imread(file)
        new_data = cv2.resize(data, (176, 197), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(save_path, "{}".format(file_name)), new_data)
    print("Total preprocess :{}".format(len(files_dir)))


def srgan_imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img


def fun():
    test_image = "D:\\SRGAN-master\\asset\\data\\yaogan\\test\\4_0.png"
    data = cv2.imread(test_image)
    data_small = cv2.resize(data, (128, 128))
    cv2.imwrite("D:\\SRGAN-master\\asset\\data\\4_0_small.png", data_small)
    data_larger = cv2.resize(data, (2048, 2048))
    cv2.imwrite("D:\\SRGAN-master\\asset\\data\\4_0_larger.png", data_larger)


def get_imgs_fn(file_name, path):
    return scipy.misc.imread(path + file_name, mode='RGB')


def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=384, hrg=384, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x


def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[96, 96], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x


def splitimage(src, dst, target_size=256):
    img = Image.open(src)
    w, h = img.size
    rownum, colnum = int(w / target_size), int(h / target_size)
    if rownum <= h and colnum <= w:
        base_name, ext = os.path.basename(src).split(".")[0], os.path.basename(src).split(".")[1]
        num = 0
        rowheight = h // rownum
        colwidth = w // colnum
        for r in range(rownum):
            for c in range(colnum):
                box = (c * colwidth, r * rowheight, (c + 1) * colwidth, (r + 1) * rowheight)
                save_path = os.path.join(dst, "{}_{}.{}".format(base_name, num, ext))
                new_image = img.crop(box)
                new_image.save(save_path)
                num = num + 1

        print('图片切割完毕，共生成 %s 张小图片。' % num)
    else:
        print('不合法的行列切割参数！')


def build_split_data(step='train'):
    path = "D:\\SuperResolution_WIth_GAN\\data2017\\yaogan_1\\{}".format(step)
    files = os.listdir(path)
    for file in files:
        save_path = "D:\\SuperResolution_WIth_GAN\\data2017\\yaogan\\images\\{}".format(step)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        splitimage(os.path.join(path, file), save_path, target_size=256)


def make_x4datasets():
    path = "D:\\SuperResolution_WIth_GAN\\data2017\\yaogan\\images\\test"
    new_path = "D:\\SuperResolution_WIth_GAN\\data2017\\yaogan\\images\\test_x4"
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    images = os.listdir(path)
    num = 0
    for image in images:
        img = imread(os.path.join(path, image))
        new_img = imresize(img, [img.shape[0] * 4, img.shape[1] * 4])
        save_path = os.path.join(new_path, "{}_x4.jpg".format(image.split(".")[0]))
        imsave(save_path, new_img)
        if num % 50 == 1:
            print("执行到： {}".format(num))
        num += 1


def fun1(batch_size=1):
    root = "D:\\SRGAN-master\\png\\"
    paths = glob("{}/*.png".format(root))
    filename_queue = tf.train.string_input_producer(list(paths), shuffle=False, seed=None)
    reader = tf.WholeFileReader()
    filename, data = reader.read(filename_queue)
    print(data)
    tf_decode = tf.image.decode_png
    image = tf_decode(data, channels=3)
    min_after_dequeue = 5000
    capacity = min_after_dequeue + 3 * batch_size
    queue = tf.train.shuffle_batch([image], batch_size=1, num_threads=4, capacity=capacity,
                                   min_after_dequeue=min_after_dequeue, name='synthetic_inputs')
    queue = tf.image.crop_to_bounding_box(queue, 50, 25, 128, 128)
    print(queue)


def copy_images(path_in, path_out):
    from shutil import copy
    i = 0
    limit = 100
    for file in os.listdir(path_in):
        file_name = os.path.join(path_in, file)
        copy(file_name, path_out)
        if i > limit:
            break
        i += 1
