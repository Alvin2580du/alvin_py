import scipy.misc
import scipy
import os
from PIL import Image
import cv2
from tqdm import tqdm
from glob import glob
import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd

root = "D:\\alvin_py\\srcnn\\"


def splitimage(src, dst, target_size=96):
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
    path = "D:\\alvin_data\\yaogan\\{}".format(step)
    files = os.listdir(path)
    for file in files:
        save_path = "./Train/yaogan_96x96/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        splitimage(os.path.join(path, file), save_path, target_size=96)


def subsample_yaogan(size=64, Hr_subsample='yaogan_64x64', origin_Hr='yaogan', step='Test'):
    save_dir = os.path.join(root, step, Hr_subsample)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    hr_dir = os.path.join(root, step, origin_Hr)
    files = os.listdir(hr_dir)
    for file in files:
        fileName = os.path.join(hr_dir, file)
        image = cv2.imread(fileName)
        imageresize = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(save_dir, file), imageresize)


def upsample_yaogan(step="Test", lr_upsample_path='yaogan_lr_256x256', Hr_subsample='yaogan_64x64'):
    lr_path = os.path.join(root, step, lr_upsample_path)
    if not os.path.exists(lr_path):
        os.makedirs(lr_path)
    files = os.listdir(os.path.join(root, step, Hr_subsample))
    for file in files:
        fileName = os.path.join(root, step, Hr_subsample, file)
        image = cv2.imread(fileName)
        imageresize = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(lr_path, file), imageresize)


def write2file_for_train(step='Test', hr_path='yaogan', lr_path='yaogan_lr_256x256'):
    fw = open(os.path.join(root, '{}\\yaogan.txt'.format(step)), 'w', encoding='utf-8')
    files_hr = os.listdir(os.path.join(root, step, hr_path))
    files_lr = os.listdir(os.path.join(root, step, lr_path))
    length = len(files_hr)

    for i in range(length):
        files_hr_name = files_hr[i]
        files_lr_name = files_lr[i]
        save_str = "{}|{}".format(os.path.join(root, step, hr_path, files_hr_name),
                                  os.path.join(root, step, lr_path, files_lr_name))
        fw.writelines(save_str + "\n")


def bsd300_to_text():
    lr_path = './Train/BSDS300/images/train_x4/'
    hr_path = './Train/BSDS300/images/train/'
    length = len(os.listdir(lr_path))
    fw = open(os.path.join(root, 'bsd300.txt'), 'w', encoding='utf-8')

    for i in range(length):
        lr_image = os.path.join(lr_path, os.listdir(lr_path)[i])
        hr_image = os.path.join(hr_path, os.listdir(hr_path)[i])

        res = "{}|{}".format(hr_image, lr_image)
        fw.writelines(res + "\n")


def celeba2text():
    path = 'D:\\alvin_data\\celeba'
    lr_path = 'D:\\alvin_data\\celeba_4x'
    if not os.path.exists(lr_path):
        os.makedirs(lr_path)

    for file in tqdm(os.listdir(path)):
        file_name = os.path.join(path, file)
        image = cv2.imread(file_name)
        imageresize = cv2.resize(image, (178 * 4, 218 * 4), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(lr_path, "4x_" + file), imageresize)


def delete():
    from pyduyp.utils.utils import is_chinese
    res = []
    with open('paper.txt', 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            for x in line:
                if is_chinese(x):
                    res.append(x)
    fw = open("paper_delet.txt", 'w', encoding='utf-8')
    fw.writelines("".join(res))


def get_celeba_names():
    fw = open(os.path.join(root, 'celeba.txt'), 'w', encoding='utf-8')
    hr_path = 'D:\\alvin_data\\celeba'
    lr_path = 'D:\\alvin_data\\celeba_4x'
    hr_images = os.listdir(hr_path)
    lr_images = os.listdir(lr_path)
    length = len(hr_images)
    for i in range(length):
        hr = os.path.join(hr_path, hr_images[i])
        lr = os.path.join(lr_path, lr_images[i])
        res = "{}|{}".format(hr, lr)
        fw.writelines(res + "\n")


def get_logs():
    losses = []
    with open('logs.txt', 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines:
            if "INFO" in line:
                loss = re.compile("loss: \d+\.\d+").findall(line)
                if loss:
                    losses.append(loss[0].split(":")[-1])
    df = pd.Series(losses)
    df.to_csv("logs.csv")


def plot_logs():
    logs = pd.read_csv("logs.csv").values
    data = [i for j in logs for i in j]
    print(data)
    index = range(0, 61900, 10)
    res = []
    for x in index:
        res.append(data[x])

    plt.figure()
    plt.plot(res)
    plt.savefig("logs.png")


def yaoganProcess():
    yaogan_path = "./Train/yaogan_96x96/"
    x_train = []
    files = os.listdir(yaogan_path)
    print(len(files))
    for file in tqdm(files):
        filename = os.path.join(yaogan_path, file)
        images = cv2.imread(filename)
        images = cv2.resize(images, (96, 96))
        x_train.append(images)

    x_train = np.array(x_train, dtype=np.float16)
    np.save('D:\\alvin_py\\srcnn\\Train\\x_train.npy', x_train)


def lfwProcess():
    persons = glob('D:\\alvin_py\\srcnn\\Train\\lfw\\*')
    paths = np.array([e for x in [glob(os.path.join(person, '*')) for person in persons] for e in x])
    np.random.shuffle(paths)
    r = int(len(paths) * 0.95)
    train_paths = paths[:r]
    test_paths = paths[r:]

    x_train = []
    for i, d in enumerate(train_paths):
        face = cv2.imread(d)
        face = cv2.resize(face, (96, 96))
        if face is None:
            continue
        x_train.append(face)
        imgpath = os.path.join('D:\\alvin_py\\srcnn\\Train\\lfw_train\\', "{}.jpg".format("{0:05d}".format(i)))
        cv2.imwrite(imgpath, face)
        print(imgpath)

    x_train = np.array(x_train, dtype=np.uint8)
    np.save('D:\\alvin_py\\srcnn\\Train\\yg_train.npy', x_train)


if __name__ == "__main__":
    method = 'yaogan'

    if method == 'first':
        build_split_data(step='train')

    if method == 'second':
        subsample_yaogan(size=64, Hr_subsample='yaogan_64x64', origin_Hr='yaogan', step='Test')

    if method == 'third':
        upsample_yaogan(step="Test", lr_upsample_path='yaogan_lr_256x256', Hr_subsample='yaogan_64x64')

    if method == 'four':
        write2file_for_train(step='Test', hr_path='yaogan', lr_path='yaogan_lr_256x256')

    if method == 'five':
        bsd300_to_text()

    if method == 'six':
        # celeba2text()
        delete()
    if method == 'senven':
        get_celeba_names()
    if method == 'plotlogs':
        plot_logs()

    if method == 'lfw':
        lfwProcess()

    if method == 'yaogan':
        yaoganProcess()