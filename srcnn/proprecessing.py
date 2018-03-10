import scipy.misc
import scipy
import os
from PIL import Image
import cv2

root = "D:\\alvin_py\\srcnn\\"


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
    path = "D:\\alvin_data\\yaogan\\{}".format(step)
    files = os.listdir(path)
    for file in files:
        save_path = "./Test/yaogan/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        splitimage(os.path.join(path, file), save_path, target_size=256)


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
        save_str = "{}|{}".format(os.path.join(root, step, hr_path, files_hr_name), os.path.join(root, step, lr_path, files_lr_name))
        fw.writelines(save_str + "\n")


if __name__ == "__main__":
    method = 'four'

    if method == 'first':
        build_split_data(step='test')

    if method == 'second':
        subsample_yaogan(size=64, Hr_subsample='yaogan_64x64', origin_Hr='yaogan', step='Test')

    if method == 'third':
        upsample_yaogan(step="Test", lr_upsample_path='yaogan_lr_256x256', Hr_subsample='yaogan_64x64')

    if method == 'four':
        write2file_for_train(step='Test', hr_path='yaogan', lr_path='yaogan_lr_256x256')
