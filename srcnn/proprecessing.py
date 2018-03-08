import scipy.misc
import scipy
import os
from PIL import Image


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


if __name__ == "__main__":
    method = 'first'
    if method == 'first':
        build_split_data(step='test')
