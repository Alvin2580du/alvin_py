import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm


def integral(img):
    # 积分图
    integ_graph = np.zeros((img.shape[0], img.shape[1]), dtype=np.int32)
    for x in range(img.shape[0]):
        sum_clo = 0
        for y in range(img.shape[1]):
            sum_clo = sum_clo + img[x][y]
            integ_graph[x][y] = integ_graph[x - 1][y] + sum_clo
    return integ_graph


def Haar(interM, weigth, height, size=1, deep=2):
    # 计算Haar特征
    dst = []
    for i in range(height - deep + 1):
        dst.append([0 for x in range(weigth - size)])
        for j in range(weigth - 2 * size + 1):
            if j == 0 and i == 0:
                whithe = int(interM[i + deep - 1][j + size - 1])
            elif i != 0 and j == 0:
                whithe = int(interM[i + deep - 1][j + size - 1]) - int(interM[i - 1][j + size - 1])
            elif i == 0 and j != 0:
                whithe = int(interM[i + deep - 1][j + size - 1]) - int(interM[i + 1][j - 1])
            else:
                whithe = int(interM[i + deep - 1][j + size - 1]) + int(interM[i - 1][j - 1]) - int(
                    interM[i + 1][j - 1]) - int(interM[i - 1][j + size - 1])
            _i = i
            _j = j + size
            if _i == 0:
                black = int(interM[_i + deep - 1][_j + size - 1]) - int(interM[_i + 1][_j - 1])
            else:
                black = int(interM[_i + deep - 1][_j + size - 1]) + int(interM[_i - 1][_j - 1]) - int(
                    interM[_i + 1][_j - 1]) - int(interM[_i - 1][_j + size - 1])
            dst[i][j] = black - whithe
    return dst


def build():
    image_dir = ['./lfw1000', './nonface']
    for d in image_dir:
        for file in tqdm(os.listdir(d)):
            file_name = os.path.join(d, file)
            if d == "./lfw1000":
                img = plt.imread(file_name, format='pgm')
            else:
                img = plt.imread(file_name, format='jpg')
                img = img[:, :, 0]
            print("{},{}".format(file, img.shape))
            _w, _h = img.shape[0], img.shape[1]
            in_b = integral(img)
            Haar_b = Haar(in_b, _h, _w)
            dst = []
            for x in range(len(Haar_b)):
                dst.append([])
                for y in range(len(Haar_b[0])):
                    if Haar_b[x][y] > 0:
                        dst[x].append(255)
                    else:
                        dst[x].append(0)
            haar_features = np.array(dst).astype('uint8')
            save_path = "./{}Save".format(d)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            im = Image.fromarray(haar_features)
            im.save(os.path.join(save_path, "{}".format(file)))


if __name__ == '__main__':
    build()
