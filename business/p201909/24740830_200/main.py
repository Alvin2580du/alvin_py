import cv2
import numpy as np
from scipy.signal import convolve2d


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))


classifier = cv2.CascadeClassifier(
    'C:\\Program Files\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml')

filepath1 = './c4_new/2019-9-2-11-37-25.jpg'
filepath2 = './c4_new/2019-9-2-11-33-2.jpg'


img = cv2.imread(filepath1)  # 读取图片
h1, w1 = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换灰色
# OpenCV人脸识别分类器
color = (0, 255, 0)  # 定义绘制颜色
# 调用识别人脸
faceRects = classifier.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(5, 5))
img1_list = []
x_num = 0
if len(faceRects):  # 大于0则检测到人脸
    for faceRect in faceRects:  # 单独框出每一张人脸
        x, y, w, h = faceRect
        face = img[y:y + h, x:x + w:]
        face = cv2.resize(face, (256, 256))
        cv2.imwrite('face1_{}.jpg'.format(x_num), face)
        img1_list.append(face)
        x_num += 1


img2 = cv2.imread(filepath2)  # 读取图片
h2, w2 = img2.shape[:2]

gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 转换灰色
# OpenCV人脸识别分类器
color = (0, 255, 0)  # 定义绘制颜色
# 调用识别人脸
faceRects2 = classifier.detectMultiScale(gray2, scaleFactor=1.15, minNeighbors=5, minSize=(5, 5))

img2_list = []
y_num = 0
if len(faceRects2):  # 大于0则检测到人脸
    for faceRect2 in faceRects2:  # 单独框出每一张人脸
        x, y, w, h = faceRect2
        face2 = img[y:y + h, x:x + w:]
        face2 = cv2.resize(face2, (256, 256))
        cv2.imwrite('face2_{}.jpg'.format(y_num), face2)
        img2_list.append(face2)
        y_num += 1


for x_ids, x in enumerate(img1_list):
    for y_ids, y in enumerate(img2_list):
        ssim = compute_ssim(x[:, :, 0], y[:, :, 0])
        if ssim > 0.5:
            print('{}, {}, 是同一个人, 相似度为：{}'.format(x_ids, y_ids, ssim))
        else:
            print('{}, {}, 不是同一个人, 相似度为：{}'.format(x_ids, y_ids, ssim))
