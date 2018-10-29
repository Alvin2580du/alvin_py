import numpy as np


def imhist(im):
    # calculates normalized histogram of an image
    m, n = im.shape
    h = [0.0] * 256
    for i in range(m):
        for j in range(n):
            h[im[i, j]] += 1
    return np.array(h) / (m * n)


def cumsum(h):
    # finds cumulative sum of a numpy array, list
    return [sum(h[:i + 1]) for i in range(len(h))]


def histeq(im):
    # calculate Histogram
    h = imhist(im)
    cdf = np.array(cumsum(h))  # cumulative distribution function
    sk = np.uint8(255 * cdf)  # finding transfer function values
    s1, s2 = im.shape
    Y = np.zeros_like(im)
    # applying transfered values for each pixels
    for i in range(0, s1):
        for j in range(0, s2):
            Y[i, j] = sk[im[i, j]]
    H = imhist(Y)
    # return transformed image, original and new istogram,
    # and transform function
    return Y, h, H, sk
