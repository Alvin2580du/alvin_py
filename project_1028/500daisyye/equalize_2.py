import cv2


def df(img):
    values = [0] * 256
    for i in range(len(img)):
        for j in img[i]:
            values[j] += 1
    return values


def cdf(hist):  # cumulative distribution frequency
    cdf = [0] * len(hist)  # len(hist) is 256
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i - 1] + hist[i]
    # Now we normalize the histogram
    cdf = [ele * 255 / cdf[-1] for ele in cdf]  # What your function h was doing before
    return cdf


def equalize_image(image):
    image_df = df(img)
    print(len(image_df), image_df)

    my_cdf = cdf(image_df)
    print(len(my_cdf), my_cdf)
    # use linear interpolation of cdf to find new pixel values. Scipy alternative exists
    import numpy as np
    # image_equalized = np.interp(image, range(0, 256), my_cdf)
    image_equalized = get_newGrey_image(image,  my_cdf, row*col )

    print(len(image_equalized), image_equalized)
    return image_equalized
# newGray = (cdforigGrayVal – cdfmin) // (area of the image – cdfmin) * 255

def get_newGrey_image(oldList, cdfList, area):
    res = []
    for j in range(256):
        newGrey = (cdfList[j] - min(oldList)) // (area - min(oldList)) * 255
        res.append(newGrey)
    return res


img = cv2.imread("test.png", 0)
print(img.shape)
row, col = img.shape[:2]

eq = equalize_image(img)
cv2.imwrite('equalized.png', eq)
