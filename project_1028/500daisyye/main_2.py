import cImage
from collections import Counter
from plotfreqs import plotFreqs
from tqdm import trange


def step_1(image):
    newly_created = []
    w = image.getWidth()
    h = image.getHeight()
    for x in range(w):
        for y in range(h):
            p = image.getPixel(x, y)
            newly_created += [p.getRed()]
    return newly_created


def makeGrey(imageFile):
    oldImage = cImage.FileImage(imageFile)
    width = oldImage.getWidth()
    hight = oldImage.getHeight()
    myimagewindow = cImage.ImageWin("Image Process", width, hight)

    oldImage.draw(myimagewindow)
    newImage = cImage.EmptyImage(width, hight)
    out = []
    for row in range(hight):
        for col in range(width):
            originalPixel = oldImage.getPixel(col, row)
            for x in originalPixel:
                out.append(x)
            aveRGB = originalPixel.getRed() + originalPixel.getGreen() + originalPixel.getBlue() // 3
            newImage.setPixel(col, row, cImage.Pixel(aveRGB, aveRGB, aveRGB))

    newImage.setPosition(width + 1, 0)
    newImage.draw(myimagewindow)
    myimagewindow.exitOnClick()
    return out, width, hight


def get_help_list(oldList):
    help_list = [0] * 256
    for x, y in Counter(oldList).most_common(300):
        help_list[x] = y
    return help_list


def cum_list(help_list):
    print(help_list)
    cdf = [0] * 256  # len(hist) is 256
    cdf[0] = help_list[0]
    for a in range(1, 256):
        cdf[a] = cdf[a - 1] + help_list[a]
    # cdf = [ele * 255 / cdf[-1] for ele in cdf]
    return cdf


def equalized(cdf_list, width, hight):
    newGrey_list = []
    for j in trange(256):
        newGrey = (cdf_list[j] - min(cdf_list)) // (width * hight - min(cdf_list)) * 255
        newGrey_list.append(newGrey)
    return newGrey_list


res, width, hight = makeGrey("./images/car.gif")
print(width, hight)
# one
df_res = get_help_list(res)
# third
cdf_list = cum_list(df_res)
print(cdf_list)
# two
after_equalize = equalized(cdf_list=cdf_list, width=width, hight=hight)
print(after_equalize)
plotFreqs(df_res, after_equalize, cdf_list)
