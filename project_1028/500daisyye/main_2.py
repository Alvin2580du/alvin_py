import cImage
from collections import Counter
from plotfreqs import plotFreqs


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
    cdf = [0] * 256  # len(hist) is 256
    cdf[0] = help_list[0]
    for a in range(1, 256):
        cdf[a] = cdf[a - 1] + help_list[a]
    cdf = [ele * 255 / cdf[-1] for ele in cdf]
    return cdf


def equalized(cdf_list, origin, width, hight):
    newGrey_list = []
    for j in range(256):
        newGrey = (cdf_list[j] - min(origin)) * 255 // ((width * hight - min(origin)))
        newGrey_list.append(newGrey)
    return newGrey_list


res, width, hight = makeGrey("./images/crowd.gif")
print(width, hight)
# one
df_res = get_help_list(res)
# third
cdf_list = cum_list(df_res)
# two
after_equalize = equalized(cdf_list=cdf_list, origin=res, width=width, hight=hight)

plotFreqs(df_res, after_equalize, cdf_list)
