import cImage
from plotfreqs import plotFreqs
import os
import sys

"""
Step 1: At the beginning, we can run through each pixel in the original image to get its pixel values of its corresponding
color and then record the values one by one to a newly-created list. Since the gray-scale values will range from 0 to 255
 and within a gray-scale image Red = Blue = Green, we do not need to worry about the color difference but just focus on
  its pixel values.

"""


def step_1(image):
    newly_created = []
    w = image.getWidth()
    h = image.getHeight()
    for x in range(w):
        for y in range(h):
            p = image.getPixel(x, y)
            newly_created += [p.getRed()]
    return newly_created


"""
Step 2: As what we did in the last homework, we call the plotFreqs function (module already imported).
For now, use the values in the list we just created in the first step.

"""


def step_2(newly_created):
    plotFreqs(newly_created)


"""
(a) Now, we can use the same way we used in Counting Sort problem to first make a helper list of
 all zeros with the size equal to 256.
(b) Second, we can use a for-loop to iterate over the list created in the first step to count the occurrence of
 each gray-scale value in the original image into the 256 slots of the helper list.

(c) After that, we can produce a running accumulative sum of the counts from loop1 into the helper list.

(d) Finally, we can create the cumulative distribution from the histogram according to the values in the
final-version helper list.

"""


def step_3_get_help_list(oldList):
    help_list = [0] * 256
    for i in range(0, 256):
        help_list[i] = oldList.count(i)
    return help_list


def get_cumulative_from_help_list(help_list):
    cumulative_from_help_list = help_list
    for a in range(1, 256):
        help_list[a] = help_list[a - 1] + help_list[a]
    return cumulative_from_help_list


"""
Step 4: According to the property of the cumulative distribution, the first value of it will be the
cdfmin (the minimum non-zero count from the cumulative distribution).
(a) Now we create a new list with empty elements. 
(b) Then, we use a new for-loop to iterate over the helper list created in Step 2. 
(c) We will change each value in the helper list created in Step 2 to a new value based on the formula:
 newGray = (cdforigGrayVal – cdfmin) // (area of the image – cdfmin) * 255. (Here we use "//" because the
 final value for newGray needs to be an int).
(d) When we change each value, we will append each value one by one to the new list which we just created. 
By this way, we can map gray-scale values from the original image to a new value stored in the new list.

"""


def get_newGrey_image(oldList, cdfList, area):
    res = []
    for j in range(256):
        newGrey = (cdfList[j] - min(oldList)) * 255 // (area - min(oldList))
        res.append(newGrey)
    return res


def get_draw_image(newGreyList, origImage):
    w = origImage.getWidth()
    h = origImage.getHeight()

    after_image = cImage.EmptyImage(w, h)
    for i in range(w):
        for j in range(h):
            pixels = origImage.getPixel(i, j)
            pre_grey = pixels.getRed()
            after_grey = newGreyList[pre_grey]
            after_pixels = cImage.Pixel(after_grey, after_grey, after_grey)
            after_image.setPixel(i, j, after_pixels)
    return after_image


def build_step_4(file_name="./images/mozambiqueColor.gif"):
    origImage = cImage.FileImage(file_name)  # mozambiqueColor
    w = origImage.getWidth()
    h = origImage.getHeight()
    origWindow = cImage.ImageWin("Original Window", w, h)
    origImage.draw(origWindow)
    newWindow = cImage.ImageWin("New Window", w, h)
    rawImage_list = step_1(origImage)
    s3_list = step_3_get_help_list(rawImage_list)
    print(len(s3_list), s3_list[100:120])
    cdfList = get_cumulative_from_help_list(s3_list)
    print(len(cdfList), cdfList[100:120])
    after_image_list = get_newGrey_image(rawImage_list, cdfList, w * h)
    print(len(after_image_list), after_image_list[100:120])
    newImage = get_draw_image(after_image_list, origImage)
    newImage.draw(newWindow)

    return s3_list, cdfList, after_image_list
"""

Step 5: 
(a) We can now create a new image with the same height and length of the original image. 
(d) Then we can run through each pixel in the original image and change each pixel value to the
corresponding value in the new list based on the order.

Step 6: Finally, we can set the new pixel values into the new empty image we created in step 5 to
 finalize the histogram equalization.

"""


def step_6():
    # before equalization and after , cumulative plot
    file_name = sys.argv[1]
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    s3_list, cdfList, after_image_list = build_step_4(file_name=file_name)
    df1 = pd.DataFrame(s3_list)
    df1.to_excel("df1.xlsx", index=None)

    df2 = pd.DataFrame(cdfList)
    df2.to_excel("df2.xlsx", index=None)

    df3 = pd.DataFrame(after_image_list)
    df3.to_excel("df3.xlsx", index=None)

    xPos = np.arange(256)

    plt.figure(figsize=(6, 3))
    plt.bar(xPos, s3_list, align='center')
    plt.show()
    plt.close()

    plt.figure(figsize=(6, 3))
    plt.bar(xPos, cdfList, align='center')
    plt.show()
    plt.close()

    plt.figure(figsize=(6, 3))
    plt.bar(xPos, after_image_list, align='center')
    plt.show()
    plt.close()

step_6()

