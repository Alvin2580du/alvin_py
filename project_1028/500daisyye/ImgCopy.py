"""
This will copy the image pixel-by-pixel

   <--------width------->
   | 
   |
   height
   |
   |

In python, it is best to go row then col,
However, images mostly look at col then row

"""

import cImage
import sys
import math


def copyImg(img):
    rows = img.getHeight()
    cols = img.getWidth()

    newimg = cImage.EmptyImage(cols, rows)

    for r in range(rows):

        for c in range(cols):
            p = img.getPixel(c, r)
            newimg.setPixel(c, r, p)

    return newimg


def main():
    # Get file names
    oname = sys.argv[1]
    nname = sys.argv[2]

    # Open and read in original image
    oimg = cImage.FileImage(oname)

    # Display image
    w = oimg.getWidth()
    h = oimg.getHeight()
    owindow = cImage.ImageWin("Original Image", w, h)
    oimg.draw(owindow)

    # Copy Image
    nimg = copyImg(oimg)

    # Display new image
    w = nimg.getWidth()
    h = nimg.getHeight()
    nwindow = cImage.ImageWin("New Image", w, h)
    nimg.draw(nwindow)

    nimg.save(nname)

    nwindow.exitOnClick()


main()
