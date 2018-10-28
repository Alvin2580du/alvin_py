import sys
import cImage
from plotfreqs import *


def oldList(origImage, w, h):
    oldList = []
    for x in range(w):
        for y in range(h):
            p = origImage.getPixel(x, y)
            oldGrey = p.getRed()
            oldList += [oldGrey]
    return oldList


def sortList(oldList):
    helperList = [0] * 256
    for i in range(256):
        if i in oldList:
            helperList[i] = helperList[i] + oldList.count(i)
    for a in range(256):
        if a > 0:
            helperList[a] = helperList[a - 1] + helperList[a]
    return helperList


def turnGrey(oldList, cdfList, origImage):
    w = origImage.getWidth()
    h = origImage.getHeight()
    area = w * h
    newImage = cImage.EmptyImage(w, h)
    newGreyList = []
    minGrey = min(oldList)
    for j in range(256):
        newGrey = (cdfList[j] - minGrey) * 255 // (area - minGrey)
        newGreyList += [newGrey]
    for x in range(w):
        for y in range(h):
            myPixel = origImage.getPixel(x, y)
            oldGrey = myPixel.getRed()
            myNewGrey = newGreyList[oldGrey]
            newPixel = cImage.Pixel(myNewGrey, myNewGrey, myNewGrey)
            newImage.setPixel(x, y, newPixel)
    return newImage


def main():
    origImage = cImage.FileImage(sys.argv[1])
    w = origImage.getWidth()
    h = origImage.getHeight()
    origWindow = cImage.ImageWin("Original Window", w, h)
    origImage.draw(origWindow)
    newWindow = cImage.ImageWin("New Window", w, h)
    aList = oldList(origImage, w, h)
    cdfList = sortList(aList)
    newImage = turnGrey(aList, cdfList, origImage)
    newImage.draw(newWindow)
    origWindow.exitOnClick()
    plotFreqs(aList, oldList(newImage, w, h), cdfList)


main()
