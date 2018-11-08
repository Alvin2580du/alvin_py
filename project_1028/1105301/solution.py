# solution.py ---
#
# Filename: solution.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Mon Oct 15 13:06:48 2018 (-0700)
# Version:
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
# Copyright (C), Visual Computing Group @ University of Victoria.

# Code:

import time

import cv2
import numpy as np
from numpy.fft import fftshift
from numpy.fft import fft2
from numpy.fft import ifft2

# Placeholder to stop auto-syntac checking from complaining
TODO = None


def custom_blur_demo(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    cv2.imshow("custom_blur_demo", dst)


def main():
    # Read Image in grayscale and show
    img = cv2.imread("input.jpg", 0)
    cv2.imwrite("orig1.png", img)
    print(img.shape)
    print("1")
    # ----------------------------------------------------------------------
    # Create Filter
    #
    # TODO: 3 Marks: Create sharpen filter from the lecture, but with a
    # Gaussian filter form the averaging instead of the mean filter. For the
    # Gaussian filter, use a kernel with size 31x31 with sigma 5. For the unit
    # impulse set the multiplier to be 2.
    kernel = np.multiply(cv2.getGaussianKernel(31, 5), np.transpose(cv2.getGaussianKernel(31, 5)))

    # ----------------------------------------------------------------------
    # Filter with FFT
    #
    # TODO: 1 Mark: Pad filter with zeros to have the same size as the image,
    # but with the filter in the center. This creates a larger filter, that
    # effectively does the same thing as the original image.
    kernel_padded = np.zeros_like(img).astype(float)
    pad_h = (img.shape[0] - kernel.shape[0]) // 2
    pad_w = (img.shape[1] - kernel.shape[1]) // 2
    kernel_padded[pad_h:pad_h + kernel.shape[0], pad_w:pad_w + kernel.shape[1]] = kernel

    # Shift filter image to have origin on 0,0. This one is done for you. The
    # exact theory behind this was not explained in class so you may skip this
    # part. Drop by my office hours if you are interested.
    kernel_padded_shifted = fftshift(kernel_padded)

    # TODO: 1 Mark: Move all signal to Fourier space (DFT).
    img_fft = fft2(img)
    kernel_fft = fft2(kernel_padded_shifted)
    # Display signals in Fourier Space
    # I put some visualization here to help debugging :-)
    cv2.imwrite(
        "orig_fft1.png",
        np.minimum(1e-5 * np.abs(fftshift(img_fft)), 1.0) * 255.)
    cv2.imwrite(
        "filt_fft1.png",
        np.minimum(1e-1 * np.abs(fftshift(kernel_fft)), 1.0) * 255.)
    print("2")
    img_fft = fftshift(img_fft)
    kernel_fft = fftshift(kernel_fft)
    # TODO: 1 Mark: Do filtering in Fourier space
    img_filtered_fft = fftshift(fft2(kernel_padded_shifted))

    # TODO: 1 Mark: Bring back to Spatial domain (Inverse DFT)
    # TODO: 2 Marks: Throw away the imaginary part and clip between 0 and 255
    # to make it a real image.
    img_fft_copy = img_fft.copy()
    mask = np.zeros_like(img_fft_copy)
    cx = mask.shape[0] // 2
    cy = mask.shape[1] // 2
    mask[cx - 50:cx + 50, cy - 50:cy + 50] = 1
    img_fft_copy = fftshift(fftshift(img_fft_copy * mask, axes=0), axes=1)
    img_sharpened = np.real(ifft2(img_fft_copy))
    cv2.imwrite("res_fft1.png", img_sharpened.astype(np.uint8))
    print("3")
    # ----------------------------------------------------------------------
    # Filter with OpenCV
    # TODO: 1 Mark: Use padded filter and cyclic padding (wrap) to get exact results
    # TOOD: 1 Mark: Clip image for display

    img_sharpened = cv2.filter2D(img, -1, kernel_padded)
    cv2.imwrite("res_opencv1.png", img_sharpened.astype(np.uint8))
    print("4")

if __name__ == "__main__":
    main()
    exit(0)

#
# solution.py ends here
