import numpy as np
import cv2
from skimage import filters


def whitewashing(im, threshold=1):
    im64 = im.astype(np.int64)
    bg = np.abs(im64[:, :, 0] - im64[:, :, 1]) < threshold  # B == G
    gr = np.abs(im64[:, :, 1] - im64[:, :, 2]) < threshold  # G == R
    rb = np.abs(im64[:, :, 0] - im64[:, :, 2]) < threshold  # R == B
    mask = np.bitwise_and(np.bitwise_and(bg, gr), rb)
    im[mask, :] = [255, 255, 255]
    return im


def grayscale(im):
    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return gray_image


def otsu(im):
    im = grayscale(im)
    return im
    return (im > filters.threshold_otsu(im)) * 255