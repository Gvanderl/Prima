import numpy as np
import cv2
from skimage import filters
import json
from config import data_folder
from os import listdir

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


def read_points(path):
    with open(path) as f:
        points = json.load(f)
        points[0] = [tuple(x) for x in points[0]]
        points[1] = [tuple(x) for x in points[1]]
    return points


def image_iterator():
    samples = [d for d in listdir(data_folder) if "." not in d]
    for sample in samples:
        img_folder = data_folder / sample / "thumbnails"
        base_image_path = img_folder / (sample + ".png")
        assert base_image_path.exists(), f"{base_image_path} does not exist"

        for other_image in listdir(img_folder):
            other_image_path = img_folder / other_image
            if other_image_path == base_image_path:
                continue
            yield base_image_path, other_image_path
