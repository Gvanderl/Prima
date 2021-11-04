import cv2 as cv
from pathlib import Path
from os import listdir
import logging
import numpy as np
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO)

data_folder = Path("~/data/prima/data").expanduser().resolve()
samples = [d for d in listdir(data_folder) if d[0] != "."]
logging.info(f"Found samples {samples}")


def belid_match(img1, img2):
    # Initiate SIFT detector
    orb = cv.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    im3 =
    img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], flags=2)

    plt.imshow(img3)
    plt.show()


for sample in samples:
    img_folder = data_folder / sample / "thumbnails"
    base_image_path = img_folder / (sample + ".png")
    assert base_image_path.exists()
    im1 = cv.imread(base_image_path.as_posix())
    for other_image in listdir(img_folder):
        other_image_path = img_folder / other_image
        if other_image_path == base_image_path:
            continue
        im2 = cv.imread(other_image_path.as_posix())
        logging.info(f"Matching {base_image_path} with {other_image_path}")
        plt.imshow(im1)
        plt.title(base_image_path.stem)
        plt.show()
        plt.imshow(im2)
        plt.title(other_image_path.stem)
        plt.show()
        belid_match(im1, im2)


