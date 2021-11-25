import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging


def belid_match(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)
    im3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)

    show_image(im3, "Result")
    assert False

    answer = input("Continue ?")
    if answer[0] != "y":
        return


def whitewashing(im):
    im64 = im.astype(np.int64)
    threshold = 1
    bg = np.abs(im64[:, :, 0] - im64[:, :, 1]) < threshold  # B == G
    gr = np.abs(im64[:, :, 1] - im64[:, :, 2]) < threshold  # G == R
    rb = np.abs(im64[:, :, 0] - im64[:, :, 2]) < threshold  # R == B
    mask = np.bitwise_and(np.bitwise_and(bg, gr), rb)
    im[mask, :] = [255, 255, 255]
    return im


def grayscale(im):
    gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return gray_image


def show_image(im, title=None):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.imshow(im)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def baseline_match(im1_path, im2_path):
    logging.info(f"Matching {im1_path} with {im2_path}")
    im1 = cv2.imread(im1_path.as_posix())
    im2 = cv2.imread(im2_path.as_posix())
    im1, im2 = whitewashing(im1), whitewashing(im2)
    # im1, im2 = grayscale(im1), grayscale(im2)

    cv2.imwrite('im1.png', im1)
    cv2.imwrite('im2.png', im2)
    show_image(im1, im1_path.stem)
    show_image(im2, im2_path.stem)

    belid_match(im1, im2)
