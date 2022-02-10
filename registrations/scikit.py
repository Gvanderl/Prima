from skimage.registration import phase_cross_correlation
from skimage.io import imread


def compute_registration(im1_path, im2_path):
    im1 = imread(im1_path)
    im2 = imread(im2_path)
    (y, x, z), _, _ = phase_cross_correlation(im1, im2)

    return x / im1.shape[1], y / im1.shape[0]
