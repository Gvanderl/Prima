import cv2 as cv
from pathlib import Path
from os import listdir
import logging

logging.basicConfig(level=logging.DEBUG)

data_folder = Path("~/data/prima/data").expanduser().resolve()
samples = [d for d in listdir(data_folder) if d[0] != "."]
logging.info(f"Found samples {samples}")


def match_images(im1, im2):
    pass


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
        match_images(im1, im2)


