import os
from utils import read_points, image_iterator

for base_image_path, other_image_path in image_iterator():
    os.system(f"greedy -d 2 -a -m NCC 2x2 -i {base_image_path} {other_image_path} -o {base_image_path.parents[0]/(other_image_path.stem+'affine.mat')} -ia-image-centers -n 100x50x10")
