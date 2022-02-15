import numpy as np
import pathlib
from pathlib import Path
import nibabel as nib
from PIL import Image, ImageOps
import os
from utils import read_points, image_iterator
import cv2
from compute_matrices import open_image

# TODO ITK Rigides
# TODO Multiple samples per image matching; use overlap for metrics


Path("./greedy_deform_outputs").mkdir(parents=True, exist_ok=True)
output_folder = Path("./greedy_deform_outputs").resolve()
mat_folder = Path("./greedy_output").resolve()
"""
for base_image_path, other_image_path in image_iterator():
    print("creating images...")
    print("now on ")
    print("\t", base_image_path)
    print("\t", other_image_path)

    mat_path = mat_folder / (other_image_path.stem+'affine.mat')
    wrap_path = output_folder / (other_image_path.stem+'wrap.nii.gz')
    result_path = output_folder / (other_image_path.stem+'deformed.nii.gz')

    os.system(f"greedy -d 2 -m NCC 2x2 -i {base_image_path} {other_image_path} -it {mat_path} -o {wrap_path} -n 100x50x10")

    os.system(f"greedy -d 2 -rf {wrap_path} -rm {other_image_path} {result_path} -r {mat_path}")

print("finished creating images")
"""
for base_image_path, other_image_path in image_iterator():
    if other_image_path.suffix != ".png":
        continue
    result_path = output_folder / (other_image_path.stem + 'deformed.nii.gz')

    img = nib.load(str(result_path))
    img_data = img.get_fdata()
    img_array = np.array(img_data)
    img_array = np.reshape(img_array, (img_array.shape[0], img_array.shape[1], img_array.shape[-1]))

    print(img_array.shape)
    print(img_array.max())
    im = Image.fromarray(img_array.astype('uint8'), mode='RGBA')
    im = im.rotate(-90, expand=True, fillcolor=(255, 128, 0))
    im = ImageOps.mirror(im)
    tmp = "../tmp.png"
    im.save(tmp)

    img1 = open_image(str(base_image_path), "Before")
    img2 = open_image(str(other_image_path), "After")
    img3 = open_image(tmp, "After")

    blend1 = cv2.addWeighted(img1, 0.5, img2, 0.5, 0.0)

    blend2 = cv2.addWeighted(img1, 0.5, img3, 0.5, 0.0)

    while 1:
        cv2.imshow("Before", blend1)
        cv2.imshow("After", blend2)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
