from os import listdir
import logging
from utils import image_iterator, read_points
import pandas as pd
import json
import cv2
import numpy as np
from compute_matrices import open_image
from config import data_folder, output_folder
from registrations.itk_registation import itk_registration
from registrations.scikit import compute_registration
from evaluate_distance import show_results, eval_transform
from compute_matrices import compute_transformation

logging.basicConfig(level=logging.INFO)

samples = [d for d in listdir(data_folder) if "." not in d]
logging.info(f"Found samples {samples}")

out = pd.DataFrame(columns=["base", "other", "transform"])
method = "ITK"
evaluate = True
diffs = list()

for base_image_path, other_image_path in image_iterator():
    print(f"Performing registration on {base_image_path.name} and {other_image_path.name}")
    if method == "ITK":
        transform = itk_registration(fixed_input_image=base_image_path,
                                     moving_input_image=other_image_path,
                                     output_name=other_image_path.stem)
    elif method == "computed":
        json_path = output_folder / (base_image_path.stem + "_" + other_image_path.stem + ".json")
        points = read_points(json_path)
        transform = compute_transformation(points)
    else:
        raise NameError
    # x, y = compute_registration(base_image_path, other_image_path)
    print(f"Transform is {transform}")
    tmp_df = pd.DataFrame.from_dict({
        "base": base_image_path.name,
        "other": other_image_path.name,
        "transform": [transform]
    })
    out = pd.concat([out, tmp_df])

    if evaluate:
        json_path = output_folder / (base_image_path.stem + "_" + other_image_path.stem + ".json")
        if not json_path.exists():
            continue
        trans = transform.copy()
        if method == "ITK":
            other_image = open_image(other_image_path.as_posix(), "tmp")
            cv2.destroyAllWindows()
            trans[0, 2] = trans[0, 2] * other_image.shape[1]
            trans[1, 2] = trans[1, 2] * other_image.shape[0]

        points = read_points(json_path)
        new_points, diff = eval_transform(points, trans)
        diffs.append(diff)
        show_results(base_image_path, other_image_path, trans, points, new_points)

print(f"Mean diff is {np.nanmean(diffs)}")
out_path = output_folder / f"{method}_transforms.pkl"
out.to_pickle(out_path)
