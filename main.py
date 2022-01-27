from os import listdir
import logging

import pandas as pd

from config import data_folder, output_folder
from baseline import baseline_match
from itk_registation import itk_registration

logging.basicConfig(level=logging.INFO)

samples = [d for d in listdir(data_folder) if "." not in d]
logging.info(f"Found samples {samples}")

names, others, X, Y = [], [], [], []

for sample in samples:
    img_folder = data_folder / sample / "thumbnails"
    base_image_path = img_folder / (sample + ".png")
    assert base_image_path.exists(), f"{base_image_path} does not exist"

    for other_image in listdir(img_folder):
        other_image_path = img_folder / other_image
        if other_image_path == base_image_path:
            continue
        print(f"Performing registration on {base_image_path.name} and {other_image_path.name}")
        # baseline_match(base_image_path, other_image_path)
        x, y = itk_registration(fixed_input_image=base_image_path,
                                moving_input_image=other_image_path,
                                output_name=other_image_path.stem)
        names.append(base_image_path.name)
        others.append(other_image_path.name)
        X.append(x)
        Y.append(y)

df = pd.DataFrame(list(zip(names, others, X, Y)),
                  columns=['Name', 'Other', 'X', 'Y'])
df.to_csv(output_folder / 'transformations.csv', index=False)
