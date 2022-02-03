from os import listdir
import logging
from utils import image_iterator
import pandas as pd

from config import data_folder, output_folder
from baseline import baseline_match
from itk_registation import itk_registration

logging.basicConfig(level=logging.INFO)

samples = [d for d in listdir(data_folder) if "." not in d]
logging.info(f"Found samples {samples}")

names, others, X, Y = [], [], [], []

for base_image_path, other_image_path in image_iterator():
    print(f"Performing registration on {base_image_path.name} and {other_image_path.name}")
    x, y = itk_registration(fixed_input_image=base_image_path,
                            moving_input_image=other_image_path,
                            output_name=other_image_path.stem)
    names.append(base_image_path.name)
    others.append(other_image_path.name)
    X.append(x)
    Y.append(y)

df = pd.DataFrame(list(zip(names, others, X, Y)),
                  columns=['Name', 'Other', 'X', 'Y'])
df.to_csv(output_folder / 'itk_transformations.csv', index=False)
