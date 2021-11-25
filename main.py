from pathlib import Path
from os import listdir
import logging

from baseline import baseline_match
logging.basicConfig(level=logging.INFO)

data_folder = Path("~/data/prima/data").expanduser().resolve()
samples = [d for d in listdir(data_folder) if d[0] != "."]
logging.info(f"Found samples {samples}")


for sample in samples:
    img_folder = data_folder / sample / "thumbnails"
    base_image_path = img_folder / (sample + ".png")
    assert base_image_path.exists()

    for other_image in listdir(img_folder):
        other_image_path = img_folder / other_image
        if other_image_path == base_image_path:
            continue
        baseline_match(base_image_path, other_image_path)




