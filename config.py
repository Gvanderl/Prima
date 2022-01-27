from pathlib import Path
import logging
from os import listdir

data_folder = Path("~/data/prima/data").expanduser().resolve()
# data_folder = Path("/Volumes/Backup Plus/Immuno_project/data").expanduser().resolve()
# data_folder = Path("/media/gael/Space/data/Immuno_project/data").expanduser().resolve()
output_folder = Path("./outputs").resolve()
output_folder.mkdir(exist_ok=True)
