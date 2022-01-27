import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from pathlib import Path
import math
from compute_matrices import compute_transformation, string_to_tuple

data_path = Path("../immuno_project/data")

dirs = ["A00483-C", "A00776-B", "A02076-2B", "A02080-E","A02633-2A","A02634-2C", "A02969-B", "A02969-C", "A02969-F"]
# "A00483-C", "A00776-B", "A02076-2B", "A02080-E",

for dir in dirs:
    path = data_path / (dir+"/thumbnails")
    for file in path.glob('*.csv'):
        print("Neow on ", file)

        trans, points, cols = compute_transformation(file)
        sums = points.sum()
        cog = []

        for col in cols:
            cog.append((sums[col+"_x"]/points.shape[0], sums[col+"_y"]/points.shape[0]))

        new_points = []
        for index, row in points.iterrows():
            p = np.transpose(np.array([row[cols[0]+"_x"], row[cols[0]+"_y"], 1]))
            new_points.append((trans @ p))

        new_points = np.array(new_points)
        new_cog = new_points.sum(axis=0)/len(new_points)
        print(new_cog)

        print("difference avant transformation = (",
              "{:.4f}".format(cog[1][0] -cog[0][0]),
              ", ", "{:.4f}".format(cog[1][1]- cog[0][1]), ")")
        print("difference apres transformation = (",
              "{:.6f}".format(cog[1][0] -new_cog[0]),
              ", ", "{:.6f}".format(cog[1][1]- new_cog[1]), ")")
