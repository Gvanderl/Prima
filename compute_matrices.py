import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from pathlib import Path
import math

def transparancy_mask(image):
    trans_mask = image[:, :, 3] < 0.99

    # replace areas of transparency with white and not transparent
    image[trans_mask] = [255, 255, 255, 255]

    # new image without alpha channel...
    return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

def open_image(path, window_name, number):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    cv2.namedWindow(window_name)
    img = cv2.resize(img, (int(img.shape[1] * 1000 / img.shape[0]), 1000))
    return transparancy_mask(img)

data_path = Path("../immuno_project/data")

dirs = ["A00483-C", "A00776-B", "A02076-2B", "A02080-E","A02633-2A","A02634-2C", "A02969-B", "A02969-C", "A02969-F"]
# "A00483-C", "A00776-B", "A02076-2B", "A02080-E",
def string_to_tuple(df):
    for x in df.columns.values:
        df[x] = df[x].apply(eval)
    return df


def compute_transformation(csv_file):
    points = pd.read_csv(csv_file)
    points = string_to_tuple(points)

    # https://lucidar.me/en/mathematics/calculating-the-transformation-between-two-set-of-points/
    # 1) find center of gravity of each point set
    cols = points.columns.values
    for col in cols:
        points[[col + '_x', col + '_y']] = pd.DataFrame(points[col].tolist(), index=points.index)
    sums = points.sum()
    cog = []
    """for col in cols:
        cog.append((sums[col+"_x"]/points.shape[0], sums[col+"_y"]/points.shape[0]))
    print(cog)
    
    translation = (cog[0][0] - cog[1][0], cog[0][1] - cog[1][1])
    print("translation ", translation)
    
    
    N = np.zeros(shape=(2, 2))
    for index, row in points.iterrows():
        a = np.array([[row[cols[0]+"_x"], row[cols[0]+"_y"]]])
        b = np.transpose(np.array([[row[cols[1]+"_x"], row[cols[1]+"_y"]]]))
    
        N += b @ a
    
    u, s, vh = np.linalg.svd(N)
    
    rotation = np.transpose(vh) @ np.transpose(u)
    print("rotation ", rotation)
    print("rotation angle ", math.degrees(math.atan2(rotation[1][0], rotation[0][0])))
    """
    # trans = pinv(M) * proj
    # trans = pinv(img1) * img2
    M = np.zeros(shape=(len(points) * 2, 4))
    proj = np.zeros(shape=(len(points) * 2, 1))
    for index, row in points.iterrows():
        x = row[cols[0] + "_x"]
        y = row[cols[0] + "_y"]

        M[index * 2] = [x, y, 1, 0]
        M[(index * 2) + 1] = [y, -x, 0, 1]

        proj[index * 2] = row[cols[1] + "_x"]
        proj[index * 2 + 1] = row[cols[1] + "_y"]

    T = np.linalg.pinv(M) @ proj
    trans = np.float32([
        [T[0][0], T[1][0], T[2][0]],
        [-T[1][0], T[0][0], T[3][0]],
        # [0, 0, 1]
    ])
    return trans, cols

def show_results(first_image_path, other_image_path, transformation):
    img1 = open_image(str(first_image_path), "image1", 0)
    img2 = open_image(str(other_image_path), "image2", 1)

    blend1 = cv2.addWeighted(img1, 0.5, img2, 0.5, 0.0)

    """M = np.float32([
        [1, 0, translation[0]],
        [0, 1, translation[1]]
    ])
    shifted = cv2.warpAffine(img2, M, (img2.shape[1], img2.shape[0]))

    M = np.float32([
        [rotation[0][0], rotation[0][1], 0],
        [rotation[1][0], rotation[1][1], 0]
    ])
    (h, w) = img1.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY),
                                # math.degrees(math.atan2(rotation[1][0], rotation[0][0])),
                                180, 1.0)
    rotated = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))"""
    rotated = cv2.warpAffine(img1, transformation, (img1.shape[1], img1.shape[0]))

    blend2 = cv2.addWeighted(rotated, 0.5, img2, 0.5, 0.0)
    # blend3 = cv2.addWeighted(img1, 0.5, shifted, 0.5, 0.0)
    while (1):
        cv2.imshow('image1', blend1)
        cv2.imshow('image2', blend2)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

for dir in dirs:
    path = data_path / (dir+"/thumbnails")
    for file in path.glob('*.csv'):
        print("Neow on ", file)

        trans, cols = compute_transformation(file)

        first_image_path = Path(file.parents[0] / cols[0])
        other_image_path = Path(file.parents[0] / cols[1])

        show_results(first_image_path, other_image_path, trans)