import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from pathlib import Path

data_path = Path("../immuno_project/data")
old_images = []


def open_image(path, window_name, number):
    img = cv2.imread(path)
    cv2.namedWindow(window_name)
    img = cv2.resize(img, (int(img.shape[1] * 1000 / img.shape[0]), 1000))
    cv2.setMouseCallback(window_name, draw_circle, [img, number])
    old_images.append(img.copy())
    return img


def prep_path(path):
    return os.path.dirname(path) + "/" + (os.path.basename(path)).split(".")[0] + "_labeled.png"


# mouse callback function
def draw_circle(event, x, y, flags, param):  # param[0] should contain image, param[1] should contain 0 or 1
    if event == cv2.EVENT_LBUTTONDOWN:
        color = [x * 255 for x in plt.cm.tab10((len(points) // 2) % 10)[:3]]
        old_images[param[1]] = param[0].copy()
        cv2.circle(param[0], (x, y), 4, color, -1)
        points.append((x, y))
        print(color)


# Create a black image, a window and bind the function to window
dirs=["A00483-C", "A00776-B", "A02076-2B", "A02080-E", "A02633-2A","A02634-2C", "A02969-B", "A02969-C", "A02969-F"]

samples = [d for d in os.listdir(data_path) if d[0] != "."]

"""
for sample in samples:
    img_folder = data_path / sample / "thumbnails"
    base_image_path = img_folder / (sample + ".png")
    assert base_image_path.exists()

    for other_image in os.listdir(img_folder):
        other_image_path = img_folder / other_image
        if other_image_path == base_image_path:
            continue
"""

# for dir in data_path.rglob('**/*'):
i = 0
for dir in dirs:
    i += 1
    path = data_path / (dir+"/thumbnails")
    for file in path.glob('*.png'):
        if str(file).count('-') > 2:
            image1_path = str(file)
        else:
            image2_path = str(file)
    print(image2_path)
    print(image1_path)

    img1 = open_image(image1_path, "image1", 0)
    img2 = open_image(image2_path, "image2", 1)

    points = list()

    while (1):
        cv2.imshow('image1', img1)
        cv2.imshow('image2', img2)
        if cv2.waitKey(20) & 0xFF == 27:
            break
        elif cv2.waitKey(1) == ord('b'):
            print("b pressed")
            if len(points) % 2 == 0:
                img2 = old_images[1].copy()
                cv2.setMouseCallback('image2', draw_circle, [img2, 1])
                points.pop()
            else:
                img1 = old_images[0].copy()
                cv2.setMouseCallback('image1', draw_circle, [img1, 0])
                points.pop()

    p1 = []
    p2 = []
    for x1, x2 in zip(*[iter(points)] * 2):
        p1.append(x1)
        p2.append(x2)
    df = pd.DataFrame(list(zip(p1, p2)),
                      columns=[os.path.basename(image1_path), os.path.basename(image2_path)])
    if len(df) > 6:
        df.to_csv(os.path.dirname(image2_path) + "/points.csv", index=False)
        print("saved")
    print(points)

    cv2.destroyAllWindows()

    if i >= 5:
        break
