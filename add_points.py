import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from pathlib import Path
from config import data_folder, output_folder
import json

data_path = data_folder
old_images = []


def get_color(index):
    return [x * 255 for x in plt.cm.tab10(index % 10)[:3]]


def open_image(path, window_name, number):
    img = cv2.imread(path.as_posix())
    cv2.namedWindow(window_name)
    img = cv2.resize(img, (int(img.shape[1] * 1000 / img.shape[0]), 1000))
    cv2.setMouseCallback(window_name, draw_circle, [img, number])
    return img


def prep_path(path):
    return os.path.dirname(path) + "/" + (os.path.basename(path)).split(".")[0] + "_labeled.png"


# mouse callback function
def draw_circle(event, x, y, flags, param):  # param[0] should contain image, param[1] should contain 0 or 1
    if event == cv2.EVENT_LBUTTONDOWN:
        color = get_color(len(points[param[1]]))
        cv2.circle(param[0], (x, y), 4, color, -1)
        points[param[1]].append((x, y))
    elif event == cv2.EVENT_MBUTTONDOWN:
        closest_point = None, float("inf")
        for i, (x2, y2) in enumerate(points[param[1]]):
            distance = np.sqrt((x2 - x)**2 + (y2 - y)**2)
            if distance < closest_point[1]:
                closest_point = i, distance
        if closest_point[1] < 100:
            points[param[1]][closest_point[0]] = (x, y)
            if param[1] == 0:
                param[0] = draw_img1()
            else:
                param[0] = draw_img2()

def draw_img1():
    img1 = open_image(image1_path, "image1", 0)
    for i, point in enumerate(points[0]):
        cv2.circle(img1, point, 4, get_color(i), -1)
    return img1


def draw_img2():
    img2 = open_image(image2_path, "image2", 1)
    for i, point in enumerate(points[1]):
        cv2.circle(img2, point, 4, get_color(i), -1)
    return img2

# Create a black image, a window and bind the function to window
dirs=["A00483-C", "A00776-B", "A02076-2B", "A02080-E", "A02633-2A","A02634-2C", "A02969-B", "A02969-C", "A02969-F"]
quit_flag = False
samples = [d for d in os.listdir(data_path) if d[0] != "."]

for sample in samples:
    img_folder = data_path / sample / "thumbnails"
    image1_path = img_folder / (sample + ".png")
    assert image1_path.exists()

    for other_image in os.listdir(img_folder):
        image2_path = img_folder / other_image
        if image2_path == image1_path:
            continue
        print(f"Matching {image1_path} and {image2_path}")

        out_path = output_folder / f"{image1_path.stem}_{image2_path.stem}.json"
        if out_path.exists():
            with open(out_path) as f:
                points = json.load(f)
                points[0] = [tuple(x) for x in points[0]]
                points[1] = [tuple(x) for x in points[1]]
        else:
            points = [list(), list()]

        img1 = draw_img1()
        img2 = draw_img2()

        while 1:
            cv2.imshow('image1', img1)
            cv2.imshow('image2', img2)
            if cv2.waitKey(50) & 0xFF == 27:
                break
            if cv2.waitKey(50) == ord('q'):
                quit_flag = True
                break
            elif cv2.waitKey(50) == ord('b'):
                print("b pressed")
                if len(points[0]) == 0 and len(points[1]) == 0:
                    print("No points to delete")
                elif len(points[0]) > len(points[1]):
                    removed = points[0].pop()
                    img1 = draw_img1()
                    print(f"Removed {removed}")
                else:
                    removed = points[1].pop()
                    img2 = draw_img2()
                    print(f"Removed {removed}")

                print(points)

        with open(out_path, 'w') as f:
            json.dump(points, f)
        print(f"Saved to {out_path}")
        print(points)
        cv2.destroyAllWindows()

        if quit_flag:
            break
    if quit_flag:
        break
