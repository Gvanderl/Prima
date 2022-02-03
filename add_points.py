import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from config import data_folder, output_folder
import json
from os import listdir
from utils import read_points, image_iterator


class Pointer:

    def __init__(self, image1_path, image2_path):
        self.points = [list(), list()]
        self.image1_path = image1_path
        self.image2_path = image2_path

    def get_color(self, index):
        return [x * 255 for x in plt.cm.tab10(index % 10)[:3]]

    def open_image(self, path, window_name, number):
        img = cv2.imread(path.as_posix())
        cv2.namedWindow(window_name)
        img = cv2.resize(img, (int(img.shape[1] * 1000 / img.shape[0]), 1000))
        cv2.setMouseCallback(window_name, self.draw_circle, [img, number])
        return img

    def prep_path(self, path):
        return os.path.dirname(path) + "/" + (os.path.basename(path)).split(".")[0] + "_labeled.png"

    # mouse callback function
    def draw_circle(self, event, x, y, flags, param):  # param[0] should contain image, param[1] should contain 0 or 1
        if event == cv2.EVENT_LBUTTONDOWN:
            color = self.get_color(len(self.points[param[1]]))
            cv2.circle(param[0], (x, y), 4, color, -1)
            self.points[param[1]].append((x, y))
            print(f"{len(self.points[param[1]])} Pair of points")
        elif event == cv2.EVENT_MBUTTONDOWN:
            closest_point = None, float("inf")
            for i, (x2, y2) in enumerate(self.points[param[1]]):
                distance = np.sqrt((x2 - x)**2 + (y2 - y)**2)
                if distance < closest_point[1]:
                    closest_point = i, distance
            if closest_point[1] < 100:
                self.points[param[1]][closest_point[0]] = (x, y)
                if param[1] == 0:
                    param[0] = self.draw_img1()
                else:
                    param[0] = self.draw_img2()

    def draw_img1(self):
        img1 = self.open_image(self.image1_path, self.image1_path.stem, 0)
        for i, point in enumerate(self.points[0]):
            cv2.circle(img1, point, 4, self.get_color(i), -1)
        return img1

    def draw_img2(self):
        img2 = self.open_image(self.image2_path, self.image2_path.stem, 1)
        for i, point in enumerate(self.points[1]):
            cv2.circle(img2, point, 4, self.get_color(i), -1)
        return img2

    def run(self):
        # Create a black image, a window and bind the function to window
        quit_flag = False

        print(f"Matching {self.image1_path} and {self.image2_path}")

        out_path = output_folder / f"{self.image1_path.stem}_{self.image2_path.stem}.json"
        if out_path.exists():
            self.points = read_points(out_path)

        img1 = self.draw_img1()
        img2 = self.draw_img2()

        while 1:
            cv2.imshow(self.image1_path.stem, img1)
            cv2.imshow(self.image2_path.stem, img2)
            if cv2.waitKey(50) & 0xFF == 27:
                break
            if cv2.waitKey(50) == ord('q'):
                quit_flag = True
                break
            elif cv2.waitKey(50) == ord('b'):
                print("b pressed")
                if len(self.points[0]) == 0 and len(self.points[1]) == 0:
                    print("No points to delete")
                elif len(self.points[0]) > len(self.points[1]):
                    removed = self.points[0].pop()
                    img1 = self.draw_img1()
                    print(f"Removed {removed}")
                else:
                    removed = self.points[1].pop()
                    img2 = self.draw_img2()
                    print(f"Removed {removed}")

                print(self.points)

        with open(out_path, 'w') as f:
            json.dump(self.points, f)
        print(f"Saved to {out_path}")
        print(self.points)
        cv2.destroyAllWindows()

        return quit_flag


if __name__ == '__main__':
    for base_image_path, other_image_path in image_iterator():
        pointer = Pointer(base_image_path, other_image_path)
        quit = pointer.run()
        if quit:
            break
