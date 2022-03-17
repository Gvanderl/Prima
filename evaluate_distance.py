import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from utils import read_points, image_iterator
from config import output_folder
import cv2
import scipy.io
from compute_matrices import compute_transformation, show_results, open_image


def euclidian_distance(x1, x2, y1, y2):
    return math.sqrt(math.pow(x1-x2, 2) + math.pow(y1-y2, 2))


def eval_transform(points, trans):
    new_points = []
    for x, y in points[1]:
        p = np.transpose(np.array([x, y, 1]))
        new_points.append((trans @ p))
    new_points = np.array(new_points)

    distances = np.zeros((len(points[0])))
    for index, ((x1, y1), (x2, y2)) in enumerate(zip(points[0], points[1])):
        distances[index] = euclidian_distance(x1, x2, y1, y2)
    distances_np = np.zeros((len(points[0])))
    for index, ((x1, y1), (x2, y2)) in enumerate(zip(points[0], new_points)):
        distances_np[index] = euclidian_distance(x1, x2, y1, y2)

    diff = 100 * (distances.mean() - distances_np.mean()) / distances.mean()

    # plt.hist(distances_np)
    # plt.title(f"Distribution of distances_np for {json_path.stem}")
    # plt.show()

    print(f"Mean distance went from {distances.mean():.2f} to {distances_np.mean():.2f} ({diff:.2f}% reduction)\n")
    return new_points, diff


if __name__ == '__main__':
    itk_transformations = pd.read_csv(output_folder / "itk_transformations.csv")

    for base_image_path, other_image_path in image_iterator():
        json_path = output_folder / (base_image_path.stem + "_" + other_image_path.stem + ".json")
        if not json_path.exists():
            continue
        print("\n\nNeow on ", json_path.stem)

        points = read_points(json_path)

        # Computed
        print("Using computed transformation")
        trans = compute_transformation(points)
        new_points = eval_transform(points, trans)
        # show_results(base_image_path, other_image_path, trans, points, new_points)

        # ITK
        print("Using ITK")
        x, y = itk_transformations[(itk_transformations["Name"] == base_image_path.name) &
                                   (itk_transformations["Other"] == other_image_path.name)][["X", "Y"]].values[0]
        other_image = open_image(other_image_path.as_posix(), "tmp")
        cv2.destroyAllWindows()
        trans = np.array(
            [[1, 0, -x * other_image.shape[1]],     # Why negative ?
             [0, 1, -y * other_image.shape[0]]]
        )
        new_points = eval_transform(points, trans)
        # show_results(base_image_path, other_image_path, trans, points, new_points)

        print("Using Greedy")
        path = output_folder.parent / "greedy_output" / (other_image_path.stem + "affine.mat")
        if not path.exists():
            continue
        trans = pd.read_csv(path, sep=" ", header=None).iloc[:2, :3]
        full_image = cv2.imread(other_image_path.as_posix())
        trans.iloc[0, 2] = (trans.iloc[0, 2] / full_image.shape[1]) * other_image.shape[1]
        trans.iloc[1, 2] = (trans.iloc[1, 2] / full_image.shape[0]) * other_image.shape[0]
        trans = np.array(trans)
        new_points = eval_transform(points, trans)
        show_results(base_image_path, other_image_path, trans, points, new_points)
