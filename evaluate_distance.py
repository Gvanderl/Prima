import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from utils import read_points, image_iterator
from config import output_folder, data_folder
from compute_matrices import compute_transformation, string_to_tuple, show_results


def euclidian_distance(x1, x2, y1, y2):
    return math.sqrt(math.pow(x1-x2, 2) + math.pow(y1-y2, 2))


itk_transformations = pd.read_csv(output_folder / "itk_transformations.csv")

for base_image_path, other_image_path in image_iterator():
    json_path = output_folder / (base_image_path.stem + "_" + other_image_path.stem + ".json")
    if not json_path.exists():
        continue
    print("Neow on ", json_path.stem)

    points = read_points(json_path)

    # Computed
    # trans = compute_transformation(points)

    # ITK
    x, y = itk_transformations[(itk_transformations["Name"] == base_image_path.name) &
                               (itk_transformations["Other"] == other_image_path.name)][["X", "Y"]].values[0]
    trans = np.array(
        [[1, 0, -x],
         [0, 1, -y]]
    )

    new_points = []
    for x, y in points[1]:
        p = np.transpose(np.array([x, y, 1]))
        new_points.append((trans @ p))

    new_points = np.array(new_points)

    distances = np.zeros((len(points[0])))
    for index, ((x1, y1), (x2, y2)) in enumerate(zip(points[0], new_points)):
        distances[index] = euclidian_distance(x1, x2, y1, y2)

    plt.hist(distances)
    plt.title(f"Distribution of distances for {json_path.stem}")
    plt.show()

    print(f"Mean distance is {distances.mean():.2f} for {json_path.stem}\n\n")

    show_results(base_image_path, other_image_path, trans, points, new_points)



