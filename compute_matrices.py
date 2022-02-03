import cv2
import numpy as np


def transparancy_mask(image):
    if image.shape[2] < 4:
        return image
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


def string_to_tuple(df):
    for x in df.columns.values:
        df[x] = df[x].apply(eval)
    return df


def compute_transformation(points):
    M = np.zeros(shape=(len(points[0]) * 2, 4))
    proj = np.zeros(shape=(len(points[0]) * 2, 1))
    for index, ((x1, y1), (x2, y2)) in enumerate(zip(points[0], points[1])):

        M[index * 2] = [x2, y2, 1, 0]
        M[(index * 2) + 1] = [y2, -x2, 0, 1]

        proj[index * 2] = x1
        proj[index * 2 + 1] = y1

    T = np.linalg.pinv(M) @ proj
    trans = np.float32([
        [T[0][0], T[1][0], T[2][0]],
        [-T[1][0], T[0][0], T[3][0]],
        # [0, 0, 1]
    ])
    return trans


def show_results(first_image_path, other_image_path, transformation, points, new_points):
    img1 = open_image(str(first_image_path), "Before", 0)
    img2 = open_image(str(other_image_path), "After", 1)

    blend1 = cv2.addWeighted(img1, 0.5, img2, 0.5, 0.0)
    rotated = cv2.warpAffine(img2, transformation, (img2.shape[1], img2.shape[0]))

    blend2 = cv2.addWeighted(rotated, 0.5, img1, 0.5, 0.0)
    # blend3 = cv2.addWeighted(img1, 0.5, shifted, 0.5, 0.0)

    for (x1, y1), (x2, y2) in zip(points[0], points[1]):
        cv2.circle(blend1, (x1, y1), 4, (255, 0, 0), -1)
        cv2.circle(blend2, (x1, y1), 4, (255, 0, 0), -1)
        cv2.circle(blend1, (x2, y2), 4, (150, 200, 0), -1)

    for x, y in new_points:
        cv2.circle(blend2, (int(x), int(y)), 4, (150, 200, 0), -1)

    while (1):
        cv2.imshow("Before", blend1)
        cv2.imshow("After", blend2)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()