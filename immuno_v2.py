from typing import List, Tuple, Union, Dict
import logging
import os
import json
import numpy as np
from skimage.measure import label, regionprops
from skimage.measure._regionprops import RegionProperties
from skimage.filters import threshold_multiotsu, gaussian
from skimage import transform
from matplotlib import cm
from scipy import ndimage
from PIL import Image
import pathlib
from pathlib import Path
from utils import image_iterator, whitewashing, otsu, grayscale
import cv2
from os import listdir
from config import data_folder
from skimage.filters.rank import median
from skimage.morphology import disk
import matplotlib.pyplot as plt
from registrations.itk_registation import itk_registration

Coords = Tuple[float, float] or List[int]


def get_tissue_pale_check(image: Image) -> np.array:
    """
    this function offers an alternative to get the mask of the tissue on an image.
    It has been designed to process very pale immuno tissue as well as he(s) dark regions
    :param image:
    :return:
    """
    image_array = np.array(image)
    # projection on red channel
    image_projection = image_array[:, :, 0]
    # preprocess the image so that black background is set to white
    image_projection[image_projection == 0] = 255

    thresholds_otsu = threshold_multiotsu(image_projection, classes=2)
    threshold = thresholds_otsu[0]
    # if threshold computed with two classes is too low, it probably means the tissue is very pale
    # to get is segmented, we are less discriminative bu increasing the number of otsu classes and taking the
    # second class otsu threshold
    if threshold < 200:  # hard coded threshold limit set based on trial and error
        threshold = threshold_multiotsu(image_projection, classes=3)[1]

    # apply threshold
    mask = (image_projection < threshold) * 255  # MAX_PIXEL_VALUE
    return mask


def breakdown_mask_into_pieces(mask: np.array) -> List[np.array]:
    """
    function that gets individual tissue pieces from a slide binary mask
    process :
    - get mask region properties
    - parse them in descending area order
    - for each prop, cluster it with its neighbors
    - the process ends when every prop belongs to a cluster
    - for each cluster, a mask is extracted from the bbox
    - returns the list of obtained cluster mask
    :param mask:
    :return:
    """

    def are_neighbors(prop_1: RegionProperties, prop_2: RegionProperties) -> bool:
        """
        determine if two props are neighbor based on custom distance criterion
        :param prop_1:
        :param prop_2:
        :return:
        """
        # get a distance between prop bboxes
        x_min = min(prop_1.bbox[3], prop_2.bbox[3])
        x_max = max(prop_1.bbox[1], prop_2.bbox[1])
        y_min = min(prop_1.bbox[2], prop_2.bbox[2])
        y_max = max(prop_1.bbox[0], prop_2.bbox[0])
        delta_x = x_min - x_max
        delta_y = y_min - y_max
        dist = ((delta_x * (delta_x > 0)) ** 2 + (delta_y * (delta_y > 0)) ** 2) ** 0.5
        biggest_object = prop_1 if prop_1.area > prop_2.area else prop_2
        biggest_bbox = biggest_object.bbox
        biggest_box_longest_side = max(biggest_bbox[3] - biggest_bbox[1], biggest_bbox[2] - biggest_bbox[0])
        return dist < 0.3 * biggest_box_longest_side  # custom distance, value 0.3 set based on trial and error,
        # gives appropriate clusters

    def cluster_props(props: List[RegionProperties], clusters: List[List[RegionProperties]]) -> \
            List[List[RegionProperties]]:
        """
        recursively clusters the region properties based on distance neighboring criteria
        NB : this method may output a different result based on order of props => props shall be ordered based on
        their size with descending order before a call to this function
        :param props:
        :param clusters:
        :return:
        """
        seed_obj = props[0]
        current_cluster = [seed_obj] + [_prop for _prop in props[1:] if are_neighbors(seed_obj, _prop)]
        clusters.append(current_cluster)
        for obj in current_cluster:
            props.remove(obj)
        if len(props) == 0:
            return clusters
        elif len(props) == 1:
            clusters.append([props[0]])
            return clusters
        else:
            return cluster_props(props, clusters)

    cluster_objects = []
    individual_mask = label(mask, connectivity=2)
    prop = regionprops(individual_mask)
    # get max prop area
    max_area = max(p.area for p in prop if p.label > 1)
    # exclude properties that are too small
    min_area_proportion = 0.3
    filtered_prop = [p for p in prop if p.area > min_area_proportion * max_area and p.label > 1]
    sorted_objects = sorted(filtered_prop, key=lambda x: x.area, reverse=True)

    final_out = np.zeros(mask.shape + (3,))
    for i, cluster in enumerate(sorted_objects):
        color = cm.Set1(i % 8)[:3]
        color_mask = (np.stack(
            [color[0] * cluster.image, color[1] * cluster.image, color[2] * cluster.image],
            axis=2) / 2)
        final_out[cluster.bbox[0]: cluster.bbox[2], cluster.bbox[1]: cluster.bbox[3], :] += color_mask
        final_out = np.clip(final_out, 0, 1)
    plt.imshow(final_out)
    plt.title("Different clusters")
    plt.show()

    # form clusters
    all_clusters = cluster_props(props=sorted_objects, clusters=[])
    logging.debug('len final ', len(all_clusters))
    # extract a mask object for each cluster
    rows = []
    cols = []
    for cluster in all_clusters:
        row_min = min(o.bbox[0] for o in cluster)
        col_min = min(o.bbox[1] for o in cluster)
        row_max = max(o.bbox[2] for o in cluster)
        col_max = max(o.bbox[3] for o in cluster)
        rows.append((row_min, row_max))
        cols.append((col_min, col_max))
        current_mask = mask[row_min:row_max, col_min:col_max]
        logging.debug('mask ratio', np.sum(current_mask > 0) / np.prod(np.shape(current_mask)))
        current_mask = ndimage.binary_fill_holes(current_mask).astype(np.uint8) * 255
        cluster_objects.append(current_mask)
    return cluster_objects, rows, cols


def get_crops(im_path):
    print(f"\n***** Getting crops for {im_path.stem} *****\n")

    img = Image.open(str(im_path))
    img_cv = cv2.imread(str(im_path))

    """
    mask = get_tissue_pale_check(img1)
    mask = Image.fromarray(mask)
    mask.show()
    """

    mask2 = get_tissue_pale_check(img)
    mask = (otsu(
        grayscale(
            whitewashing(
                img_cv,
                threshold=1
            ))) == 0) * 255

    mask2 = median(mask, disk(10))
    mask = ndimage.binary_fill_holes(mask) * 255
    # print(mask)
    # while 1:
    #     cv2.imshow("image", mask.astype(np.uint8))
    #     if cv2.waitKey(20) & 0xFF == 27:
    #         break
    # cv2.destroyAllWindows()

    clusters, rows, cols = breakdown_mask_into_pieces(mask)

    print("nb clusters = ", len(clusters))
    cropped_images = list()
    clusters_images = img_cv.copy()
    for i, c, (r_min, r_max), (c_min, c_max) in zip(range(len(clusters)), clusters, rows, cols):
        clusters_images = cv2.cvtColor(clusters_images, cv2.COLOR_RGB2RGBA)
        clusters_images = cv2.rectangle(clusters_images, (c_min, r_min), (c_max, r_max), (148, 240, 100), 5)
        clusters_images = cv2.circle(clusters_images, (int((c_min + c_max) / 2.0), int((r_min + r_max) / 2.0)), 5,
                                     (148, 100, 240), 5)
        cropped_images.append(img_cv[r_min:r_max, c_min:c_max, :])
    clusters_images = cv2.resize(clusters_images,
                                 (int(clusters_images.shape[1] * 1000 / clusters_images.shape[0]), 1000))
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, (int(mask.shape[1] * 1000 / mask.shape[0]), 1000))
    mask2 = mask2.astype(np.uint8)
    mask2 = cv2.resize(mask2, (int(mask2.shape[1] * 1000 / mask2.shape[0]), 1000))

    plt.imshow(cv2.cvtColor(clusters_images, cv2.COLOR_BGR2RGB))
    plt.title("Image clusters")
    plt.show()
    plt.imshow(cv2.cvtColor(cropped_images[0], cv2.COLOR_BGR2RGB))
    plt.title("First crop")
    plt.show()
    plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    plt.title("mask")
    plt.show()
    plt.imshow(cv2.cvtColor(mask2, cv2.COLOR_BGR2RGB))
    plt.title("mask2")
    plt.show()

    return cropped_images


def score_match(im1, im2):
    mask1 = (otsu(
        grayscale(
            whitewashing(
                im1,
                threshold=1
            ))) == 0) * 255
    mask1 = median(mask1, disk(10))

    mask2 = (otsu(
        grayscale(
            whitewashing(
                im2,
                threshold=1
            ))) == 0) * 255
    mask2 = median(mask2, disk(10))

    intersection = np.sum((mask1 == 255) & (mask2 == 255))
    union = np.sum((mask1 == 255)) + np.sum((mask2 == 255)) - intersection

    iou = intersection / union
    plt.imshow(np.rollaxis(np.stack([mask1, mask2, np.zeros((mask1.shape))]), 0, 3))
    plt.title(f"IoU = {iou}")
    plt.show()
    return iou


if __name__ == '__main__':

    base_crops = dict()
    for im1_path, im2_path in image_iterator():
        if im1_path.stem != "A02634-2C":
        # if im1_path.stem != "A02969-F":
            continue
        if im1_path not in base_crops:
            base_crops[im1_path] = get_crops(im1_path)
        other_crops = get_crops(im2_path)
        for base_crop in base_crops[im1_path]:
            plt.imshow(cv2.cvtColor(base_crop, cv2.COLOR_BGR2RGB))
            plt.show()
            for other_crop in other_crops:
                plt.imshow(cv2.cvtColor(other_crop, cv2.COLOR_BGR2RGB))
                plt.show()
                tmp1, tmp2 = Path('tmpcrop1.png'), Path('tmpcrop2.png')
                cv2.imwrite(tmp1.as_posix(), base_crop)
                cv2.imwrite(tmp2.as_posix(), other_crop)
                transform = itk_registration(fixed_input_image=tmp1,
                                             moving_input_image=tmp2,
                                             output_name=im2_path.stem)
                transform[0, 2] = transform[0, 2] * other_crop.shape[1]
                transform[1, 2] = transform[1, 2] * other_crop.shape[0]
                os.remove(tmp1.as_posix())
                os.remove(tmp2.as_posix())
                other_crop = cv2.warpAffine(other_crop, transform, base_crop.shape[:2:][::-1])
                plt.imshow(cv2.cvtColor(other_crop, cv2.COLOR_BGR2RGB))
                plt.show()
                score_match(base_crop, other_crop)
