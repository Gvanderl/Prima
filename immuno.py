from typing import List, Tuple, Union, Dict
import logging
import os
import json
import numpy as np
from skimage.measure import label, regionprops
from skimage.measure._regionprops import RegionProperties
from skimage import transform
from matplotlib import cm
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from PIL import Image
from utils import image_iterator
from skimage.filters import threshold_multiotsu


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
    mask = (image_projection < threshold) * 255
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
    min_area_proportion = 0.1
    filtered_prop = [p for p in prop if p.area > min_area_proportion * max_area and p.label > 1]
    sorted_objects = sorted(filtered_prop, key=lambda x: x.area, reverse=True)
    # form clusters
    all_clusters = cluster_props(props=sorted_objects, clusters=[])
    logging.debug('len final ', len(all_clusters))
    # extract a mask object for each cluster
    for cluster in all_clusters:
        row_min = min(o.bbox[0] for o in cluster)
        col_min = min(o.bbox[1] for o in cluster)
        row_max = max(o.bbox[2] for o in cluster)
        col_max = max(o.bbox[3] for o in cluster)
        current_mask = mask[row_min:row_max, col_min:col_max]
        logging.debug('mask ratio', np.sum(current_mask > 0) / np.prod(np.shape(current_mask)))
        current_mask = ndimage.binary_fill_holes(current_mask).astype(np.uint8) * 255
        cluster_objects.append(current_mask)
    return cluster_objects


for im_path in image_iterator(False):
    img = cv2.imread(str(im_path))

    mask = get_tissue_pale_check(img)

    clusters = breakdown_mask_into_pieces(mask)
    cluster_image = img.copy()
    cluster_image = cv2.resize(cluster_image, (clusters[0].shape[1], clusters[0].shape[0]))
    for i, cluster in enumerate(clusters):
        color = cm.Set1(i % 8)[:3]
        color_mask = (np.stack([color[2] * cluster, color[1] * cluster, color[0] * cluster], axis=2) / 2).astype(int)
        # tmp[r_min:r_max, c_min:c_max] = [242, 148, 211]
        cluster_image = np.clip(cluster_image + color_mask, 0, 255)
        # tmp[r_max:, c_max:] = [0, 0, 0]
        # tmp[:r_min, :c_min] = [0, 0, 0]

    # mask = mask.astype(np.uint8)
    # mask = cv2.resize(mask, (int(mask.shape[1] * 1000 / mask.shape[0]), 1000))
    while 1:
        cv2.imshow("image clusters", cluster_image)
        # cv2.imshow("mask", mask)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
