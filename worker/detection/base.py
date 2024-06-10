import numpy as np
import cv2
import rasterio

from utils.find_features import get_keypoints_and_descriptors, add_trans_descript
from utils.extract_geotiff import get_image_center_coord, transform_coord


def detect(layout: np.ndarray, crop: np.ndarray) -> dict:

    # orig_coord_layout = get_image_center_coord(layout)

    max_size = max(crop.shape[0], crop.shape[1])
    crop = cv2.resize(crop, (max_size, max_size))

    sift = cv2.SIFT_create()

    kp1, des1 = get_keypoints_and_descriptors(layout, sift)

    keypoints_list = []
    descriptors_list = []
    transformations_list = []
    keypoints_list, descriptors_list, transformations_list = add_trans_descript(crop, sift, keypoints_list, descriptors_list, transformations_list)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    all_matches = []
    for i, des2 in enumerate(descriptors_list):
        matches = bf.match(des1, des2)
        for match in matches:
            match.imgIdx = i
        all_matches.extend(matches)

    all_matches = sorted(all_matches, key=lambda x: x.distance)

    good_matches = all_matches[:50]

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_list[m.imgIdx][m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    h, w = crop.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    with rasterio.open("/Volumes/T7/opensource_datasets/SPACE_HACK_summer/layouts/layout_2022-03-17.tif") as src:
        transform = src.transform

        transformed_points = [transform_coord(point[0][0], point[0][1], transform) for point in dst]

    return transformed_points
