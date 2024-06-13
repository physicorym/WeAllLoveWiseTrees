import numpy as np
import cv2
import rasterio

from utils.find_features import get_keypoints_and_descriptors, add_trans_descript
from utils.extract_geotiff import get_image_center_coord, transform_coord
from utils.image import normalize_channel, generate_crop_transformations


def detect(layout_path: str, crop: np.ndarray) -> dict:

    with rasterio.open(layout_path) as image:
        layout_image = image.read()
        layout_image_meta = image.meta

    normalized_channels = []
    for i in range(layout_image.shape[0]):
        normalized_channel = normalize_channel(layout_image[i])
        normalized_channels.append(normalized_channel)

    large_image_normalized = np.stack(normalized_channels, axis=-1)
    large_image_bgr = cv2.cvtColor(large_image_normalized, cv2.COLOR_RGB2BGR)
    large_image_normalized_orig = large_image_normalized
    large_image_normalized = cv2.resize(large_image_normalized, (5000, 5000))

    normalized_channels_crop = []
    for i in range(crop.shape[0]):
        normalized_channel_crop = normalize_channel(crop[i])
        normalized_channels_crop.append(normalized_channel_crop)

    crop_image_normalized = np.stack(normalized_channels_crop, axis=-1)
    crop_image_bgr = cv2.cvtColor(crop_image_normalized, cv2.COLOR_RGB2BGR)

    crop_transformations = generate_crop_transformations(crop_image_bgr)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(large_image_bgr, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    best_matches = None
    best_transformation = None
    max_good_matches = 0

    for crop_trans in crop_transformations:
        kp2, des2 = sift.detectAndCompute(crop_trans, None)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50]
        
        if len(good_matches) > max_good_matches:
            max_good_matches = len(good_matches)
            best_matches = good_matches
            best_transformation = crop_trans

    if best_matches:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        h, w = best_transformation.shape[:2]
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

    return dst
