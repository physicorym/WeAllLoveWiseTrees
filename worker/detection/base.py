import cv2
import numpy as np
import logging
import rasterio
from detection.utils.find_dead_pixels import process_and_display_image
from detection.utils.image import normalize_channel, generate_crop_transformations


logger = logging.getLogger(__name__)


def pixel_to_geo(transform, pixel_x, pixel_y):
    geo_x = transform[2] + pixel_x * transform[0] + pixel_y * transform[1]
    geo_y = transform[5] + pixel_x * transform[3] + pixel_y * transform[4]
    return geo_x, geo_y


def detect(layout_name: str, crop: np.ndarray) -> dict:
    with rasterio.open(f"./layouts/{layout_name}") as image:
        layout_image = image.read()
        layout_image_meta = image.meta

    normalized_channels = [normalize_channel(layout_image[i]) for i in range(layout_image.shape[0])]
    large_image_normalized = np.stack(normalized_channels, axis=-1)
    large_image_normalized = cv2.resize(large_image_normalized, (8000, 8000))

    normalized_channels_crop = [normalize_channel(crop[i]) for i in range(crop.shape[0])]
    crop_image_normalized = np.stack(normalized_channels_crop, axis=-1)
    df_dead_pix_crop = process_and_display_image(crop_image_normalized)

    if df_dead_pix_crop is None:
        print("Processed crop image is None.")
        return {
            'geo_coord': None,
            'pixel_coord': None,
            'median_geo_coord': None,
            'df_dead_pix_crop': df_dead_pix_crop,
            'crop_fix': None
        }
    
    df_dead_pix, crop_fix = df_dead_pix_crop
    crop_transformations = generate_crop_transformations(crop_fix)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(large_image_normalized, None)
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
        try:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            h, w = best_transformation.shape[:2]
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

            dst = cv2.perspectiveTransform(pts, M)

            transform = layout_image_meta['transform']
            geo_coords = [pixel_to_geo(transform, pt[0][0], pt[0][1]) for pt in dst]

            median_pixel_x = np.median([pt[0][0] for pt in dst])
            median_pixel_y = np.median([pt[0][1] for pt in dst])
            median_geo_coord = pixel_to_geo(transform, median_pixel_x, median_pixel_y)
            return {
                'geo_coord': {
                    'ul': f'{geo_coords[0][0]}; {geo_coords[0][1]}',
                    'ur': f'{geo_coords[1][0]}; {geo_coords[1][1]}',
                    'br': f'{geo_coords[2][0]}; {geo_coords[2][1]}',
                    'bl': f'{geo_coords[3][0]}; {geo_coords[3][1]}',
                },
                'crs': 'EPSG:32637',
                'median_geo_coord': median_geo_coord,
                'pixel_geo_coord': dst.tolist(),
                'dead_pixel_coord': df_dead_pix.to_records().tostring().hex(),
                'crop_fix': crop_fix.tobytes().hex()
            }
        except Exception as exc:
            logger.error(f"An exception occurred {type(exc)} {exc}")

    return {
        'geo_coord': None,
        'median_geo_coord': None,
        'pixel_geo_coord': None,
        'dead_pixel_coord': df_dead_pix.to_records().tostring().hex(),
        'crop_fix': crop_fix.tobytes().hex()
    }
