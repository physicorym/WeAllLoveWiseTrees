import tifffile as tiff
import cv2
import numpy as np


def load_multichannel_tiff_image(file_path):
    image = tiff.imread(file_path)
    return image


def create_tiff_rgb(file_path):

    image = load_multichannel_tiff_image(file_path)
    img_rgb = cv2.merge([image[:, :, 0], image[:, :, 1], image[:, :, 2]])

    return img_rgb


def normalize_channel(channel, lower_percentile=2, upper_percentile=98):
    lower = np.percentile(channel, lower_percentile)
    upper = np.percentile(channel, upper_percentile)
    channel = np.clip(channel, lower, upper)
    if upper > lower:
        channel = (channel - lower) / (upper - lower) * 255.0
    return channel.astype(np.uint8)


def generate_crop_transformations(crop_image):
    transformations = []
    angles = [0, 90, 180, 270]
    
    for angle in angles:
        if angle == 0:
            rotated = crop_image
        elif angle == 90:
            rotated = cv2.rotate(crop_image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(crop_image, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(crop_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        transformations.append(rotated)
        flipped = cv2.flip(rotated, 1)
        transformations.append(flipped)
    
    return transformations
