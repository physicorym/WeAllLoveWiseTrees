import tifffile as tiff
import cv2

def load_multichannel_tiff_image(file_path):
    image = tiff.imread(file_path)
    return image

def create_tiff_rgb(file_path):

    image = load_multichannel_tiff_image(file_path)
    img_rgb = cv2.merge([image[:, :, 0], image[:, :, 1], image[:, :, 2]])

    return img_rgb
