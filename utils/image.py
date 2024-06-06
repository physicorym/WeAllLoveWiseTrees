import tifffile as tiff

def load_multichannel_tiff_image(file_path):
    image = tiff.imread(file_path)
    return image
