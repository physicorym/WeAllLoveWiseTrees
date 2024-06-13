
import rasterio

def get_image_center_coord(file_path):

    with rasterio.open(file_path) as dataset:
        x_center = dataset.bounds.left + (dataset.bounds.right - dataset.bounds.left) / 2
        y_center = dataset.bounds.bottom + (dataset.bounds.top - dataset.bounds.bottom) / 2

        metadata = dataset.meta
        print(metadata)

    return x_center, y_center

def transform_coord(x, y, transform):
    return transform * (x, y)

def calc_new_coord():

    with rasterio.open("/Volumes/T7/opensource_datasets/SPACE_HACK_summer/layouts/layout_2022-03-17.tif") as src:
        transform = src.transform

        transformed_points = [transform_coord(point[0][0], point[0][1], transform) for point in dst]

    return transformed_points


file_path = "/Users/nikitakamenev/Documents/scince/DL_PROJECT/tiff_image/layout_2021-06-15.tif"

center_coordinates = get_image_center_coord(file_path)
print("EPSG:32637:", center_coordinates)

