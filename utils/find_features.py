import cv2

def get_keypoints_and_descriptors(image, sift):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray_image, None)
    return kp, des



def add_transformed_descriptors(image, sift, keypoints_list, descriptors_list, transformations_list):
    transformations = [None, cv2.flip(image, 1)]
    for angle in range(0, 360, 90):
        for transform in transformations:
            if transform is not None:
                transformed_image = transform
            else:
                transformed_image = image
            rotated_image = cv2.rotate(transformed_image, cv2.ROTATE_90_CLOCKWISE) if angle != 0 else transformed_image
            kp, des = get_keypoints_and_descriptors(rotated_image, sift)
            keypoints_list.append(kp)
            descriptors_list.append(des)
            transformations_list.append((transform, angle))