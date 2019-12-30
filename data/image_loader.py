import cv2
import numpy as np
from PIL import Image


def pil_to_cv_image(image_pil, output_bgr=True):
    image_cv = np.array(image_pil)
    if output_bgr:
        image_cv = image_cv[:, :, ::-1]
    return image_cv


def cv_to_pil_image(image_cv, output_rgb=True):
    if output_rgb:
        image_cv = image_cv[..., ::-1]  # Convert BGR opencv image to RGB first
    return Image.fromarray(image_cv)


def opencv_loader(path):
    img_bgr = cv2.imread(path)
    return cv_to_pil_image(img_bgr)

