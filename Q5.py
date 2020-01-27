import cv2
import numpy as np


# BGR -> HSV
def BGR2HSV(_img):

    img = _img.copy() / 255.
    
    hsv = np.zeros_like(img, dtype=np.float32)

    max_v = np.max(img, axis=2).copy()
    min_v = no.min(img, axis=2).copy()
    min_arg = np.arg(img, axis=2)

    

