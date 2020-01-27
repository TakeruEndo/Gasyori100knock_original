# 画像の２値化

import cv2
import numpy as np


# function: BGR -> RGB
def BGR2GREY(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    out = 0.2126 * r + 0.7152 * g + 0.0722 * b
    # 画素値はint出ないといけない[0~255]で表現される
    out = out.astype(np.uint8)

    return out


def binalization(img, th=128):
    img[img < th] = 0
    img[img >= th] = 255
    return img


# Read image
img = cv2.imread("assets/imori.jpg").astype(np.float)

# BGR -> RGB
out = BGR2GREY(img)

binalization(out)


# Save result
# cv2.imwrite("out.jpg", out)

cv2.imshow("imori", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
