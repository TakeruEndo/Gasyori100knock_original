# 大津の２値化
# Sb^2 = w0 * w1 * (M0 - M1) ^2が最大となるthを探す

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


def deceide_th(img):
    sigm_max = 0
    max_th = 0
    H, W = img.shape
    for i in range(1, 256):
        v0 = img[np.where(img < i)]
        m0 = np.mean(v0) if len(v0) > 0 else 0
        w0 = len(v0) / (H * W)
        v1 = img[np.where(img >= i)]
        m1 = np.mean(v1) if len(v1) > 0 else 0
        w1 = len(v1) / (H * W)
        sigm = w0 * w1 * ((m0 - m1) ** 2)
        if (sigm > sigm_max):
            sigm_max = sigm
            max_th = i
    return max_th


def binalization(img, th):
    img[img < th] = 0
    img[img >= th] = 255
    return img


# Read image
img = cv2.imread("assets/imori.jpg").astype(np.float)

# BGR -> RGB
out = BGR2GREY(img)

th = deceide_th(out)

binalization(out, th)

cv2.imshow("imori", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
