# 画像の２値化

import cv2
import numpy as np


# function: BGR -> RGB
def RGB2HSV(img):
    B = img[:, :, 0].copy() / 255
    G = img[:, :, 1].copy() / 255
    R = img[:, :, 2].copy() / 255

    Max = max(B, G, R)
    Min = min(B, G, R)

    if (Min == Max):
        H = 0
    elif (Min == B):
        H = 60 * (G - R) / (Max - Min) + 60
    elif (Min == R):
        H = 60 * (B - G) / (Max - Min) + 180
    elif (Min == G):
        H = 60 * (R - B) / (Max - Min) + 300
    
    V = Max

    S = Max - Min


def binalization(img, th=128):
    img[img < th] = 0
    img[img >= th] = 255
    return img


# Read image
img = cv2.imread("assets/imori.jpg").astype(np.float)

# BGR -> RGB
out = RGB2HSV(img)

binalization(out)


# Save result
# cv2.imwrite("out.jpg", out)

cv2.imshow("imori", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
