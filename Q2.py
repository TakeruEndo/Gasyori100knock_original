# RGBをBGRに入れ替える

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
    print(out)

    return out


# Read image
img = cv2.imread("assets/imori.jpg").astype(np.float)

# BGR -> RGB
out = BGR2GREY(img)

# Save result
# cv2.imwrite("out.jpg", out)
cv2.imshow("imori", out)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 実行結果
# .astype(np.uint8)がないと画像が表示されない
