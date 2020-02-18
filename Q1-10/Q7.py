# 画像の２値化

import cv2


def dicrease_color(img):
    for i in range(17):
    img[0:8, i:i + 8]
    out = img.copy()

    out = out // 64 * 64 + 32

    return out


# Read image
img = cv2.imread("assets/imori.jpg")

# Dicrease color
out = dicrease_color(img)

# Save result
# cv2.imwrite("out.jpg", out)

cv2.imshow("imori", out)
cv2.waitKey(0)
cv2.destroyAllWindows()



>>> img2 = img.copy()
>>> img2[:50, :50] = 0