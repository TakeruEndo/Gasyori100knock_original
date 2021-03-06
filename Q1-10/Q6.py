import cv2


def dicrease_color(img):
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
