import cv2
import numpy as np


# function: BGR -> RGB
def BGR2GREY(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    # Gray scale
    out = 0.2126 * r + 0.7152 * g + 0.0722 * b
    out = out.astype(np.uint8)

    return out


# sobel_filter
def sobel_filter(img, K_size=3):
    H, W = img.shape

    # Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
    out[pad:pad + H, pad:pad + W] = img.copy().astype(np.float)
    tmp = out.copy()

    out_v = out.copy()
    out_h = out.copy()

    # vertical kernel
    Kv = [[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]
    # horizontal kernel
    Kh = [[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]]

    # filtering
    for y in range(H):
        for x in range(W):
            out_v[pad + y, pad + x] = np.sum(Kv * tmp[y:y + K_size, x:x + K_size])
            out_h[pad + y, pad + x] = np.sum(Kh * tmp[y:y + K_size, x:x + K_size])

    out_v = np.clip(out_v, 0, 255)
    out_h = np.clip(out_h, 0, 255)

    out_v = out_v[pad:pad + H, pad:pad + W].astype(np.uint8)
    out_h = out_h[pad:pad + H, pad:pad + W].astype(np.uint8)

    return out_v, out_h


# Read image
img = cv2.imread("./assets/imori.jpg").astype(np.float)

gray = BGR2GREY(img)

# Median Filter
out_v, out_h = sobel_filter(gray, K_size=3)

# Save result
cv2.imwrite("out.jpg", out_v)
cv2.imshow("result", out_v)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("out_h.jpg", out_h)
cv2.imshow('result', out_h)
cv2.waitKey(0)
cv2.destroyAllWindows()
