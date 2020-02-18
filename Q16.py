import numpy as np
import cv2


def BGR2GREY(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    # Gray scale
    out = 0.2126 * r + 0.7152 * g + 0.0722 * b
    out = out.astype(np.uint8)

    return out


def prewitt_filter(img, K_size=3):

    H, W = img.shape

    # zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
    out[pad: H + pad, pad: W + pad] = img.copy().astype(np.float)
    tmp = out.copy()

    out_v = out.copy()
    out_h = out.copy()

    Kv = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
    Kh = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]

    for y in range(H):
        for x in range(W):
            # out_v[x: x + K_size, y: y + K_size] = np.sum(Kv * tmp[x: x + K_size, y: y + K_size])
            # out_h[x: x + K_size, y: y + K_size] = np.sum(Kh * tmp[x: x + K_size, y: y + K_size])
            # 正解
            out_v[pad + y, pad + x] = np.sum(Kv * (tmp[y: y + K_size, x: x + K_size]))
            out_h[pad + y, pad + x] = np.sum(Kh * tmp[y: y + K_size, x: x + K_size])

    out_v = np.clip(out_v, 0, 255)
    out_h = np.clip(out_h, 0, 255)

    out_v = out_v[pad:pad + H, pad:pad + W].astype(np.uint8)
    out_h = out_h[pad:pad + H, pad:pad + W].astype(np.uint8)

    return out_v, out_h


# Read image
img = cv2.imread("./assets/imori.jpg").astype(np.float)

gray = BGR2GREY(img)

out_v, out_h = prewitt_filter(gray, K_size=3)

# Save result
cv2.imwrite("out.jpg", out_v)
cv2.imshow("result", out_v)
cv2.waitKey(0)
cv2.destroyAllWindows()
