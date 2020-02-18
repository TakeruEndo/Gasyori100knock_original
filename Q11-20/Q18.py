import numpy as np
import cv2


# GreyScale
def BGR2GRAY(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    # Gray scale
    out = 0.2126 * r + 0.7152 * g + 0.0722 * b
    out = out.astype(np.uint8)

    return out


def emboss_filter(img, K_size=3):
    H, W = img.shape

    # zero_padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)
    tmp = out.copy()

    K = [[-2., -1., 0.], [-1., 1., 1.], [0., 1., 2.]]

    # 自分の実装
    for y in range(H):
        for x in range(W):
            out[pad + y, pad + x] = np.sum(K * (tmp[y: y + K_size, x: x + K_size]))

    out = np.clip(out, 0, 255)

    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)

    return out


img = cv2.imread('./assets/imori.jpg').astype(np.float)

grey = BGR2GRAY(img)

out = emboss_filter(grey, K_size=3)

cv2.imwrite("out.jpg", out)
cv2.imshow("result", out)
cv2.waitKey(0)
cv2.destroyAllWindows()





