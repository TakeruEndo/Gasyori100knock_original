import cv2
import numpy as np

# Read image
img = cv2.imread("assets/imori.jpg").astype(np.float32)

# 画像の形式
# 縦の画素(axis=0) * 横の画素(axis=1) * kernel(RGB)(axis=2)
print(img.shape)

img_max_axis_0 = np.max(img, axis=0).copy()
# 行方向の圧縮(最大値)
print(img_max_axis_0.shape)

img_max_axis_1 = np.max(img, axis=1).copy()
# 列方向の圧縮(最大値)
print(img_max_axis_1.shape)

img_max_axis_2 = np.max(img, axis=2).copy()
# RGBの中の最大値を返す
print(img_max_axis_2.shape)

# 参考文献
# https://deepage.net/features/numpy-axis.html

# hsv[..., 0]とは！？

a = np.arange(6).reshape((3, 2))
b = np.array([a, [[3, 5], [2, 8], [1, 10]]])
print(b)
print(b[..., 0])
print(b[0, ...][1])
