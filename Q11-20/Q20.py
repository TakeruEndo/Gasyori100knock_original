import cv2
import numpy as np
import matplotlib.pyplot as plt

# ReadImage
img = cv2.imread("./assets/imori_dark.jpg").astype(np.float)

# DisplayHistgram
plt.hist(img.ravel(), bins=255, rwidth=0.8, range=(0, 255))
plt.savefig('out.png')
plt.show()
