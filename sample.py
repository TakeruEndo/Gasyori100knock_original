import cv2
# import matplotlib.pyplot as plt
# import numpy as np

img = cv2.imread("assets/imori.jpg")
cv2.imshow("imori", img)
cv2.waitKey(0)
cv2.destroyAllWindows()