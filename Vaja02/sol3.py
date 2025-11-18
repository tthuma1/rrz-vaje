import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

### c)
im = cv2.imread('slike/coins.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

im_canny = cv2.Canny(im, threshold1=300, threshold2=450)

plt.subplot(1,2,1)
plt.imshow(im)
plt.title("Originalna slika")

plt.subplot(1,2,2)
plt.imshow(im_canny)
plt.title("Canny")

plt.show()

