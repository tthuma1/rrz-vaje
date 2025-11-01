import cv2
import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np
from matplotlib.widgets import Slider

### a)

# 0 0 0 0 0 0 0 1 0
# 0 0 0 1 0 0 0 1 1
# 0 1 0 1 0 0 1 1 1
# 1 1 1 1 0 1 0 0 0
# 1 0 0 1 1 0 0 0 0
# 0 0 0 0 0 0 0 0 1
# 0 0 0 0 0 0 1 1 1
# 0 1 0 0 0 1 1 1 1
# 1 1 0 0 0 0 0 0 1

# Prvi prehod:

# 0  0 0 0 0 0  0 1 0
# 0  0 0 2 0 0  0 1 1
# 0  3 0 2 0 0  4 1 1
# 5  3 3 2 0 6  0 0 0
# 5  0 0 2 2 0  0 0 0
# 0  0 0 0 0 0  0 0 7
# 0  0 0 0 0 0  8 8 7
# 0  9 0 0 0 10 8 8 7
# 11 9 0 0 0 0  0 0 7

# konflikti = ((4,1), (5,3), (3,2), (7,8), (10,8), (11, 9))
# urejeni konflikti = ((4,1), (5,3,2) (7,8,10) (11, 9))

# Drugi prehod:

# 0 0 0 0 0 0 0 1 0
# 0 0 0 2 0 0 0 1 1
# 0 2 0 2 0 0 1 1 1
# 2 2 2 2 0 6 0 0 0
# 2 0 0 2 2 0 0 0 0
# 0 0 0 0 0 0 0 0 7
# 0 0 0 0 0 0 7 7 7
# 0 9 0 0 0 7 7 7 7
# 9 9 0 0 0 0 0 0 7


### b)

img = cv2.imread('slike/regions.png', cv2.IMREAD_GRAYSCALE)

_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
num_labels, labels = cv2.connectedComponents(binary)

print("Število regij (vključno z ozadjem):", num_labels)

plt.imshow(labels, cmap='magma')
plt.show()




# I = I.astype(np.uint8)
# k = np.array()
# k = k.astype(np.uint8)
             
# er = cv2.erode(I, k)
# er = cv2.dilate(I, k)

# cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)) namesto ročno np.array()
# cv2.open() in cv2.close() za opening in closing