import cv2
import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np
from matplotlib.widgets import Slider
import matplotlib.patches

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


### c)

fig, ax = plt.subplots()

centroidi = []
boxes = []

for i in range(num_labels):
    if i == 0: continue

    koordinate = np.argwhere(labels == i)
    cy, cx = koordinate.mean(axis=0)
    centroidi.append((cx, cy))

    y_min, x_min = koordinate.min(axis=0)
    y_max, x_max = koordinate.max(axis=0)
    boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

for i in range(num_labels - 1):
    cx, cy = centroidi[i]
    plt.scatter(cx, cy)

    x, y, w, h = boxes[i]
    rect = matplotlib.patches.Rectangle((x, y), w, h, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

plt.imshow(labels, cmap='magma')
plt.show()


### d)

# 0 0 0 0 0 0 0 0 1
# 0 0 0 1 0 0 0 1 0
# 0 1 0 1 0 0 1 1 0
# 1 1 1 1 0 1 0 0 0
# 1 0 0 1 1 0 0 0 0
# 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 1 1 0
# 0 1 0 0 0 1 1 1 0

# kernel:

# 1 1 1
# 1 1 1
# 0 1 0

# Razširi (dilate):

# 0 0 0 1 0 0 0 1 1
# 0 1 1 1 1 0 1 1 1
# 1 1 1 1 1 1 1 1 1
# 1 1 1 1 1 1 1 1 1
# 1 1 1 1 1 1 1 0 0
# 1 1 1 1 1 1 1 1 0
# 0 1 0 0 0 1 1 1 1
# 1 1 1 0 1 1 1 1 1

# Skrči (erode):

# 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0
# 0 0 0 0 0 0 0 0 0


### e)

img = cv2.imread('slike/regions_noise.png', cv2.IMREAD_GRAYSCALE)

_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
num_labels, labels = cv2.connectedComponents(binary)

print("Število regij regions_noise (vključno z ozadjem):", num_labels)

plt.subplot(1,2,1)
plt.imshow(binary, cmap='gray')
plt.title('Binariziran regions_noise')

plt.subplot(1,2,2)
plt.imshow(labels, cmap='magma')
plt.title('Regije regions_noise')
plt.show()

# Dobimo zelo veliko zelo majhnih regij. Piksli šuma postanejo svoje regije.

### f)

plt.subplot(2,2,1)
plt.imshow(binary, cmap='gray')
plt.title('Original')

erode = cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)))
dilate = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
dilate2 = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

print(cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)))

plt.subplot(2,2,2)
plt.imshow(erode, cmap='gray')
plt.title('Erode (cross 3x3)')

plt.subplot(2,2,3)
plt.imshow(dilate, cmap='gray')
plt.title('Dilate (rect 3x3)')

plt.subplot(2,2,4)
plt.imshow(dilate2, cmap='gray')
plt.title('Dilate (ellipse 5x5)')

plt.show()


### g)






# I = I.astype(np.uint8)
# k = np.array()
# k = k.astype(np.uint8)
             
# er = cv2.erode(I, k)
# er = cv2.dilate(I, k)

# cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)) namesto ročno np.array()
# cv2.open() in cv2.close() za opening in closing