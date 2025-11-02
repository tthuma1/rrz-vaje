import cv2
import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np
from matplotlib.widgets import Slider
import matplotlib.patches

# ### a)

# # 0 0 0 0 0 0 0 1 0
# # 0 0 0 1 0 0 0 1 1
# # 0 1 0 1 0 0 1 1 1
# # 1 1 1 1 0 1 0 0 0
# # 1 0 0 1 1 0 0 0 0
# # 0 0 0 0 0 0 0 0 1
# # 0 0 0 0 0 0 1 1 1
# # 0 1 0 0 0 1 1 1 1
# # 1 1 0 0 0 0 0 0 1

# # Prvi prehod:

# # 0  0 0 0 0 0  0 1 0
# # 0  0 0 2 0 0  0 1 1
# # 0  3 0 2 0 0  4 1 1
# # 5  3 3 2 0 6  0 0 0
# # 5  0 0 2 2 0  0 0 0
# # 0  0 0 0 0 0  0 0 7
# # 0  0 0 0 0 0  8 8 7
# # 0  9 0 0 0 10 8 8 7
# # 11 9 0 0 0 0  0 0 7

# # konflikti = ((4,1), (5,3), (3,2), (7,8), (10,8), (11, 9))
# # urejeni konflikti = ((4,1), (5,3,2) (7,8,10) (11, 9))

# # Drugi prehod:

# # 0 0 0 0 0 0 0 1 0
# # 0 0 0 2 0 0 0 1 1
# # 0 2 0 2 0 0 1 1 1
# # 2 2 2 2 0 6 0 0 0
# # 2 0 0 2 2 0 0 0 0
# # 0 0 0 0 0 0 0 0 7
# # 0 0 0 0 0 0 7 7 7
# # 0 9 0 0 0 7 7 7 7
# # 9 9 0 0 0 0 0 0 7


# ### b)

# img = cv2.imread('slike/regions.png', cv2.IMREAD_GRAYSCALE)

# _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# num_labels, labels = cv2.connectedComponents(binary)

# print("Število regij (vključno z ozadjem):", num_labels)

# plt.imshow(labels, cmap='magma')
# plt.show()


# ### c)

# fig, ax = plt.subplots()

# centroidi = []
# boxes = []

# for i in range(num_labels):
#     if i == 0: continue

#     koordinate = np.argwhere(labels == i)
#     cy, cx = koordinate.mean(axis=0)
#     centroidi.append((cx, cy))

#     y_min, x_min = koordinate.min(axis=0)
#     y_max, x_max = koordinate.max(axis=0)
#     boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

# for i in range(num_labels - 1):
#     cx, cy = centroidi[i]
#     plt.scatter(cx, cy)

#     x, y, w, h = boxes[i]
#     rect = matplotlib.patches.Rectangle((x, y), w, h, edgecolor='red', facecolor='none')
#     ax.add_patch(rect)

# plt.imshow(labels, cmap='magma')
# plt.show()


# ### d)

# # 0 0 0 0 0 0 0 0 1
# # 0 0 0 1 0 0 0 1 0
# # 0 1 0 1 0 0 1 1 0
# # 1 1 1 1 0 1 0 0 0
# # 1 0 0 1 1 0 0 0 0
# # 0 0 0 0 0 0 0 0 0
# # 0 0 0 0 0 0 1 1 0
# # 0 1 0 0 0 1 1 1 0

# # kernel:

# # 1 1 1
# # 1 1 1
# # 0 1 0

# # Razširi (dilate):

# # 0 0 0 1 0 0 0 1 1
# # 0 1 1 1 1 0 1 1 1
# # 1 1 1 1 1 1 1 1 1
# # 1 1 1 1 1 1 1 1 1
# # 1 1 1 1 1 1 1 0 0
# # 1 1 1 1 1 1 1 1 0
# # 0 1 0 0 0 1 1 1 1
# # 1 1 1 0 1 1 1 1 1

# # Skrči (erode):

# # 0 0 0 0 0 0 0 0 0
# # 0 0 0 0 0 0 0 0 0
# # 0 0 0 0 0 0 0 0 0
# # 0 0 0 0 0 0 0 0 0
# # 0 0 0 0 0 0 0 0 0
# # 0 0 0 0 0 0 0 0 0
# # 0 0 0 0 0 0 0 0 0
# # 0 0 0 0 0 0 0 0 0


# ### e)

# img = cv2.imread('slike/regions_noise.png', cv2.IMREAD_GRAYSCALE)

# _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# num_labels, labels = cv2.connectedComponents(binary)

# print("Število regij regions_noise (vključno z ozadjem):", num_labels)

# plt.subplot(1,2,1)
# plt.imshow(binary, cmap='gray')
# plt.title('Binariziran regions_noise')

# plt.subplot(1,2,2)
# plt.imshow(labels, cmap='magma')
# plt.title('Regije regions_noise')
# plt.show()

# # Dobimo zelo veliko zelo majhnih regij. Piksli šuma postanejo svoje regije.

# ### f)

# plt.subplot(2,2,1)
# plt.imshow(binary, cmap='gray')
# plt.title('Original')

# erode = cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)))
# dilate = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
# dilate2 = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

# plt.subplot(2,2,2)
# plt.imshow(erode, cmap='gray')
# plt.title('Erode (cross 3x3)')

# plt.subplot(2,2,3)
# plt.imshow(dilate, cmap='gray')
# plt.title('Dilate (rect 3x3)')

# plt.subplot(2,2,4)
# plt.imshow(dilate2, cmap='gray')
# plt.title('Dilate (ellipse 5x5)')

# plt.show()


# ### g)

# # ročno
# erode = cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
# opened = cv2.dilate(erode, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))

# dilate = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
# closed = cv2.erode(dilate, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))

# plt.subplot(3,2,1)
# plt.imshow(binary, cmap='gray')
# plt.title('Original')

# plt.subplot(3,2,3)
# plt.imshow(opened, cmap='gray')
# plt.title('Odpiranje')

# plt.subplot(3,2,5)
# plt.imshow(closed, cmap='gray')
# plt.title('Zapiranje')

# # morphologyEx

# opened2 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
# closed2 = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))

# plt.subplot(3,2,4)
# plt.imshow(opened2, cmap='gray')
# plt.title('Odpiranje (morphologyEx)')

# plt.subplot(3,2,6)
# plt.imshow(closed2, cmap='gray')
# plt.title('Zapiranje (morphologyEx)')

# # erode + close
# opened3 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
# no_noise = cv2.morphologyEx(opened3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))

# plt.subplot(3,2,2)
# plt.imshow(no_noise, cmap='gray')
# plt.title('Brez šuma')

# plt.tight_layout()
# plt.show()


### h)

img = cv2.imread('slike/bird.jpg')

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue = img_hsv[..., 0]
value = img_hsv[..., 2]
thres_hue =  (value > 55)
# thres_hue = (10 <= hue) & (hue <= 12) | (170 <= hue) & (hue <= 170) & (value > 50)

img = cv2.imread('slike/bird.jpg', cv2.IMREAD_GRAYSCALE)

ret2, otsu = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

gauss = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

otsu = otsu == 255.0
img_thresh2 = thres_hue | otsu
img_thresh2 = img_thresh2.astype(np.float32) * 255.0

gauss = gauss == 255.0
gauss_opened = cv2.morphologyEx((~gauss).astype(np.float32) * 255.0, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_DIAMOND, (3,3)))
gauss_opened = gauss_opened == 255.0
img_thresh2 = (gauss_opened | otsu | thres_hue).astype(np.float32) * 255.0

closed = cv2.morphologyEx(img_thresh2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_DIAMOND, (3,3)))
no_noise = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_DIAMOND, (3,3)))
no_noise = cv2.morphologyEx(no_noise, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5)))
no_noise = cv2.morphologyEx(no_noise, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
no_noise = cv2.morphologyEx(no_noise, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))



plt.subplot(2,3,1)
plt.imshow(img, cmap='gray')
plt.title('Original siva')


plt.subplot(2,3,2)
plt.imshow(gauss_opened, cmap='gray')
plt.title("Upragovanje z Otsujevo metodo")

plt.subplot(2,3,3)
plt.imshow(otsu, cmap='gray')
plt.title("Adaptivno Gaussovo upragovanje")

plt.subplot(2,3,4)
plt.imshow(thres_hue, cmap='gray')
plt.title("hue only")

plt.subplot(2,3,5)
plt.imshow(img_thresh2, cmap='gray')
plt.title("hue + otsu")

plt.subplot(2,3,6)
plt.imshow(no_noise, cmap='gray')
plt.title('Brez šuma')

plt.tight_layout()
plt.show()


def hsv_thresholding(filepath):
    img = cv2.imread(filepath)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hue = img_hsv[..., 0]
    value = img_hsv[..., 2]
    sat = img_hsv[..., 1]

    low_init, high_init = 90, 150

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.subplots_adjust(bottom=0.25)

    ax1.imshow(img_rgb)
    ax1.set_title('RGB slika')

    img_thresh = (low_init <= hue) & (hue <= high_init)  & (value > 0.9) & (sat > 0.9)
    im2 = ax2.imshow(img_thresh, cmap='gray')
    ax2.set_title(f'{low_init} <= H <= {high_init}')

    ax_low = plt.axes([0.25, 0.1, 0.55, 0.03])
    ax_high = plt.axes([0.25, 0.05, 0.55, 0.03])
    slider_low = Slider(ax_low, 'Spodnja meja', 0, 179, valinit=low_init, valstep=1)
    slider_high = Slider(ax_high, 'Zgornja meja', 0, 179, valinit=high_init, valstep=1)

    # Update callback
    def update(_):
        low = slider_low.val
        high = slider_high.val
        img_thresh = (hue >= low) & (hue <= high) & (value > 50)
        im2.set_data(img_thresh)
        ax2.set_title(f'{low} <= H <= {high}')
        fig.canvas.draw_idle()

    slider_low.on_changed(update)
    slider_high.on_changed(update)

    plt.show()

hsv_thresholding('slike/bird.jpg')


### --------------

# I = I.astype(np.uint8)
# k = np.array()
# k = k.astype(np.uint8)
             
# er = cv2.erode(I, k)
# er = cv2.dilate(I, k)

# cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3)) namesto ročno np.array()
# cv2.open() in cv2.close() za opening in closing

