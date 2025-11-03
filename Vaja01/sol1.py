import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

### a)

ime_slike = 'slike/umbrellas.jpg'
im = cv2.imread(ime_slike)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

print(f'{im.shape=}')
print(f'{im.dtype=}')

plt.clf()
plt.imshow(im)
plt.show()

### b)

# Prvi način
# gray_im = im.astype(np.float32)
# for i in range(im.shape[0]):
#     for j in range(im.shape[1]):
#         gray_im[i,j] = (gray_im[i,j,0] + gray_im[i,j,1] + gray_im[i,j,2]) / 3
# gray_im = gray_im.astype(np.uint8)

# Drugi način
# gray_im = np.mean(im, axis=2)

# Tretji način
gray_im = im.astype(np.float32)
gray_im = (gray_im[..., 0] + gray_im[..., 1] + gray_im[..., 2]) / 3
gray_im = gray_im.astype(np.uint8)
# ... = poberi vse iz ostalih dimenzij (vse razen zadnje dimenzije ostane isto)

plt.clf()
plt.subplot(2,2,1)
plt.imshow(gray_im)
plt.title('Sivinska slika, default cmap')

plt.subplot(2,2,2)
plt.imshow(gray_im, cmap='gray')
plt.title('Sivinska slika, cmap=gray')

plt.subplot(2,2,3)
plt.imshow(gray_im, cmap='jet')
plt.title('Sivinska slika, cmap=jet')

plt.subplot(2,2,4)
plt.imshow(gray_im, cmap='bone')
plt.title('Sivinska slika, cmap=bone')

plt.show()

### c)

top = 100
bottom = 250
left = 100
right = 450

cropped = im[top:bottom, left:right, :]
im_copy = im.copy()
im_copy[top:bottom, left:right, 2] = 0

plt.clf()
plt.subplot(1, 2, 1) # visina, sirina, indeks
plt.title('Pravokotna regija')
plt.imshow(cropped)

plt.subplot(1, 2, 2)
plt.title('Pravokotna regija je brez modre')
plt.imshow(im_copy)
plt.show()

### d)

neg_gray = gray_im.copy()

cut = neg_gray[top:bottom, left:right]
# cut[...] = 255 - cut[...]
# cut[..., :] = 255 - cut[..., :]
cut = 255 - cut
neg_gray[top:bottom, left:right] = cut

plt.clf()
plt.title('Pravokotna regija je negirana')
plt.imshow(neg_gray, cmap='gray')
plt.show()

# Negiranje uint8 => 255 - x
# Negiranje float => 1.0 - x
# Negiranje int16 => (2**15 - 1) - x

### e)

### Threshold s sliderjem

def apply_threshold(data, thresh):
    return np.where(data > thresh, 1, 0)
    # return (data > thresh).astype(np.uint8)

th_init = 150
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
im = ax.imshow(apply_threshold(gray_im, th_init), cmap='gray')
ax.set_title('Upragovana slika')

ax_thresh = plt.axes([0.25, 0.1, 0.65, 0.03])
slider = Slider(ax=ax_thresh, label='Threshold', valmin=0, valmax=255, valinit=th_init)

def update(val):
    thresh = slider.val
    im.set_data(apply_threshold(gray_im, thresh))
    fig.canvas.draw_idle()

# thresholded = gray_im > 150
slider.on_changed(update)
plt.show()
