import cv2
import matplotlib.pyplot as plt
import numpy as np

ime_slike = 'slike/umbrellas.jpg'
im = cv2.imread(ime_slike)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

print(f'{im.shape=}')
print(f'{im.dtype=}')

plt.clf()
plt.imshow(im)
plt.title('Originalna slika')
plt.show()

# gray_im = im.astype(np.float32)
# for i in range(im.shape[0]):
#     for j in range(im.shape[1]):
#         gray_im[i,j] = np.mean(gray_im[i,j])
# gray_im = gray_im.astype(np.uint8)

gray_im = np.mean(im, axis=2)

plt.clf()
plt.imshow(gray_im)
plt.title('Sivinska slika, default cmap')
plt.show()

plt.clf()
plt.imshow(gray_im, cmap='gray')
plt.title('Sivinska slika, gray cmap')
plt.show()

top = 100
bottom = 250
left = 100
right = 450

cropped = im[top:bottom, left:right, :]
im_copy = im.copy()
im_copy[top:bottom, left:right, 2] = 0

plt.clf()
plt.subplot(1, 2, 1)
plt.title('Pravokotna regija')
plt.imshow(cropped)

plt.subplot(1, 2, 2)
plt.title('Pravokotna regija je brez modre')
plt.imshow(im_copy)
plt.show()

neg_gray = gray_im.copy()
# če bi imeli float, bi rabili dati na range 0..1 in potem odštevati od 1
neg_gray[top:bottom, left:right] = 255 - gray_im[top:bottom, left:right]
plt.clf()
plt.title('Pravokotna regija je negirana')
plt.imshow(neg_gray, cmap='gray')
plt.show()

thresholded = (gray_im > 150).astype(np.uint8)
plt.clf()
plt.title('Upragovana slika')
plt.imshow(thresholded, cmap='gray')
plt.show()
