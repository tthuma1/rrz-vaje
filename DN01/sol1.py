import cv2
import matplotlib.pyplot as plt
import numpy as np

ime_slike = 'slike/umbrellas.jpg'
im = cv2.imread(ime_slike)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

print(f'{im.shape=}')
print(f'{im.dtype=}')

# plt.clf()
# plt.imshow(im)
# plt.show()

# gray_im = im.astype(np.float32)
# for i in range(im.shape[0]):
#     for j in range(im.shape[1]):
#         gray_im[i,j] = (gray_im[i,j,0] + gray_im[i,j,1] + gray_im[i,j,2]) / 3
# gray_im = gray_im.astype(np.uint8)

gray_im = np.mean(im, axis=2)

gray_im = im.astype(np.float32)
gray_im = (gray_im[..., 0] + gray_im[..., 1] + gray_im[..., 2]) / 3
gray_im = gray_im.astype(np.uint8)
# ... = poberi vse iz ostalih dimenzij (vse razen zadnje dimenzije ostane isto)

plt.clf()
plt.imshow(gray_im)
plt.title('Sivinska slika, default cmap')
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
