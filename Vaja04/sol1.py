import cv2
import matplotlib.pyplot as plt
import numpy as np

### a)

ime_slike = 'capture_f1.jpg'
im = cv2.imread(ime_slike)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# print(im.shape)

# plt.clf()
# plt.imshow(im)
# plt.title('Originalna slika')

# src_pts = np.float32(plt.ginput(4)).reshape(-1,1,2)
# np.save('tocke', src_points)

src_pts = np.load('tocke.npy')
dst_pts = np.float32([[[0, 0]],
                      [[400, 0]],
                      [[400, 270]],
                      [[0, 270]]])

H, mask = cv2.findHomography(src_pts, dst_pts)
im2 = cv2.warpPerspective(im, H, (400, 270))

plt.subplot(1,2,1)
plt.imshow(im)
plt.title('Original')

plt.subplot(1,2,2)
plt.imshow(im2)
plt.title('Warped')

plt.tight_layout()
plt.show()

### b)

xs = np.linspace(0, 400, 41)
ys = np.linspace(0, 270, 28)

xv, yv = np.meshgrid(xs, ys)

mesh_points = np.vstack((np.reshape(xv, -1), np.reshape(yv, -1), np.ones_like(np.reshape(xv, -1))))

H_inv = np.linalg.inv(H)

mesh_points_inv = H_inv @ mesh_points
mesh_points_inv /= mesh_points_inv[-1,:]

plt.subplot(1,2,1)
plt.imshow(im)
plt.plot(mesh_points_inv[0,:], mesh_points_inv[1,:], 'r.', markersize=0.5)
plt.title('Original')

plt.subplot(1,2,2)
plt.imshow(im2)
plt.plot(xv, yv, 'r.', markersize=0.5)
plt.title('Warped')

plt.tight_layout()
plt.show()


### c)

