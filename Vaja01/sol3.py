import numpy as np
import matplotlib.pyplot as plt
import cv2

### a)

img = np.array([
    [5,3,2,7,1],
    [7,1,0,0,0],
    [4,5,7,1,1],
    [1,3,2,1,1],
    [5,3,1,6,3],
])

img = np.reshape(img, -1)
# print(img)
h = np.zeros(8)
for x in img:
    h[x] += 1

print("Histogram slike (3-bit):", h)

plt.clf()
plt.bar(np.linspace(0, 7, 8), h)
plt.title("Histogram slike (3-bit)")
plt.show()

### b) kumulativni histogram

h_kum = np.zeros(8)
for i in range(len(h)):
    h_kum[i] = h_kum[i - 1] + h[i] if i > 0 else h[i]

print("Kumulativni histogram (3-bit):", h_kum)

plt.clf()
plt.bar(np.linspace(0, 7, 8), h_kum)
plt.title("Kumulativni histogram (3-bit)")
plt.show()

### c) 4-bitna slika

h = np.zeros(16)
for x in img:
    h[x] += 1

print("Histogram slike (4-bit):", h)

h_kum = np.zeros(16)
for i in range(len(h)):
    h_kum[i] = h_kum[i - 1] + h[i] if i > 0 else h[i]

print("Kumulativni histogram (4-bit):", h_kum)

plt.clf()

plt.subplot(1, 2, 1)
plt.bar(np.linspace(0, 15, 16), h)
plt.title("Histogram slike (4-bit)")

plt.subplot(1, 2, 2)
plt.bar(np.linspace(0, 15, 16), h_kum)
plt.title("Kumulativni histogram (4-bit)")

plt.show()

### d)
print()

img = cv2.imread('slike/umbrellas.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray_im = np.mean(img, axis=2)

hist, _ = np.histogram(gray_im.flatten(), bins=8)
print("Histogram umbrellas z numpy (bins=8):", hist)

plt.clf()
plt.title("Histogram umbrellas z numpy (bins=8)")
plt.bar(np.linspace(0, 255, 8), hist, width=256/8)
plt.show()

print()

hist, _ = np.histogram(gray_im.flatten(), bins=256)
# print("Histogram umbrellas z numpy (bins=256):", hist)

plt.clf()
plt.title("Histogram umbrellas z numpy (bins=256)")
plt.bar(np.linspace(0, 255, 256), hist, width=1)
plt.show()


img = np.array([
    [5,3,2,7,1],
    [7,1,0,0,0],
    [4,5,7,1,1],
    [1,3,2,1,1],
    [5,3,1,6,3],
])

hist, _ = np.histogram(img, bins=8)
hist2, _ = np.histogram(img, bins=16, range=(0, 15))

plt.clf()

plt.subplot(1, 2, 1)
plt.title("Histogram slike z numpy (3-bit)")
plt.bar(np.linspace(0, 7, 8), hist, width=1)

plt.subplot(1, 2, 2)
plt.title("Histogram slike z numpy (4-bit)")
plt.bar(np.linspace(0, 15, 16), hist2, width=1)

plt.show()