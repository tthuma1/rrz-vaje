import numpy as np
import matplotlib.pyplot as plt

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
