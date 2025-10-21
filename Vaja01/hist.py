import numpy as np

img = np.array([[2,4,1,6,2,4],
       [4,2,4,2,2,5],
       [1,2,7,2,1,4],
       [2,1,6,7,6,5],
       [2,1,2,7,2,1],
       [2,4,4,4,4,4]])

img = np.reshape(img, -1)
# print(img)
h = np.zeros(8)
for x in img:
    h[x//2] += 1

print(h)
