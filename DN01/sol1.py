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
plt.show()