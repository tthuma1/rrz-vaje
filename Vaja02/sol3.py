import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def simple_gaussdx(sigma):
    x = np.linspace(-3*sigma, 3*sigma, 2 * math.ceil(3 * sigma) + 1)
    # res = - 1/ (sigma**3 * np.sqrt(2*math.pi)) * x * np.exp(-x**2 / (2 * sigma**2))
    res = -x * np.exp(-x**2 / (2 * sigma**2)) # prvi faktor lahko spustiš, ker itak potem normaliziraš (prvi faktor samo raztegne po y-osi)
    res /= 1/2 * np.sum(np.abs(res))
    
    return res

### a)

def gradient_magnitude(I):
    k = np.expand_dims(simple_gaussdx(2), axis=0)
    Ix = cv2.filter2D(I, -1, k)
    Iy = cv2.filter2D(I, -1, k.T)

    mag = np.sqrt(Ix**2 + Iy**2)
    dir = np.arctan2(Iy, Ix)

    return Ix, Iy, mag, dir

I = cv2.imread("slike/museum.jpg", cv2.COLOR_BGR2RGB)
I = np.mean(I, axis=2)

Ix, Iy, mag, dir = gradient_magnitude(I)

plt.subplot(2, 3, 1)
plt.title("I")
plt.imshow(I, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.title("Ix")
plt.imshow(Ix, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.title("Iy")
plt.imshow(Iy, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.title("I_mag")
plt.imshow(mag, cmap="gray")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.title("I_dir")
plt.imshow(dir, cmap="gray")
plt.axis("off")

mag_norm = mag / mag.max()

# dir ima interval [-pi, pi], zato ga premaknes na [0, 2pi], potem [0, 360], potem [0, 180]
H = ((dir + np.pi) * 180 / np.pi / 2).astype(np.uint8)
S = np.full_like(H, 255, dtype=np.uint8)
V = (mag_norm * 255).astype(np.uint8)

H = (H + 90) % 180 # na primeru slike so barve zarotirane za 90 stopinj

dir_hsv = np.dstack((H, S, V))
dir_rgb = cv2.cvtColor(dir_hsv, cv2.COLOR_HSV2RGB)

plt.subplot(2, 3, 6)
plt.title("I_dir(HSV)")
plt.imshow(dir_rgb, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()



### c)
im = cv2.imread('slike/coins.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

im_canny = cv2.Canny(im, threshold1=300, threshold2=450)

plt.subplot(1,2,1)
plt.imshow(im)
plt.title("Originalna slika")

plt.subplot(1,2,2)
plt.imshow(im_canny)
plt.title("Canny")

plt.show()

