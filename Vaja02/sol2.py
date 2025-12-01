import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def simple_gauss(sigma):
    x = np.linspace(-3*sigma, 3*sigma, 2 * math.ceil(3 * sigma) + 1)
    # res = (1 / (sigma * math.sqrt(2 * math.pi))) * np.exp(-x**2 / (2 * sigma**2))
    res = np.exp(-x**2 / (2 * sigma**2)) # prvi faktor lahko spustiš, ker itak potem normaliziraš (prvi faktor samo raztegne po y-osi)
    res /= np.sum(res)
    
    return res

### a)

def simple_gaussdx(sigma):
    x = np.linspace(-3*sigma, 3*sigma, 2 * math.ceil(3 * sigma) + 1)
    # res = - 1/ (sigma**3 * np.sqrt(2*math.pi)) * x * np.exp(-x**2 / (2 * sigma**2))
    res = -x * np.exp(-x**2 / (2 * sigma**2)) # prvi faktor lahko spustiš, ker itak potem normaliziraš (prvi faktor samo raztegne po y-osi)
    res /= 1/2 * np.sum(np.abs(res))
    
    return res

k = simple_gaussdx(1)
print("Odvod Gaussovega jedra (1):", k)
print("Vsota elementov v polovici (1):", np.sum(k[:len(k)//2]))

plt.subplot(1,2,1)
plt.plot(k)
plt.title("Odvod Gaussa (1)")

plt.subplot(1,2,2)
plt.plot(simple_gaussdx(2))
plt.title("Odvod Gaussa (2)")

plt.tight_layout()
plt.show()


im = cv2.imread("slike/museum.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def simple_gaussdx_2D(sigma, im):
    im = np.mean(im, axis=2)
    kx = np.expand_dims(simple_gaussdx(sigma), axis=0)

    im_edges = cv2.filter2D(cv2.filter2D(im, -1, kx), -1, kx.T)

    plt.subplot(2,2,1)
    plt.imshow(kx.T @ kx, cmap='gray')
    plt.title("Jedro odvoda Gaussa 2D (sigma=" + str(sigma) + ")")

    plt.subplot(2,2,2)
    plt.imshow(im, cmap='gray')
    plt.title("Original")

    plt.subplot(2,2,3)
    plt.imshow(im_edges, cmap='gray')
    plt.title("Original")

    plt.tight_layout()
    plt.show()

    # print()
    # print("2D odvod Gaussa (" + str(sigma) + "):", kx)

simple_gaussdx_2D(2, im)

### b)

# Ali je zaporedje ukazov pomembno?
#     Zaporedje ukazov ni pomembno, ker je konvolucija komutativna.

size = 101
dirac = np.zeros((size, size))
dirac[size//2, size//2] = 1

G = np.expand_dims(simple_gauss(7), axis=0)
D = np.expand_dims(simple_gaussdx(7), axis=0)
GT = G.T
DT = D.T

plt.subplot(2,3,1)
plt.imshow(dirac, cmap='gray')
plt.title("I")

plt.subplot(2,3,4)
plt.imshow(cv2.filter2D(cv2.filter2D(dirac, -1, G), -1, GT), cmap='gray')
plt.title("(I * G) * Gᵀ")

plt.subplot(2,3,2)
plt.imshow(cv2.filter2D(cv2.filter2D(dirac, -1, G), -1, DT), cmap='gray')
plt.title("(I * G) * Dᵀ")

plt.subplot(2,3,3)
plt.imshow(cv2.filter2D(cv2.filter2D(dirac, -1, D), -1, GT), cmap='gray')
plt.title("(I * D) * Gᵀ")

plt.subplot(2,3,5)
plt.imshow(cv2.filter2D(cv2.filter2D(dirac, -1, GT), -1, D), cmap='gray')
plt.title("(I * Gᵀ) * D")

plt.subplot(2,3,6)
plt.imshow(cv2.filter2D(cv2.filter2D(dirac, -1, DT), -1, G), cmap='gray')
plt.title("(I * Dᵀ) * G")

plt.tight_layout()
plt.show()

