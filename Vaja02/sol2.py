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
