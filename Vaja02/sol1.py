import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

### a)

# konvolucija
# f = [0, 1, 1, 1, 0, 0.7, 0.5, 0.2, 0, 0, 1, 0]
# k = [0.5, 1, 0.3]
# obrnjen kernel = [0.3, 1, 0.5]
# res = [  1.5, 1.8, 1.3, 0.65, 0.95, 0.81, 0.35, 0.06, 0.5, 1  ]

### b)

def simple_convolution(signal, kernel):
    # kernel = kernel[::-1] # imamo simetricno jedro
    N = (len(kernel) - 1) // 2
    res = []
    for i in range(N, len(signal) - N):
        # sum = 0
        # for j in range(len(kernel)):
        #     sum += kernel[j] * signal[i-N + j]

        # res.append(sum)

        res.append(np.dot(signal[i-N:i+N+1], kernel))

    return res


signal = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0,
0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

kernel = [0.0022181959, 0.0087731348, 0.027023158, 0.064825185,
0.12110939, 0.17621312, 0.19967563, 0.17621312, 0.12110939, 0.064825185,
0.027023158, 0.0087731348, 0.0022181959]
print(sum(kernel))

conv1 = simple_convolution(signal, kernel)
print("Rezultat konvolucije:", [round(x, 5) for x in np.array(conv1).tolist()])

signal2 = [0, 1, 1, 1, 0, 0.7, 0.5, 0.2, 0, 0, 1, 0]
kernel2 = [0.5, 1, 0.3]

# print()
# print("Rezultat konvolucije:", simple_convolution(signal2, kernel2))

# Ali prepoznate obliko jedra kernel? Kakšna je vsota vseh elementov jedra in zakaj je to pomembno?
#     Oblika jedra je enaka gaussovi funkciji. Vsota elementov jedra je enaka 1.
#     To je pomembno, da se amplituda signala ne spremeni (ne gre čez 1).

### c)

conv2 = np.convolve(signal, kernel, mode='valid')

print()
print("Rezultat konvolucije z numpy:", conv2)
print("Ali sta rezultata enaka:", np.allclose(conv1, conv2))

# V čem se ta funkcije razlikuje od vaše implementacije simple_convolution()?
#     np.convolve obravnava še robne vrednosti z zero paddingom (okoli signala doda N ničel).
#     S parametrom `mode` lahko nastavimo, koliko robnih vrednosti obravnava.
#     Parameter 'same' premika sredino jedro od prvega elementa signala do zadnjega.


### d)

def simple_gauss(sigma):
    x = np.linspace(-3*sigma, 3*sigma, 2 * math.ceil(3 * sigma) + 1)
    # res = (1 / (sigma * math.sqrt(2 * math.pi))) * np.exp(-x**2 / (2 * sigma**2))
    res = np.exp(-x**2 / (2 * sigma**2)) # prvi faktor lahko spustiš, ker itak potem normaliziraš (prvi faktor samo raztegne po y-osi)
    res /= np.sum(res)
    
    return res

print()
print("Gaussovo jedro (1):", simple_gauss(1))
print("Vsota elementov v Gaussovem jedru (1):", np.sum(simple_gauss(1)))

### e)

print()
def draw_gauss(sigmas):
    for i, sigma in enumerate(sigmas):
        kernel = simple_gauss(sigma)
        x = np.linspace(-3*sigma, 3*sigma, 2 * math.ceil(3 * sigma) + 1)

        print("Vsota elementov v Gaussovem jedru (" + str(sigma) + "):", np.sum(kernel))

        plt.subplot(2,3,i+1)
        plt.plot(x, kernel)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Gauss sigma = " + str(sigma))

    plt.tight_layout()
    plt.show()

draw_gauss([0.5, 1, 2, 3, 4])

### f)

# Prvo jedro zgledi robove, drugo jedro zazna robove.
# Pri drugem jedru bo naraščanje signala naredilo špico gor, padanje pa špico dol.
# Hitreje kot signal narašča/pada, večja bo špica.

### g)

signal = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0,
0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

k1 = simple_gauss(2)
k2 = np.array([0.1, 0.6, 0.4])

conv1 = np.convolve(np.convolve(signal, k1), k2)
conv2 = np.convolve(np.convolve(signal, k2), k1)
conv3 = np.convolve(signal, np.convolve(k1, k2))

print()
print("(signal ⊗ k1) ⊗ k2:", [round(x, 2) for x in conv1.tolist()])
print("(signal ⊗ k2) ⊗ k1:", [round(x, 2) for x in conv2.tolist()])
print("signal ⊗ (k1 ⊗ k2):", [round(x, 2) for x in conv3.tolist()])
print("Ali so vsi trije signali enakovredni:", np.all([np.allclose(conv1, conv2), np.allclose(conv2, conv3), np.allclose(conv1, conv3)]))

plt.subplot(2,2,1)
plt.plot(np.arange(len(signal)), signal)
plt.xlabel("x")
plt.ylabel("y")
plt.title("signal")

plt.subplot(2,2,2)
plt.plot(np.arange(len(conv1)), conv1)
plt.xlabel("x")
plt.ylabel("y")
plt.title("(signal ⊗ k1) ⊗ k2")

plt.subplot(2,2,3)
plt.plot(np.arange(len(conv2)), conv2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("(signal ⊗ k2) ⊗ k1")

plt.subplot(2,2,4)
plt.plot(np.arange(len(conv3)), conv3)
plt.xlabel("x")
plt.ylabel("y")
plt.title("signal ⊗ (k1 ⊗ k2)")

plt.tight_layout()
plt.show()

### h)

def gauss_filter():
    im = cv2.imread('slike/lena.png')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    k = np.expand_dims(simple_gauss(3), axis=1)
    kT = k.T
    k_2d = k@k.T

    im_gauss1 = cv2.filter2D(cv2.filter2D(im, -1, k), -1, kT)
    im_gauss2 = cv2.filter2D(im, -1, k_2d)

    print()
    print("Ali je rezultat ločenega Gaussovega jedra enak:", np.allclose(im_gauss1, im_gauss2, rtol=0.1, atol=0.1))

    plt.subplot(2,2,1)
    plt.imshow(im)
    plt.title("Originalna slika")

    plt.subplot(2,2,2)
    plt.imshow(im_gauss1)
    plt.title("(Slika ⊗ k) ⊗ k.T")

    plt.subplot(2,2,3)
    plt.imshow(im_gauss2)
    plt.title("Slika ⊗ (k ⋅ k.T)")

    plt.tight_layout()
    plt.show()

    im_g = cv2.imread('slike/lena_gauss.png')
    im_g = cv2.cvtColor(im_g, cv2.COLOR_BGR2RGB)
    im_sp = cv2.imread('slike/lena_sp.png')
    im_sp = cv2.cvtColor(im_sp, cv2.COLOR_BGR2RGB)

    k = np.expand_dims(simple_gauss(2), axis=1)
    kT = k.T

    im_g_filt = cv2.filter2D(cv2.filter2D(im_g, -1, k), -1, kT)
    im_sp_filt = cv2.filter2D(cv2.filter2D(im_sp, -1, k), -1, kT)

    plt.subplot(2,2,1)
    plt.imshow(im_g)
    plt.title("Gaussov šum")

    plt.subplot(2,2,2)
    plt.imshow(im_sp)
    plt.title("Sol poper šum")

    plt.subplot(2,2,3)
    plt.imshow(im_g_filt)
    plt.title("Gauss filtriran")

    plt.subplot(2,2,4)
    plt.imshow(im_sp_filt)
    plt.title("Sol poper filtriran")

    plt.tight_layout()
    plt.show()


gauss_filter()
# Gaussov filter bolje filtrira Gaussov šum kot sol-poper šum.

### i)

k1 = [[0,0,0],
      [0,2,0],
      [0,0,0]]
k2 = 1/9 * np.array([[1,1,1],[1,1,1],[1,1,1]])
k = np.subtract(k1, k2)

im = cv2.imread("slike/fox.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

im_edge1 = cv2.filter2D(im, -1, k)
im_edge2 = cv2.filter2D(im_edge1, -1, k)
im_edge3 = cv2.filter2D(im_edge2, -1, k)

plt.subplot(2,2,1)
plt.imshow(im)
plt.title("Originalna slika")

plt.subplot(2,2,2)
plt.imshow(im_edge1)
plt.title("Ostrenje (1. prehod)")

plt.subplot(2,2,3)
plt.imshow(im_edge2)
plt.title("Ostrenje (2. prehod)")

plt.subplot(2,2,4)
plt.imshow(im_edge3)
plt.title("Ostrenje (3. prehod)")

plt.tight_layout()
plt.show()

# Iz slike vzamemo detajle in jih bolj poudarimo
# Gauss odstrani detajle, to odštejemo od originalne slike in dobimo sliko brez detajlov.

### j)

x = np.concatenate((np.zeros(14), np.ones(11), np.zeros(15)))
xc = np.copy(x)
xc[11] = 5
xc[18] = 5

# def median_filter():



