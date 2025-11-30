import numpy as np
import matplotlib.pyplot as plt
import cv2

### c)

# def make_accumulator(path):
#     im = cv2.imread(path)
#     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

#     im_canny = cv2.Canny(im, threshold1=100, threshold2=200)

#     # Resolucija akumulatorskega polja:
#     bins_theta = 300
#     bins_rho = 300

#     max_rho = np.sqrt(im.shape[0]**2 + im.shape[1]**2) # Navadno je to diagonala slike

#     val_theta = np.linspace(-90, 90, bins_theta) / 180 * np.pi # vrednosti theta
#     val_rho = np.linspace(-max_rho, max_rho, bins_rho)
#     A = np.zeros((bins_rho, bins_theta))

#     def fill_accumulator(x, y):
#         rho = x * np.cos(val_theta) + y * np.sin(val_theta) # Izračunamo rho za vse vrednosti theta
#         bin_rho = np.round((rho + max_rho) / (2 * max_rho) * len(val_rho))

#         for i in range(bins_theta):
#             if bin_rho[i] >= 0 and bin_rho[i] <= bins_rho - 1:
#                 A[int(bin_rho[i]), i] += 1

#     xs, ys = np.nonzero(im_canny)
#     for (x, y) in zip(xs, ys):
#         fill_accumulator(x, y)
    
#     return A, im, im_canny

# A, im, im_canny = make_accumulator('slike/oneline.png')
# A2, im2, im_canny2 = make_accumulator('slike/rectangle.png')

# plt.subplot(3,2,1)
# plt.imshow(im)
# plt.title("Originalna slika")

# plt.subplot(3,2,3)
# plt.imshow(im_canny, cmap='gray')
# plt.title("Canny")

# plt.subplot(3,2,5)
# plt.imshow(A)
# plt.title("Akumulatorsko polje")

# plt.subplot(3,2,2)
# plt.imshow(im2)
# plt.title("Originalna slika")

# plt.subplot(3,2,4)
# plt.imshow(im_canny2, cmap='gray')
# plt.title("Canny")

# plt.subplot(3,2,6)
# plt.imshow(A2)
# plt.title("Akumulatorsko polje")

# plt.tight_layout()
# plt.show()

### d)

im = cv2.imread('slike/pier.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
canny = cv2.Canny(gray, threshold1=100, threshold2=200)

lines = cv2.HoughLinesP(
    canny,
    rho=1,
    theta=np.pi/360,
    threshold=140,
    minLineLength=30,
    maxLineGap=50
)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(im, (x1, y1), (x2, y2), (0, 255, 0), 1)

plt.subplot(2,2,1)
plt.imshow(im)
plt.title("pier.jpg")

plt.subplot(2,2,3)
plt.imshow(canny, cmap='gray')
plt.title("Canny")

im = cv2.imread('slike/building.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
canny = cv2.Canny(gray, threshold1=100, threshold2=200)

lines = cv2.HoughLinesP(
    canny,
    rho=1,
    theta=np.pi/180,
    threshold=120,
    minLineLength=10,
    maxLineGap=50
)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(im, (x1, y1), (x2, y2), (0, 255, 0), 1)

plt.subplot(2,2,2)
plt.imshow(im)
plt.title("building.jpg")

plt.subplot(2,2,4)
plt.imshow(canny, cmap='gray')
plt.title("Canny")

plt.tight_layout()
plt.show()