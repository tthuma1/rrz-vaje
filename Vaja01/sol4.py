import cv2
import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np

### a)

R = 255
G = 34
B = 126

C_high = max(R,G,B) # 255
C_low = min(R,G,B) # 34
C_rng = C_high - C_low # 221

S = C_rng / C_high if C_high > 0 else 0 # 0.87
V = C_high / 255 # 1

R_0 = (C_high - R) / C_rng
G_0 = (C_high - G) / C_rng
B_0 = (C_high - B) / C_rng

if R == C_high:
    H_0 = B_0 - G_0
elif G == C_high:
    H_0 = R_0 - B_0 + 2
elif B == C_high:
    H_0 = G_0 - R_0 + 4

H = 1/6 * (H_0 + 6) if H_0 < 0 else 1/6 * H_0

print(f"RGB ({R}, {G}, {B}) = HSV ({H}, {S}, {V})")
print("RGB v HSV z matplotlib funkcijo:", matplotlib.colors.rgb_to_hsv([R/255, G/255, B/255]))
print()

### b)

H = 0.65
S = 0.7
V = 0.15

H_0 = (6.0 * H) % 6
c1 = math.floor(H_0)
c2 = H_0 - c1

x = V * (1.0 - S)
y = (1.0 - (S * c2)) * V
z = (1.0 - S * (1 - c2)) * V

if c1 == 0:
    R_0, G_0, B_0 = (V, z, x)
elif c1 == 1:
    R_0, G_0, B_0 = (y, V, x)
elif c1 == 2:
    R_0, G_0, B_0 = (x, V, z)
elif c1 == 3:
    R_0, G_0, B_0 = (x, y, V)
elif c1 == 4:
    R_0, G_0, B_0 = (z, x, V)
elif c1 == 5:
    R_0, G_0, B_0 = (V, x, y)

N = 255

R = min(round(N * R_0), N)
G = min(round(N * G_0), N)
B = min(round(N * B_0), N)
    
print(f"HSV ({H}, {S}, {V}) = RGB ({R}, {G}, {B})")
print("HSV v RGB z matplotlib funkcijo:", matplotlib.colors.hsv_to_rgb([H, S, V]) * 255)
print()

### c)

img = cv2.imread('slike/trucks.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.subplot(2,2,1)
plt.title('RGB slika')
plt.imshow(img_rgb)

plt.subplot(2,2,2)
plt.imshow(img_rgb[...,0], cmap='gray')
plt.title('R kanal')

plt.subplot(2,2,3)
plt.imshow(img_rgb[...,1], cmap='gray')
plt.title('G kanal')

plt.subplot(2,2,4)
plt.imshow(img_rgb[...,2], cmap='gray')
plt.title('B kanal')

plt.show()

# pretvorba v HSV
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

plt.subplot(2,2,1)
plt.title('RGB slika')
plt.imshow(img_rgb)

plt.subplot(2,2,2)
plt.imshow(img_hsv[...,0] / 180.0, cmap='gray') # COLOR_RGB2HSV naredi H na intervalu [0,180]
plt.title('H kanal')

plt.subplot(2,2,3)
plt.imshow(img_hsv[...,1], cmap='gray')
plt.title('S kanal')

plt.subplot(2,2,4)
plt.imshow(img_hsv[...,2], cmap='gray')
plt.title('V kanal')

plt.show()

### d)
# Modri kanal
blue = img_rgb[:,:,2]

# Pragovanje
blue_thres = blue > 150

hue = img_hsv[:, :, 0]

# Uporabi logične pogoje
blue_thres_hsv = (hue >= 90) & (hue <= 130)

# Prikaz
plt.subplot(1,3,1)
plt.imshow(img_rgb)
plt.title('Original')

plt.subplot(1,3,2)
plt.imshow(blue_thres, cmap='gray')
plt.title('Upragovana slika (B>150)')

plt.subplot(1,3,3)
plt.imshow(blue_thres_hsv, cmap='gray')
plt.title('Upragovana slika (100<H<130)')

plt.show()
