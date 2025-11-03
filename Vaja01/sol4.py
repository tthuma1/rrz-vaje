import cv2
import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np
from matplotlib.widgets import Slider

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

blue = img_rgb[...,2]
blue_thres = blue > 150

# Prikaz
plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title('Original')

plt.subplot(1,2,2)
plt.imshow(blue_thres, cmap='gray')
plt.title('Upragovana slika (B>150)')

plt.show()

### e)

sum_channels = np.sum(img_rgb, axis=2, keepdims=True)
sum_channels[sum_channels == 0] = 1e-6 # da nimamo deljenja z 0

norm_rgb = img_rgb / sum_channels

blue_thres = norm_rgb[..., 2] > 0.5

# 6. Prikaz rezultatov
plt.subplot(1,2,1)
plt.imshow(norm_rgb)
plt.title('Normaliziran RGB')

plt.subplot(1,2,2)
plt.imshow(blue_thres, cmap='gray')
plt.title('Upragovana normalizirana slika (B>150)')

plt.show()

### f)

def hsv_thresholding(filepath):
    img = cv2.imread(filepath)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hue = img_hsv[..., 0]

    low_init, high_init = 90, 102

    fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.subplots_adjust(bottom=0.25)

    ax1.imshow(img_rgb)
    ax1.set_title('RGB slika')

    img_thresh = (low_init <= hue) & (hue <= high_init)
    im2 = ax2.imshow(img_thresh, cmap='gray')
    ax2.set_title(f'{low_init} <= H <= {high_init}')

    ax_low = plt.axes([0.25, 0.1, 0.55, 0.03])
    ax_high = plt.axes([0.25, 0.05, 0.55, 0.03])
    slider_low = Slider(ax_low, 'Spodnja meja', 0, 179, valinit=low_init, valstep=1)
    slider_high = Slider(ax_high, 'Zgornja meja', 0, 179, valinit=high_init, valstep=1)

    # Update callback
    def update(_):
        low = slider_low.val
        high = slider_high.val
        img_thresh = (hue >= low) & (hue <= high)
        im2.set_data(img_thresh)
        ax2.set_title(f'{low} <= H <= {high}')
        fig.canvas.draw_idle()

    slider_low.on_changed(update)
    slider_high.on_changed(update)

    plt.show()

hsv_thresholding('slike/trucks.jpg')

### g)

hsv_thresholding('slike/color_wheel.jpg')
hsv_thresholding('slike/image.png')

### h)

def im_mask(im, mask):
    return im * mask

im = cv2.imread('slike/trucks.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

mask = np.random.randint(0, 2, size=(im.shape[0], im.shape[1]), dtype=np.uint8)

mask[:200, :100] = 1
mask[20:60, 60:70] = 0
mask[-100:, -200:] = 0

mask = np.expand_dims(mask, axis=2) # dodaj channel dimenzijo
mask = np.repeat(mask, 3, axis=2) # channel dimenzija naj ima 3 channele

plt.subplot(2,2,1)
plt.imshow(im)
plt.title('Originalna slika')

plt.subplot(2,2,2)
plt.imshow(mask * 255)
plt.title('Maska')

plt.subplot(2,2,3)
plt.imshow(im_mask(im, mask))
plt.title('Slika z masko')
plt.show()
