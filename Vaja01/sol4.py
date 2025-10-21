import cv2
import matplotlib.pyplot as plt

# 1. Branje slike
img = cv2.imread('slike/trucks.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. Prikaz RGB in posameznih kanalov
plt.figure(figsize=(10,6))
plt.subplot(2,2,1); plt.imshow(img_rgb); plt.title('RGB slika')
plt.subplot(2,2,2); plt.imshow(img_rgb[:,:,0], cmap='gray'); plt.title('R kanal')
plt.subplot(2,2,3); plt.imshow(img_rgb[:,:,1], cmap='gray'); plt.title('G kanal')
plt.subplot(2,2,4); plt.imshow(img_rgb[:,:,2], cmap='gray'); plt.title('B kanal')
plt.show()

# 3. Pretvorba v HSV
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

# 4. Prikaz HSV komponent
plt.figure(figsize=(10,6))
plt.subplot(2,2,1); plt.imshow(img_hsv[:,:,0], cmap='gray'); plt.title('H kanal')
plt.subplot(2,2,2); plt.imshow(img_hsv[:,:,1], cmap='gray'); plt.title('S kanal')
plt.subplot(2,2,3); plt.imshow(img_hsv[:,:,2], cmap='gray'); plt.title('V kanal')
plt.show()

# Modri kanal
blue = img_rgb[:,:,2]

# Pragovanje
blue_thres = blue > 150

hue = img_hsv[:, :, 0]

# Uporabi logične pogoje
blue_thres_hsv = (hue >= 90) & (hue <= 130)

# Prikaz
plt.figure(figsize=(10,5))
plt.subplot(1,3,1); plt.imshow(img_rgb); plt.title('Original')
plt.subplot(1,3,2); plt.imshow(blue_thres, cmap='gray'); plt.title('Upragovana slika (B>150)')
plt.subplot(1,3,3); plt.imshow(blue_thres_hsv, cmap='gray'); plt.title('Upragovana slika (100<H<130)')
plt.show()
