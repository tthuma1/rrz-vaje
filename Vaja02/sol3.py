import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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

H = (H + 90) % 180 # na primeru slike v navodilih so barve zarotirane za 90 stopinj

dir_hsv = np.dstack((H, S, V))
dir_rgb = cv2.cvtColor(dir_hsv, cv2.COLOR_HSV2RGB)

plt.subplot(2, 3, 6)
plt.title("I_dir(HSV)")
plt.imshow(dir_rgb, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()


### b)

I = cv2.imread("slike/museum.jpg", cv2.COLOR_BGR2RGB)
I = np.mean(I, axis=2)

def edges_simple(I):
    th_init = 100
    Ix, Iy, mag, dir = gradient_magnitude(I)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    plt.subplots_adjust(bottom=0.25)

    ax1.set_title("I")
    ax1.imshow(I, cmap="gray")
    ax1.set_axis_off()

    ax2.set_title("Ix")
    ax2.imshow(Ix, cmap="gray")
    ax2.set_axis_off()

    ax3.set_title("Iy")
    ax3.imshow(Iy, cmap="gray")
    ax3.set_axis_off()

    ax4.set_title("I_mag")
    im_mag = ax4.imshow(mag, cmap="gray")
    ax4.set_axis_off()

    def mag_to_rgb(mag, dir):
        mag_norm = mag / mag.max()

        # dir ima interval [-pi, pi], zato ga premaknes na [0, 2pi], potem [0, 360], potem [0, 180]
        H = ((dir + np.pi) * 180 / np.pi / 2).astype(np.uint8)
        S = np.full_like(H, 255, dtype=np.uint8)
        V = (mag_norm * 255).astype(np.uint8)

        H = (H + 90) % 180 # na primeru slike v navodilih so barve zarotirane za 90 stopinj

        dir_hsv = np.dstack((H, S, V))
        dir_rgb = cv2.cvtColor(dir_hsv, cv2.COLOR_HSV2RGB)

        return dir_rgb

    ax5.set_title("I_dir(HSV)")
    im_mag_hsv = ax5.imshow(mag_to_rgb(mag, dir), cmap="gray")
    ax5.set_axis_off()

    ax6.set_title("I_mag binary")
    im_th_bin = ax6.imshow(mag >= th_init, cmap="gray")
    ax6.set_axis_off()

    ax_slider = plt.axes([0.25, 0.1, 0.55, 0.03])
    slider = Slider(ax_slider, 'Threshold', 0, 330, valinit=th_init, valstep=1)

    def update(_):
        th = slider.val
        mag_th = np.copy(mag)
        mag_th[mag_th < th] = 0

        im_mag.set_data(mag_th)
        im_mag_hsv.set_data(mag_to_rgb(mag_th, dir))
        im_th_bin.set_data(mag >= th)

        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show()

edges_simple(I)

### c)
im = cv2.imread('slike/coins.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

im_canny = cv2.Canny(im, threshold1=300, threshold2=450)
# Preučite nabor njegovih vhodnih parametrov
#    Lahko damo noter `edges` parameter, v katerega se bo shranil rezulat Canny operatorja
#    `apertureSize` = veliksot kernela za Sobelov filter
#    `L2gradient` = ali naj uporablja Evklidsko ali Mnahattansko razdaljo

plt.subplot(2,3,1)
plt.imshow(im)
plt.title("Originalna slika")
plt.axis('off')

plt.subplot(2,3,4)
plt.imshow(im_canny, cmap='gray')
plt.title("Canny")
plt.axis('off')

im = cv2.imread('slike/crossword.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

im_canny = cv2.Canny(im, threshold1=200, threshold2=350)

plt.subplot(2,3,2)
plt.imshow(im)
plt.title("Originalna slika")
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(im_canny, cmap='gray')
plt.title("Canny")
plt.axis('off')

im = cv2.imread('slike/cukec.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

im_canny = cv2.Canny(cv2.GaussianBlur(gray, (9,9), 0), threshold1=40, threshold2=165)

plt.subplot(2,3,3)
plt.imshow(im, cmap='gray')
plt.title("Originalna slika")
plt.axis('off')

plt.subplot(2,3,6)
plt.imshow(im_canny, cmap='gray')
plt.title("Blur + Canny")
plt.axis('off')

plt.tight_layout()
plt.show()


### d)

img = cv2.imread("slike/building.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
gray = np.float32(gray)
gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)

dst = cv2.cornerHarris(gray, blockSize=7, ksize=5, k=0.05)
dst_blur = cv2.cornerHarris(gray_blur, blockSize=3, ksize=3, k=0.01)
# Preučite nabor njegovih vhodnih parametrov
#    `blockSize` = Velikost soseske, nad katero se izračuna kovariančna matrika gradientov.
#       velika vrednost => manj vogalov, ker bo gledal večji prostor okoli točke, da potrdi, ali je res vogal.
#    `kszie` = Velikost Sobelovega operatorja za izračun gradientov.
#       velika vrednost => manj vogalov, ker bo več glajenja.
#    `k` = Harrisov prosto nastavljivi parameter občutljivosti.
#       velika vrednost => manj vogalov, ker morajo biti dovolj očitni

local_max = cv2.dilate(dst, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))) # local maximum suppression
mask = (dst == local_max) & (dst > 0.4 * dst.max())
ys, xs = np.nonzero(mask)

local_max_blur = cv2.dilate(dst_blur, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))) # local maximum suppression
mask_blur = (dst_blur == local_max_blur) & (dst_blur > 0.27 * dst_blur.max())
ys_blur, xs_blur = np.nonzero(mask_blur)

plt.subplot(2,2,1)
plt.title('Original')
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(2,2,2)
plt.title('Vogali')
plt.imshow(dst, cmap='gray')
plt.axis('off')

ax = plt.subplot(2,2,3)
plt.title('Original + vogali')
plt.imshow(img)
plt.axis('off')

ax_blur = plt.subplot(2,2,4)
plt.title('Blur + vogali')
plt.imshow(img)
plt.axis('off')

for x, y in zip(xs, ys):
    ax.add_patch(plt.Circle((x, y), 5, color='red', fill=True))

for x, y in zip(xs_blur, ys_blur):
    ax_blur.add_patch(plt.Circle((x, y), 5, color='red', fill=True))

plt.tight_layout()
plt.show()

img = cv2.imread("slike/crossword.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
gray = np.float32(gray)

dst = cv2.cornerHarris(gray, blockSize=9, ksize=3, k=0.05)

local_max = cv2.dilate(dst, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))) # local maximum suppression
mask = (dst == local_max) & (dst > 0.28 * dst.max())
ys, xs = np.nonzero(mask)

plt.subplot(2,2,1)
plt.title('Original')
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(2,2,2)
plt.title('Vogali')
plt.imshow(dst, cmap='gray')
plt.axis('off')

ax = plt.subplot(2,2,3)
plt.title('Original + vogali')
plt.imshow(img)
plt.axis('off')

for x, y in zip(xs, ys):
    ax.add_patch(plt.Circle((x, y), 5, color='red', fill=True))

plt.tight_layout()
plt.show()

cap = cv2.VideoCapture(0)  # 0 = privzeta spletna kamera

# While zanka za posodabljanje pridobljene slike do prekinitve
while(True):
    # Poskusajmo pridobiti trenutno sliko iz spletne kamere
    ret, frame = cap.read()
    
    # Ce to ni mogoce (kamera izkljucena, itd.), koncajmo z izvajanjem funkcije
    if ret == False:
        break

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, blockSize=5, ksize=3, k=0.02)

    local_max = cv2.dilate(dst, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))) # local maximum suppression
    mask = (dst == local_max) & (dst > 0.25 * dst.max())
    ys, xs = np.nonzero(mask)

    for x, y in zip(xs, ys):
        cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

    # Izrisimo rezultat
    cv2.imshow('frame', frame)
    
    # Ob pritisku tipke 'q' prekini izvajanje funkcije
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Izklopi kamero in zapri okno
cap.release()
cv2.destroyAllWindows()

plt.imshow(frame)
