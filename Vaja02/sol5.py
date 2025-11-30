import numpy as np
import matplotlib.pyplot as plt
import cv2

### f)
# Pogosto lahko obravnavamo problem iskanja krožnic, ko imamo radij že poznan.
# Kakšna je v tem primeru enačba, ki jo ena točka generira v parametričnem prostoru?
#     Enačba je enaka `r^2 = (x - x_c)^2 + (y - yc)^2`, kjer so `r`, `x` in `y` podani. V parametričnem prostoru
#     nam možne rešitve `x_c` in `y_c` torej opisujejo krožnico.
#
#     V parametričnem prostoru za posamezno točko krožnice dobimo vse točke, ki se od nje oddaljene za
#     določen radij. Torej za vsako točko na krožnici dobimo krožnico v parametričnem prostoru. Pravo središče
#     krožnice je točka v parametričnem prostoru, skozi katero gre največ krožnic v parametričnem prostoru
#     (točka, ki prejme največ glasov).

### h)

im = cv2.imread("slike/eclipse.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

canny = cv2.Canny(gray, threshold1=50, threshold2=25)

circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=20,
    param1=50,
    param2=20,
    minRadius=45,
    maxRadius=50
)

if circles is not None:
    circles = np.around(circles).astype(np.uint16)
    for x, y, r in circles[0]:
        cv2.circle(im, (x, y), r, (0, 255, 0), 2)

plt.subplot(2,3,1)
plt.imshow(im)
plt.title("eclipse.jpg")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(canny, cmap='gray')
plt.title("Canny")
plt.axis("off")

im = cv2.imread("slike/coins.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

gray_blur = cv2.GaussianBlur(gray,(5,5),0)
canny = cv2.Canny(gray_blur, threshold1=150, threshold2=300)

circles = cv2.HoughCircles(
    gray_blur,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=20,
    param1=300,
    param2=23,
    minRadius=80,
    maxRadius=90
)

if circles is not None:
    circles = np.around(circles).astype(np.uint16)
    for x, y, r in circles[0]:
        cv2.circle(im, (x, y), r, (0, 255, 0), 2)

plt.subplot(2,3,2)
plt.imshow(im)
plt.title("coins.jpg")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(canny, cmap='gray')
plt.title("Canny")
plt.axis("off")


im = cv2.imread("slike/coins_camera.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

gray_blur = cv2.GaussianBlur(gray,(5,5),0)
canny = cv2.Canny(gray_blur, threshold1=120, threshold2=240)

circles = cv2.HoughCircles(
    gray_blur,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=20,
    param1=240,
    param2=30,
    minRadius=30,
    maxRadius=50
)

circles2 = cv2.HoughCircles(
    gray_blur,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=20,
    param1=240,
    param2=30,
    minRadius=190,
    maxRadius=200
)

if circles is not None:
    circles = np.around(circles).astype(np.uint16)
    for x, y, r in circles[0]:
        cv2.circle(im, (x, y), r, (0, 255, 0), 2)

if circles2 is not None:
    circles2 = np.around(circles2).astype(np.uint16)
    for x, y, r in circles2[0]:
        cv2.circle(im, (x, y), r, (0, 255, 0), 2)

plt.subplot(2,3,3)
plt.imshow(im)
plt.title("coins_camera.jpg")
plt.axis("off")

plt.subplot(2,3,6)
plt.imshow(canny, cmap='gray')
plt.title("Canny")
plt.axis("off")

plt.tight_layout()
plt.show()

### i)

cap = cv2.VideoCapture(0)  # 0 = privzeta spletna kamera

# While zanka za posodabljanje pridobljene slike do prekinitve
while(True):
    # Poskusajmo pridobiti trenutno sliko iz spletne kamere
    ret, frame = cap.read()
    
    # Ce to ni mogoce (kamera izkljucena, itd.), koncajmo z izvajanjem funkcije
    if ret == False:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray,(3,3),0)
    canny = cv2.Canny(gray_blur, threshold1=65, threshold2=130)
    canny = np.repeat(np.expand_dims(canny, axis=2), 3, axis=2)

    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=130,
        param2=20,
        minRadius=40,
        maxRadius=50
    )

    if circles is not None:
        circles = np.around(circles).astype(np.uint16)
        for x, y, r in circles[0]:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)    

    stack = np.vstack((frame, canny))
    cv2.imshow('frame', stack)
    
    # Ob pritisku tipke 'q' prekini izvajanje funkcije
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Izklopi kamero in zapri okno
cap.release()
cv2.destroyAllWindows()
