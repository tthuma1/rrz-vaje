import numpy as np
import cv2

cap = cv2.VideoCapture(0) # 0 = privzeta spletna kamera
# Namesto številke lahko podamo tudi ime video datoteke

# Zanka za pridobitev naslednje slike
while(True):
    # Poskusimo pridobiti trenutno sliko s spletne kamere
    ret, frame = cap.read()

    # Če to ni mogoče (kamera izključena, itd.), končamo z izvajanjem funkcije
    if not ret:
        break

    gray = np.mean(frame, axis=2).astype(np.uint8)
    gray = np.expand_dims(gray, axis=2)
    gray = np.repeat(gray, 3, axis=2)
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    frame_flipped = cv2.flip(frame, 1)
    gray_flipped = cv2.flip(gray, 1)

    stack1 = np.hstack((frame, gray))
    stack2 = np.hstack((frame_flipped, gray_flipped))
    res = np.vstack((stack1, stack2))

    # Prikaži trenutno sliko
    cv2.imshow('frame', res)

    # Ob pritisku tipke 'q' prekini izvajanje funkcije
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zapri kamero in okno
cap.release()
cv2.destroyAllWindows()