import numpy as np
import cv2

def part_b():
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

part_b()

def part_c():
    cap = cv2.VideoCapture(0) # 0 = privzeta spletna kamera
    # Namesto številke lahko podamo tudi ime video datoteke

    # Zanka za pridobitev naslednje slike
    while(True):
        # Poskusimo pridobiti trenutno sliko s spletne kamere
        ret, frame = cap.read()

        # Če to ni mogoče (kamera izključena, itd.), končamo z izvajanjem funkcije
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # omeji po h, v in s
        # mask = cv2.inRange(hsv, (90, 50, 50), (150, 255, 255))
        hue = hsv[...,0]
        sat = hsv[...,1]
        val = hsv[...,2]
        mask = (90 <= hue) & (hue <= 150) & (sat >= 50) & (val >= 50)
        mask = mask.astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Prikaži trenutno sliko
        cv2.imshow('frame', np.vstack((mask_bgr, frame)))

        # Ob pritisku tipke 'q' prekini izvajanje funkcije
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Zapri kamero in okno
    cap.release()
    cv2.destroyAllWindows()

part_c()
