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
        hue = hsv[..., 0]

        # mask = (40 <= hue) & (hue <= 60)
        # mask = mask.astype(np.uint8) # * 255
        # mask = np.expand_dims(mask, axis=2)
        # mask = np.repeat(mask, 3, axis=2)

        mask = cv2.inRange(hsv, (40, 100, 100), (60, 255, 255))

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        result = frame.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)


        # Prikaži trenutno sliko
        # cv2.imshow('frame', np.vstack((np.hstack((frame, result)), mask)))
        cv2.imshow('frame', np.hstack((frame, result)))

        # Ob pritisku tipke 'q' prekini izvajanje funkcije
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Zapri kamero in okno
    cap.release()
    cv2.destroyAllWindows()

part_c()
