import cv2
import numpy as np

# Nastavitve
w, h = 1600, 1200
camera_id = 1
output_name = "capture.jpg"

# Naloži kalibracijske podatke
calib = np.load("wide_calibration_data.npz")
M = calib["camera_matrix"]
D = calib["dist_coeffs"]

# Inicializacija kamere
cap = cv2.VideoCapture(camera_id)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

if not cap.isOpened():
    raise RuntimeError("Kamera ni bila uspešno odprta.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Odstrani radialno distorzijo
    undistorted = cv2.undistort(frame, M, D)

    # Rotacija za 180°
    rotated = cv2.rotate(undistorted, cv2.ROTATE_180)

    # Prikaz slike
    cv2.imshow("frame", rotated)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        cv2.imwrite(output_name, rotated)
        print("Slika shranjena.")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
