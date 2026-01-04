import cv2
import matplotlib.pyplot as plt
import numpy as np

ime_slike = 'capture_f1.jpg'
im = cv2.imread(ime_slike)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, arucoParams)

(corners, ids, rejected) = detector.detectMarkers(im)
cv2.aruco.drawDetectedMarkers(im, corners, ids)

plt.imshow(im)
plt.show()