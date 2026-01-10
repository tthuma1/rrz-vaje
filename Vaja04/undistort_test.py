import cv2
import numpy as np
import matplotlib.pyplot as plt

im = cv2.imread('moja_slika4.jpg')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

w, h = 1600, 1200
im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

calib = np.load("wide_calibration_data.npz")
M = calib["camera_matrix"]
D = calib["dist_coeffs"]

undistorted = cv2.undistort(im, M, D)
rotated = cv2.rotate(undistorted, cv2.ROTATE_180)

plt.imshow(rotated)
plt.show()