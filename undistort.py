import numpy as np
import glob

import cv2, os
from cv2 import aruco
import copy

camera_matrix = np.load('camera_matrix.npy')
dist_coeffs = np.load('dist_coeffs.npy')
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
img = copy.deepcopy(frame)

cv2.imshow('original img', frame)
cv2.waitKey(0)
cv2.imwrite('og_img.png', frame)
undistorted_image = cv2.undistort(img, camera_matrix, dist_coeffs)

cv2.imshow('undistorted img', undistorted_image)
cv2.waitKey(0)
cv2.imwrite('undistort.png', undistorted_image)