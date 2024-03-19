import numpy as np
import glob

import cv2, os
from cv2 import aruco
import copy, time


# ENTER YOUR PARAMETERS HERE:
# ARUCO_DICT = cv2.aruco.DICT_4X4_1000 # cv2.aruco.DICT_6X6_1000
ARUCO_DICT = cv2.aruco.DICT_6X6_1000
SQUARES_VERTICALLY = 6
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = .035 # 0.14
MARKER_LENGTH = .025 # 0.10 
LENGTH_PX = 640   # total length of the page in pixels
MARGIN_PX = 20    # size of the margin in pixels
SAVE_NAME = 'ChArUco_Marker.png'
# ------------------------------

def create_and_save_new_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
    img = cv2.aruco.CharucoBoard.generateImage(board, (LENGTH_PX, int(LENGTH_PX*size_ratio)), marginSize=MARGIN_PX)
    cv2.imshow("img", img)
    cv2.waitKey(2000)
    cv2.imwrite("armarker.png", img)

# create_and_save_new_board()


def calibrate_and_save_parameters():
    # Define the aruco dictionary and charuco board
    cap = cv2.VideoCapture(0)
    # for vert in range(2, 8):
    #     for horizon in range(2, 8):
    #         for dict in [cv2.aruco.DICT_4X4_100, cv2.aruco.DICT_4X4_1000, cv2.aruco.DICT_4X4_250, cv2.aruco.DICT_4X4_50,
    #                      cv2.aruco.DICT_5X5_100, cv2.aruco.DICT_5X5_1000, cv2.aruco.DICT_5X5_250, cv2.aruco.DICT_5X5_50,
    #                      cv2.aruco.DICT_6X6_100, cv2.aruco.DICT_6X6_1000, cv2.aruco.DICT_6X6_250, cv2.aruco.DICT_6X6_50,
    #                      cv2.aruco.DICT_7X7_100, cv2.aruco.DICT_7X7_1000, cv2.aruco.DICT_7X7_250, cv2.aruco.DICT_7X7_50]:



    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    # board = cv2.aruco.CharucoBoard((vert, horizon), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    board.setLegacyPattern(True)
    params = cv2.aruco.DetectorParameters()

    # Load PNG images from folder
    all_charuco_corners = []
    all_charuco_ids = []
    frames = list()
    
    for i in range(5):
        ret, frame = cap.read()
        img = copy.deepcopy(frame)
        img = cv2.rotate(img, cv2.ROTATE_180)
        # cv2.imshow("frame", img)
        # cv2.waitKey(2000)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)board.setLegacyPattern(True)
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(img, dictionary, parameters=params)

        # If at least one marker is detected
        if len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(img, marker_corners, marker_ids)
            cv2.imshow('frame',img)
            cv2.imwrite('detected_corners.png', img)
            cv2.waitKey(2000)
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, frame, board)
            # print(charuco_retval, charuco_corners, charuco_ids)
            if charuco_retval:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
        frames.append(img)

    # Calibrate camera
    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board, frame.shape[:2], None, None)
    print("rot, trans", rvecs, tvecs)

    # Save calibration data
    np.save('camera_matrix.npy', camera_matrix)
    np.save('dist_coeffs.npy', dist_coeffs)

    # # Iterate through displaying all the images

    # cv2.destroyAllWindows()

def run_intrinsics():
    cap = cv2.VideoCapture(0)
    camera_matrix, dist_coeffs = np.load('camera_matrix.npy'), np.load('dist_coeffs.npy')
    while True:
        ret, image = cap.read()
        undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
        # cv2.imshow('Undistorted Image', undistorted_image)
        cv2.imshow('normal Image', image)
        cv2.waitKey(100)


if __name__ == "__main__":

    calibrate_and_save_parameters()
    run_intrinsics()
    # angles = [-14, 180+21, 0]
    # translation =[0.2286, 0.2286, 1.7907]
    # extrinsics = convert_camera_extrinsic(angles, translation)
    # np.save('camera_extrinsics.npy', extrinsics)


# def calibrate_camera(allCorners,allIds,imsize):
#     """
#     Calibrates the camera using the dected corners.
#     """
#     print("CAMERA CALIBRATION")
#     aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
#     board = aruco.CharucoBoard_create(7, 5, 1, .8, aruco_dict)
#     cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
#                                  [    0., 1000., imsize[1]/2.],
#                                  [    0.,    0.,           1.]])

#     distCoeffsInit = np.zeros((5,1))
#     flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
#     #flags = (cv2.CALIB_RATIONAL_MODEL)
#     (ret, camera_matrix, distortion_coefficients0,
#      rotation_vectors, translation_vectors,
#      stdDeviationsIntrinsics, stdDeviationsExtrinsics,
#      perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
#                       charucoCorners=allCorners,
#                       charucoIds=allIds,
#                       board=board,
#                       imageSize=imsize,
#                       cameraMatrix=cameraMatrixInit,
#                       distCoeffs=distCoeffsInit,
#                       flags=flags,
#                       criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

#     return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors

# while(True):
#     ret, frame = cap.read()

#     cv2.imshow('frame',frame)
#     cv2.waitKey(1)
#     print(cv2.__version__)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Find the chess board corners
#     detector = cv2.aruco.ArucoDetector()
#     detector.setDictionary(cv2.aruco.Dictonary(cv2.aruco.DICT_5X5_50))
#     print(detector.detectMarkers(frame))

#     arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_50)
#     arucoParams = cv2.aruco.DetectorParameters_create()
#     (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict,
#             parameters=arucoParams)
#     print(corners)
#     # cv2.refineDetectedMarkers()
#     # ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
#     # print(ret)
#     # # If found, add object points, image points (after refining them)
#     # if ret == True:
#     #     objpoints.append(objp)
#     #     corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
#     #     imgpoints.append(corners2)
#     #     # Draw and display the corners
#     #     cv2.drawChessboardCorners(frame, (7,6), corners2, ret)
#     #     cv2.imshow('img', frame)
#     #     cv2.waitKey(500)
# cv2.destroyAllWindows()