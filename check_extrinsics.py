import numpy as np
import cv2

from scipy.spatial.transform import Rotation as R
import numpy as np

def convert_camera_extrinsic(angles, translation):

    rot = R.from_euler("xyz", angles, degrees=True).as_matrix()
    flipy = np.eye(3)
    flipy[1,1] = -1
    rot = np.matmul(flipy, rot)
    # print(flipy, rot    )
    # print(np.matmul(rot, np.array([[1,0,0]]).T))
    return np.concatenate([rot, np.array([[translation[0]], [translation[1]], [translation[2]]])], axis=-1)

def convert_tcp(tool_position):
    angles = [180+14, -21, 0]
    translation =[0.2286, 0.2286, 1.7907]
    extrinsics = convert_camera_extrinsic(angles, translation)

    # print(extrinsics)
    print (np.matmul(extrinsics, tool_position))
    cap = cv2.VideoCapture(0)
    while True:

        ret, frame = cap.read()
        cv2.imshow('frame',frame)
        cv2.waitKey(2000)

# convert_tcp(np.array([-0.60504, 0.170, 0.31979, 1]))