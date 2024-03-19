import time
rtde_frequency = 500.0
from pynput import keyboard
import multiprocessing
import pickle

PRESS_VAL = ""

import sys
import select
import tty
import termios
import inspect
import numpy as np
from servol_homography import compute_poly, compute_rect

class NonBlockingConsole(object):

    def __enter__(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, type, value, traceback):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


    def get_data(self):
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return False
    
class ProtectedArray:
    def __init__(self, array):
        self.array = array
        self.lock = multiprocessing.Lock()

    def __getitem__(self, index):
        with self.lock:
            return self.array[index]

    def __setitem__(self, index, value):
        with self.lock:
            self.array[index] = value


##### TO CALL MOVEL, PLEASE USE THE SINGLE POSE ARGUMENT FORMAT #####
from transforms import RobosuiteTransforms, compute_affine_transform

import cv2

from check_extrinsics import convert_camera_extrinsic


mousepos = (0,0,1)
def camera_callback(shared_array):
    cap = cv2.VideoCapture(0)

    while True:
        ret, image = cap.read()
        image = cv2.rotate(image, cv2.ROTATE_180)
        cv2.imshow('image',image)
        cv2.setMouseCallback('image', move_event)
        shared_array[0] = mousepos[0]
        shared_array[1] = mousepos[1]
        shared_array[2] = mousepos[2]
        cv2.waitKey(1)

def move_event(event, x, y, flags, params):
    global mousepos
    if event==cv2.EVENT_MOUSEMOVE:
  
        # displaying the coordinates
        # on the Shell
        # print(x, ' ', y)
  
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (x, y)
        mousepos = (x,y,1)



def main():
    # ctrl = RTDEControl("172.22.22.2", rtde_frequency, RTDEControl.FLAG_USE_EXT_UR_CAP)
    # rcv = RTDEReceive("172.22.22.2")

    shared_mouse_pos = multiprocessing.Array("f", 3)
    shared_mouse_pos[0] = 0
    shared_mouse_pos[1] = 0
    shared_mouse_pos[2] = 1
    protected_mouse_pos = ProtectedArray(shared_mouse_pos)
    camera_process = multiprocessing.Process(target=camera_callback, args=(protected_mouse_pos,))
    camera_process.start()
    
    # print(rcv.getTargetTCPPose())
    # cap = cv2.VideoCapture(0)
    # extrinsics = convert_camera_extrinsic([180+14, -21, 0], [0.2286, 0.2286, 1.7907])
    angles = [180-6, -24, 1]
    translation = [-0.0889, 0.00635, 1.84]
    extrinsics = convert_camera_extrinsic(angles, translation)
    extrinsics = np.concatenate([extrinsics, np.array([[0,0,0,1]])], axis=0)
    print(extrinsics.shape)
    intrinsics = np.load('camera_matrix.npy')

    transforms = RobosuiteTransforms(extrinsics, intrinsics)

    # robot_numbers = np.array([[-0.54, -0.39], [-0.8, -.38], [-.78,0.39], [-.45, .39], [-.43,0]])
    # camera_numbers = np.array([[0.029, 0.09], [-0.296, 0.044], [-0.285, -0.421], [0.0224, -0.418], [0.039, -0.152]])

    # A, t = compute_affine_transform(camera_numbers, robot_numbers)
    vel = 0.8 # velocity limit
    acc = 0.8 # acceleration limit 
    xscale = 0.7 # scaling parameters for speed control
    yscale = 0.7
    rmax = 0.2 # polar coordinates maximum radius # TODO: make this an ellipsoid short edge y
    # rmax_x = 0.23
    # rmax_y = 0.12
    # rmax_x = 0.23
    # rmax_y = 0.11
    rmax_x = 0.1
    rmax_y = 0.05
    # rmax_x = 0.1
    # rmax_y = 0.05
    block_time = 0.048 # time for the robot to reach a position (blocking)
    lookahead = 0.2 # smooths more with larger values (0.03-0.2)
    gain = 700 # 100-2000
    # angle = [-0.05153677648744038, -2.9847520618606172, 0.]
    angle = [-0.00153677648744038, -3.0647520618606172, 0.]
    zslope = 0.07277
    zslope = 0.02577
    computez = lambda x: zslope * (x + 0.310) - 0.310
    offset_constants = np.array((2100, 500))
    measured_values = list()
    # max workspace limits
    x_min_lim = -0.8
    x_max_lim = -0.35
    y_min = -0.3382
    y_max = 0.388
    try:
        with NonBlockingConsole() as nbc:
            i = 0
            

            # Setting a reset pose for the robot
            count = 0
            while True:
                start = time.time()
                time.sleep(0.1)
                # ret, image = cap.read()
                # cv2.imshow('image',image)
                # cv2.setMouseCallback('image', move_event)
                # cv2.waitKey(1)
                pixel_coord = [0, 0, 1]
                pixel_coord[0] = protected_mouse_pos[0]
                pixel_coord[1] = protected_mouse_pos[1]
                pixel_coord[2] = protected_mouse_pos[2]
                print("Consumer Side Pixel Coord: ", pixel_coord)
                relative_coord = transforms.get_relative_coord(pixel_coord)
                world_coord = transforms.pixel_to_world_coord(np.array(pixel_coord), solve_for_z=False)
                print('Before projection: ', world_coord)
                # world_coord = (A @ world_coord[:2] + t)  * np.array([1,1.512])
                # print('World coord', world_coord)

                print("To exit press 'q'")
                val = nbc.get_data()
                if val == 'q': break


    finally:
        camera_process.kill()
        # ctrl.forceModeStop()
        # ctrl.stopScript()




if __name__ == "__main__":
    main()
