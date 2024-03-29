import rtde_control
import time
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
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

def apply_negative_z_force(ctrl, rcv=None):
    if rcv is None:
        task_frame = [0, 0, 0, 0, 0, 0]
        wrench_down = [0.0, 0.0, -5, 0.0, 0.0, 0.0]
    else:
        task_frame = rcv.getTargetTCPPose()
        wrench_down = [0.0, 0.0, 5, 0.0, 0.0, 0.0]
    selection_vector = [0, 0, 1, 0, 0, 0]
    ctrl_type = 2  # Assuming no transformation of the force frame
    limits = [2.0, 2.0, 1.5, 1.0, 1.0, 1.0]

    ctrl.forceMode(task_frame, selection_vector, wrench_down, ctrl_type, limits)


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
    ctrl = RTDEControl("172.22.22.2", rtde_frequency, RTDEControl.FLAG_USE_EXT_UR_CAP)
    rcv = RTDEReceive("172.22.22.2")

    shared_mouse_pos = multiprocessing.Array("f", 3)
    shared_mouse_pos[0] = 0
    shared_mouse_pos[1] = 0
    shared_mouse_pos[2] = 1
    protected_mouse_pos = ProtectedArray(shared_mouse_pos)
    camera_process = multiprocessing.Process(target=camera_callback, args=(protected_mouse_pos,))
    camera_process.start()
    
    print(rcv.getTargetTCPPose())
    # cap = cv2.VideoCapture(0)
    extrinsics = convert_camera_extrinsic([180+14, -21, 0], [0.2286, 0.2286, 1.7907])
    extrinsics = np.concatenate([extrinsics, np.array([[0,0,0,1]])], axis=0)
    print(extrinsics.shape)
    intrinsics = np.load('camera_matrix.npy')

    transforms = RobosuiteTransforms(extrinsics, intrinsics)

    robot_numbers = np.array([[-0.54, -0.39], [-0.8, -.38], [-.78,0.39], [-.45, .39], [-.43,0]])
    camera_numbers = np.array([[0.029, 0.09], [-0.296, 0.044], [-0.285, -0.421], [0.0224, -0.418], [0.039, -0.152]])

    A, t = compute_affine_transform(camera_numbers, robot_numbers)

    vel = 0.8 # velocity limit
    acc = 0.8 # acceleration limit 
    xscale = 0.7 # scaling parameters for speed control
    yscale = 0.7
    rmax = 0.2 # polar coordinates maximum radius # TODO: make this an ellipsoid short edge y
    rmax_x = 0.23
    rmax_y = 0.12
    block_time = 0.048 # time for the robot to reach a position (blocking)
    lookahead = 0.2 # smooths more with larger values (0.03-0.2)
    gain = 400 # 100-2000
    angle = [-0.05153677648744038, -2.9847520618606172, 0.]
    measured_values = list()
    try:
        with NonBlockingConsole() as nbc:
            i = 0
            for j in range(100000):
                time.sleep(0.01)  # To prevent high CPU usage
                i += 1
                if nbc.get_data() == ' ':  # x1b is ESC
                    break
            

            # Setting a reset pose for the robot
            reset_pose = ([-0.68, 0., 0.33] + angle, vel,acc)
            # reset_pose = ([-0.68, 0., 0.43] + angle, vel,acc)
            print("reset to initial pose:", ctrl.moveL(reset_pose[0], reset_pose[1], reset_pose[2], False))
            count = 0
            # apply_negative_z_force(ctrl, rcv)
            while True:
                start = time.time()
                time.sleep(block_time)
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
                world_coord = (A @ world_coord[:2] + t)  * np.array([1,1.512])
                # error = np.array(world_coord[:-1]) - obs[-3:]
                # print(obs[-3:], world_coord[:-1], error)
                # action = np.append(np.ones(6) * 300, world_coord[:-1], axis=0)
                # obs, reward, done, info, _ = env.step(action)
                # ret, image = cap.read()

                print("To exit press 'q'")
                val = nbc.get_data()
                if val == 'q': break

                # force control, need it to keep it on the table
                ### MUST RUN THIS BEFORE MOVEL ###
                apply_negative_z_force(ctrl, rcv)

                # randomly sampling from a box in the workspace
                # x_min = -1.5
                # x_max = -0.1
                # y_min = -5
                # y_max = 5

                # x_min = -0.8
                # x_max = -0.4
                # y_min = -0.30
                # y_max = 0.30

                x_min = -0.8
                x_max = -0.4
                y_min = -0.338
                y_max = 0.383

                

                # x = np.random.random() * (x_max - x_min) + x_min
                # y = np.random.random() * (y_max - y_min) + y_min

                # world_coord[0] = np.clip(world_coord[0], x_min, x_max, )
                # world_coord[1] = np.clip(world_coord[1], y_min, y_max, )
                # world_coord[2] += 0.202086849139
                # print(world_coord)
                x, y = world_coord
                true_pose = rcv.getTargetTCPPose()
                true_speed = rcv.getTargetTCPSpeed()
                true_force = rcv.getActualTCPForce()
                measured_acc = rcv.getActualToolAccelerometer()
                # print(true_pose)

                ###### speedL ###### DOESN'T WORK
                # spx, spy = (x - true_pose[0]) * xscale, (y - true_pose[1]) * yscale # Speed control
                # spx, spy = np.clip(spx, -0.8, 0.8, ), np.clip(spy, -0.8, 0.8, )
                # speed = [spx, spy, 0.0]
                # print("speedl",speed,true_speed, acc, ctrl.speedL(speed, acc, time=10))

                ###### movel ###### DOES WORK
                # 0.32 here is a default z value. Force control allows us not to compute the *exact* z value here
                pose = ([x, -y, 0.33] + angle, vel,acc)
                # pose = ([x, -y, 0.43] + angle, vel,acc)
                # print("movel",true_speed, ctrl.moveL(pose[0], vel, acc, asynchronous=False))

                ###### servoL ##### BETTER WORK
                relx, rely = (x - true_pose[0]), (-y-true_pose[1])
                rad = lambda x,y: np.sqrt(x ** 2 + y ** 2)
                dist = rad(relx, rely)
                polx, poly = min(dist, rmax_x) * relx / dist + true_pose[0], min(dist, rmax_y) * rely / dist + true_pose[1] # Project to circle
                lastpose = [polx,poly]
                polx = np.clip(polx, x_min, x_max, ) # Workspace limits
                poly = np.clip(poly, y_min, y_max, )
                srvpose = ([polx, poly, 0.33] + angle, vel,acc)
                # print(pose, dist, relx, rely)

                
                # TODO: change of direction is currently very sudden, we need to tune that
                # print("servl", srvpose[0][1], true_speed, true_force, measured_acc, ctrl.servoL(srvpose[0], vel, acc, block_time, lookahead, gain))
                measured_values.append([srvpose, true_pose, true_speed, true_force, measured_acc])
                

                # apply force control again just in case the arm leaves the table
                # apply_negative_z_force(ctrl, rcv) 
                print("time", time.time() - start)

                # print()
                count += 1
                print("COUNTER", count)
                if count % 1000 == 0:
                    with open(f'measured_values_{count}.pkl', 'wb') as file: 
                        
                        # A new file will be created 
                        pickle.dump(measured_values, file) 

    finally:
        camera_process.kill()
        ctrl.forceModeStop()
        ctrl.stopScript()




if __name__ == "__main__":
    main()
