import rtde_control
import time
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
rtde_frequency = 500.0
from pynput import keyboard
import multiprocessing
import pickle
from data_storage import store_data, get_data
import readline

PRESS_VAL = ""

import sys
import select
import tty
import termios
import inspect
import numpy as np
import os
import imageio

def list_ports():
    """
    Test the ports and returns a tuple with the available ports 
    and the ones that are working.
    """
    is_working = True
    dev_port = 0
    working_ports = []
    available_ports = []
    while is_working:
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                available_ports.append(dev_port)
        dev_port +=1
    return available_ports,working_ports

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
Mimg = np.load('Mimg.npy')

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
    # TODO: Modify control type 

    ctrl.forceMode(task_frame, selection_vector, wrench_down, ctrl_type, limits)

upscale_constant = 3
original_size = np.array([640, 480])
visual_downscale_constant = 2

def save_callback(save_image_check):
    cap = cv2.VideoCapture(1)

    while True:
        start = time.time()
        ret, image = cap.read()
        image = cv2.rotate(image, cv2.ROTATE_180)
        if save_image_check[0] == 1: imageio.imsave("./temp/images/img" + str(time.time()) + ".jpg", image)
        image = cv2.resize(image, (int(640*upscale_constant), int(480*upscale_constant)), 
                    interpolation = cv2.INTER_LINEAR)
        dst = cv2.warpPerspective(image,Mimg,original_size * upscale_constant)
        showdst = cv2.resize(dst, (int(640*upscale_constant / visual_downscale_constant), int(480*upscale_constant / visual_downscale_constant)), 
                    interpolation = cv2.INTER_LINEAR)
        cv2.imshow('showdst',showdst)
        cv2.waitKey(1)


def camera_callback(shared_array, save_image_check):
    cap = cv2.VideoCapture(1)

    while True:
        start = time.time()
        ret, image = cap.read()
        image = cv2.rotate(image, cv2.ROTATE_180)
        if save_image_check[0] == 1: imageio.imsave("./temp/images/img" + str(time.time()) + ".jpg", image)
        # shared_image[:] = image.flatten()
        image = cv2.resize(image, (int(640*upscale_constant), int(480*upscale_constant)), 
                    interpolation = cv2.INTER_LINEAR)
        dst = cv2.warpPerspective(image,Mimg,original_size * upscale_constant)
        showdst = cv2.resize(dst, (int(640*upscale_constant / visual_downscale_constant), int(480*upscale_constant / visual_downscale_constant)), 
                    interpolation = cv2.INTER_LINEAR)

        # dst = cv2.resize(dst, original_size.astype(int).tolist(), 
        #             interpolation = cv2.INTER_LINEAR)
        # cv2.imshow('image',image)
        cv2.imshow('image',showdst)
        cv2.setMouseCallback('image', move_event)
        shared_array[0] = mousepos[0] * visual_downscale_constant
        shared_array[1] = mousepos[1] * visual_downscale_constant
        shared_array[2] = mousepos[2] * visual_downscale_constant
        cv2.waitKey(1)
        # print("showtime", time.time() - start)

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

def get_edge(x,y, w, h):
    # returns relative coordinate bounded by w,h
    if np.abs(x) <= w and np.abs(y) <= h:
        return np.array([x, y])
    s = (y)/(x) # slope
    if -h/2 <= s * w/2 <= h/2:
        if x > 0:
            print("high x", s, np.array([w, s * w]))
            return np.array([w, s * w])
        elif x < 0:
            print("low x", s, np.array([-w, -s * w]))
            return np.array([-w, -s * w])
    elif -w/2 <= h/(2*s) <= w/2:
        if y > 0:
            print("high y", s, np.array([h * s, h]))
            return np.array([h * s, h])
        elif y < 0:
            print("low y", s, np.array([-h * s, -h]))
            return np.array([-h * s, -h])

def smoothen(history_pos, history_vel, relative, limits):
    # smoothens trajectories at the far end by lagging the inputs at the x endpoint
    pass

def compute_poly(x,y,true_pose, lims, move_lims):
    x_min_lim, x_max_lim, y_min, y_max = lims
    rmax_x, rmax_y = move_lims
    relx, rely = (x - true_pose[0]), (-y-true_pose[1])
    rad = lambda x,y: np.sqrt(x ** 2 + y ** 2)
    dist = rad(relx, rely)
    polx, poly = min(dist, rmax_x) * relx / dist + true_pose[0], min(dist, rmax_y) * rely / dist + true_pose[1] # Project to circle
    lastpose = [polx,poly]
    x_min = x_min_lim + 0.1 * np.abs(poly)
    x_max = x_max_lim - 0.05 * np.abs(poly)
    # polx = np.clip(polx, x_min_lim, x_max_lim, ) # Workspace limits
    polx = np.clip(polx, x_min, x_max, ) # Workspace limits
    poly = np.clip(poly, y_min, y_max, )
    return polx, poly

def compute_rect(x,y,true_pose, lims, move_lims):
    relx, rely = (x - true_pose[0]), (-y-true_pose[1])
    x_min_lim, x_max_lim, y_min, y_max = lims
    rmax_x, rmax_y = move_lims
    recx, recy = get_edge(relx, rely, rmax_x, rmax_y)
    recx, recy = recx + true_pose[0], recy + true_pose[1]
    recy = np.clip(recy, y_min, y_max, )
    x_min, x_max = x_min_lim, x_max_lim
    recx = np.clip(recx, x_min, x_max, ) # Workspace limits
    return recx, recy


import shutil
def clear_images():
    folder = './temp/images/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def mimic_control(shared_array):
    cap = cv2.VideoCapture(0)

    Mimg = np.load('Mimg_tele.npy')

    while True:
        start = time.time()
        ret, image = cap.read()
        # image = cv2.rotate(image, cv2.ROTATE_180)
        # shared_image[:] = image.flatten()
        image = cv2.resize(image, (int(640*upscale_constant), int(480*upscale_constant)), 
                    interpolation = cv2.INTER_LINEAR)
        dst = cv2.warpPerspective(image,Mimg,original_size * upscale_constant)
        showdst = cv2.resize(dst, (int(640*upscale_constant / visual_downscale_constant), int(480*upscale_constant / visual_downscale_constant)), 
                    interpolation = cv2.INTER_LINEAR)

        # dst = cv2.resize(dst, original_size.astype(int).tolist(), 
        #             interpolation = cv2.INTER_LINEAR)
        # cv2.imshow('image',image)
        cv2.imshow('image',showdst)
        cv2.setMouseCallback('image', move_event)
        shared_array[0] = mousepos[0] * visual_downscale_constant
        shared_array[1] = mousepos[1] * visual_downscale_constant
        shared_array[2] = mousepos[2] * visual_downscale_constant
        cv2.waitKey(1)
        # print("showtime", time.time() - start)


def main():
    ctrl = RTDEControl("172.22.22.2", rtde_frequency, RTDEControl.FLAG_USE_EXT_UR_CAP)
    rcv = RTDEReceive("172.22.22.2")
    control_mode = 'mouse' # 'mimic'
    control_mode = 'mimic'

    shared_mouse_pos = multiprocessing.Array("f", 3)
    shared_image_check = multiprocessing.Array("f", 1)
    shared_mouse_pos[0] = 0
    shared_mouse_pos[1] = 0
    shared_mouse_pos[2] = 1
    shared_image_check[0] = 0
    protected_mouse_pos = ProtectedArray(shared_mouse_pos)
    protected_img_check = ProtectedArray(shared_image_check)
    if control_mode == 'mouse':
        camera_process = multiprocessing.Process(target=camera_callback, args=(protected_mouse_pos,protected_img_check))
        camera_process.start()
    elif control_mode == 'mimic':
        mimic_process = multiprocessing.Process(target=mimic_control, args=(protected_mouse_pos,))
        mimic_process.start()
        camera_process = multiprocessing.Process(target=save_callback, args=(protected_img_check,))
        camera_process.start()

    clear_images()
    # cap2 = cv2.VideoCapture(1)

    
    print(rcv.getTargetTCPPose())
    # cap = cv2.VideoCapture(0)

    robot_numbers = np.array([[-0.54, -0.39], [-0.8, -.38], [-.78,0.39], [-.45, .39], [-.43,0]])
    camera_numbers = np.array([[0.029, 0.09], [-0.296, 0.044], [-0.285, -0.421], [0.0224, -0.418], [0.039, -0.152]])

    A, t = compute_affine_transform(camera_numbers, robot_numbers)

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
    # max workspace limits
    x_min_lim = -0.8
    x_max_lim = -0.35
    y_min = -0.3382
    y_max = 0.388
    try:
        with NonBlockingConsole() as nbc:
            i = 0
            time.sleep(1.0)
            # tstart = int(input("Enter Trajectory Start Number: "))
            list_of_files = filter( lambda x: os.path.isfile 
                    (os.path.join(os.path.join("data","trajectories"), x)), 
                        os.listdir(os.path.join("data","trajectories")) ) 
            list_of_files = list(list_of_files)
            list_of_files.sort()
            if len(list_of_files) == 0:
                tstart = 0
            else:
                tstart = int(list_of_files[-1][len("trajectory_data"):-5]) + 1
            num_trajectories = int(input("Enter number of Trajectories: "))
            pth = input("\nEnter Path (nothing for default): ")
            pth = os.path.join("data", "trajectories/") if pth == "" else pth
            # for j in range(100000):
            #     time.sleep(0.01)  # To prevent high CPU usage
            #     i += 1
            #     if nbc.get_data() == ' ':  # x1b is ESC
            #         break
            

            # Setting a reset pose for the robot
            reset_pose = ([-0.68, 0., 0.33] + angle, vel,acc)
            # reset_pose = ([-0.68, 0., 0.43] + angle, vel,acc)
            for tidx in range(tstart, tstart + num_trajectories):
                apply_negative_z_force(ctrl, rcv)
                print("reset to initial pose:", ctrl.moveL(reset_pose[0], reset_pose[1], reset_pose[2], False))
                count = 0
                # apply_negative_z_force(ctrl, rcv)
                # wait to start moving
                start = ""
                measured_values, frames = list(), list()

                print("Press space to start")
                for j in range(10000):
                    time.sleep(0.01)  # To prevent high CPU usage
                    i += 1
                    if nbc.get_data() == ' ':  # x1b is ESC
                        break
                protected_img_check[0] = 1
                time.sleep(0.7)

                for j in range(2000):
                    start = time.time()
                    time.sleep(block_time)
                    # ret, image = cap.read()
                    # cv2.imshow('image',image)
                    # cv2.setMouseCallback('image', move_event)
                    # cv2.waitKey(1)
                    pixel_coord = np.array([0, 0])
                    pixel_coord[0] = protected_mouse_pos[0]
                    pixel_coord[1] = protected_mouse_pos[1]
                    # pixel_coord[2] = protected_mouse_pos[2]
                    print("Consumer Side Pixel Coord: ", pixel_coord)
                    # relative_coord = transforms.get_relative_coord(pixel_coord)
                    # world_coord = transforms.pixel_to_world_coord(np.array(pixel_coord), solve_for_z=False)
                    # world_coord = (A @ world_coord[:2] + t)  * np.array([1,1.512])
                    # error = np.array(world_coord[:-1]) - obs[-3:]
                    # print(obs[-3:], world_coord[:-1], error)
                    # action = np.append(np.ones(6) * 300, world_coord[:-1], axis=0)
                    # obs, reward, done, info, _ = env.step(action)
                    # ret, image = cap.read()

                    print("To exit press 'q'")
                    val = nbc.get_data()
                    if val == 'q': 
                        
                        break

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
                    

                    # x = np.random.random() * (x_max - x_min) + x_min
                    # y = np.random.random() * (y_max - y_min) + y_min

                    # world_coord[0] = np.clip(world_coord[0], x_min, x_max, )
                    # world_coord[1] = np.clip(world_coord[1], y_min, y_max, )
                    # world_coord[2] += 0.202086849139
                    # print(world_coord)
                    x, y = (pixel_coord - offset_constants) * 0.001
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
                    pose = ([x, -y, computez(x)] + angle, vel,acc)
                    # pose = ([x, -y, 0.43] + angle, vel,acc)
                    # print("movel",true_speed, ctrl.moveL(pose[0], vel, acc, asynchronous=False))

                    ###### servoL ##### BETTER WORK
                    lims = (x_min_lim, x_max_lim, y_min, y_max)
                    move_lims = (rmax_x, rmax_y)
                    polx, poly = compute_poly(x, y, true_pose, lims, move_lims)
                    # relx, rely = (x - true_pose[0]), (-y-true_pose[1])
                    # rad = lambda x,y: np.sqrt(x ** 2 + y ** 2)
                    # dist = rad(relx, rely)
                    # polx, poly = min(dist, rmax_x) * relx / dist + true_pose[0], min(dist, rmax_y) * rely / dist + true_pose[1] # Project to circle
                    # lastpose = [polx,poly]
                    # x_min = x_min_lim + 0.1 * np.abs(poly)
                    # x_max = x_max_lim - 0.05 * np.abs(poly)
                    # # polx = np.clip(polx, x_min_lim, x_max_lim, ) # Workspace limits
                    # polx = np.clip(polx, x_min, x_max, ) # Workspace limits
                    # poly = np.clip(poly, y_min, y_max, )
                    polx, poly = compute_poly(x,y,true_pose, lims, move_lims)

                    # recx, recy = get_edge(relx, rely, rmax_x, rmax_y)
                    # recx, recy = recx + true_pose[0], recy + true_pose[1]
                    # recy = np.clip(recy, y_min, y_max, )
                    # x_min, x_max = x_min_lim, x_max_lim
                    # recx = np.clip(recx, x_min, x_max, ) # Workspace limits
                    # recx, recy = compute_rect(x, y, true_pose, lims, move_lims)
                    
                    z = computez(x)
                    # srvpose = ([recx, recy, 0.30] + angle, vel,acc)
                    srvpose = ([polx, poly, 0.30] + angle, vel,acc)
                    # srvpose = ([polx, poly, z] + angle, vel,acc)
                    # print(pose, dist, relx, rely)

                    values = get_data(time.time(), tidx, count, true_pose, true_speed, true_force, measured_acc, srvpose)
                    measured_values.append(values), #frames.append(np.array(protected_img[:]).reshape(640,480,3))
                    
                    # TODO: change of direction is currently very sudden, we need to tune that
                    # print("servl", srvpose[0][1], true_speed, true_force, measured_acc, ctrl.servoL(srvpose[0], vel, acc, block_time, lookahead, gain))
                    ctrl.servoL(srvpose[0], vel, acc, block_time, lookahead, gain)
                    print("servl", pixel_coord, srvpose[0], rcv.isProtectiveStopped())# , true_speed, true_force, measured_acc, )

                    # print(z, true_force)
                    # measured_values.append([srvpose, true_pose, true_speed, true_force, measured_acc])

                    

                    # apply force control again just in case the arm leaves the table
                    # apply_negative_z_force(ctrl, rcv) 
                    print("time", time.time() - start)

                    # print()
                    count += 1
                    print("COUNTER", count)
                    # if count % 1000 == 0:
                    #     with open(f'measured_values_{count}.pkl', 'wb') as file: 
                            
                    #         # A new file will be created 
                    #         pickle.dump(measured_values, file)
                protected_img_check[0] = 0
                store_data(pth, tidx, count, os.path.join("temp", "images"), measured_values)
                clear_images()

    finally:
        camera_process.kill()
        ctrl.forceModeStop()
        ctrl.stopScript()




if __name__ == "__main__":
    main()
