import rtde_control
import time
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
rtde_frequency = 500.0
from pynput import keyboard
import multiprocessing
import pickle
from robot_operation.data_storage import store_data, get_data, clear_images
import readline

PRESS_VAL = ""

import sys
import select
import numpy as np
import os
from real_world_human_input.input_photo import find_red_hockey_paddle
import imageio
from robot_operation.multiprocessing import ProtectedArray, NonBlockingConsole
from robot_operation.teleoperation import camera_callback, mimic_control, save_callback, Mimg, save_collect
from robot_operation.robot_control import apply_negative_z_force, MotionPrimitive
from robot_operation.coordinate_transform import compute_pol, compute_rect
from autonomous.initialize import initialize_agent



import cv2

from check_extrinsics import convert_camera_extrinsic

import shutil

def main(control_mode, control_type, load_path = ""):
    '''
        @param control_mode: Where robot actions are generated: teleoperation_modes = mouse, mimic, keyboard, autonomous = BC, RL
        @param control_type: How the robot is controlled (action space), options: rect, pol, prim
    '''
    ctrl = RTDEControl("172.22.22.2", rtde_frequency, RTDEControl.FLAG_USE_EXT_UR_CAP)
    rcv = RTDEReceive("172.22.22.2")

    teleoperation_modes = ['mouse', 'mimic', 'keyboard']
    autonomous_modes = ['BC', 'RL', 'rnet']

    if control_mode in autonomous_modes:
        autonomous_model = initialize_agent(control_mode, load_path)
    # control_mode = 'mouse' # 'mimic'
    # control_mode = 'mimic'

    shared_mouse_pos = multiprocessing.Array("f", 3)
    shared_image_check = multiprocessing.Array("f", 1)
    shared_mouse_pos[0] = 0
    shared_mouse_pos[1] = 0
    shared_mouse_pos[2] = 1
    shared_image_check[0] = 0
    protected_mouse_pos = ProtectedArray(shared_mouse_pos)
    protected_img_check = ProtectedArray(shared_image_check)
    cap = None
    if control_mode == 'mouse':
        camera_process = multiprocessing.Process(target=camera_callback, args=(protected_mouse_pos,protected_img_check))
        camera_process.start()
    elif control_mode == 'mimic':
        mimic_process = multiprocessing.Process(target=mimic_control, args=(protected_mouse_pos,))
        mimic_process.start()
        camera_process = multiprocessing.Process(target=save_callback, args=(protected_img_check,))
        camera_process.start()
    else:
        cap = cv2.VideoCapture(1)
    if control_type == "prim":
        motion_primitive = MotionPrimitive()


    clear_images()
    # cap2 = cv2.VideoCapture(1)

    # cap = cv2.VideoCapture(0)

    # robot magic numbers
    vel = 0.8 # velocity limit
    acc = 0.8 # acceleration limit 

    # rmax_x = 0.23
    # rmax_y = 0.12
    # fast limits
    rmax_x = 0.26
    rmax_y = 0.12

    # safe limits 
    # rmax_x = 0.1
    # rmax_y = 0.05

    # servol control parameters and general frame rate (20Hz)
    block_time = 0.049 # time for the robot to reach a position (blocking)
    runtime = 0
    if control_mode == "mimic":
        compute_time = 0.004
    elif control_mode == "mouse":
        compute_time = 0.002
    elif control_mode == "keyboard":
        compute_time = 0.025
    # compute_time = 0.004 if control_mode == 'mimic' else 0.002 # TODO: figure out the numbers for learned policies
    lookahead = 0.2 # smooths more with larger values (0.03-0.2)
    gain = 700 # 100-2000
    
    # may need to calibrate angle of end effector
    # angle = [-0.05153677648744038, -2.9847520618606172, 0.]
    angle = [-0.00153677648744038, -3.0647520618606172, 0.]

    # if z is used to compute angles
    zslope = 0.02577
    computez = lambda x: zslope * (x + 0.310) - 0.310

    # homography offsets
    offset_constants = np.array((2100, 500))
    
    # max workspace limits
    x_min_lim = -0.8
    x_max_lim = -0.35
    # y_min = -0.3382
    # y_max = 0.388
    y_min = -0.3282
    y_max = 0.378

    # x_min = -1.5
    # x_max = -0.1
    # y_min = -5
    # y_max = 5

    # x_min = -0.8
    # x_max = -0.4
    # y_min = -0.30
    # y_max = 0.30

    # robot reset pose
    reset_pose = ([-0.68, 0., 0.34] + angle, vel,acc)
    lims = (x_min_lim, x_max_lim, y_min, y_max)
    move_lims = (rmax_x, rmax_y)

    try:
        with NonBlockingConsole() as nbc:
            i = 0
            time.sleep(1.0) # gives time for multiprocessing to start

            # gather trajectory storage information
            try:  
                os.mkdir(os.path.join("data", control_mode, "trajectories/"))
                print("made ", os.path.join("data", control_mode, "trajectories/"))
            except OSError as error:
                print(error)
                pass
            list_of_files = filter( lambda x: os.path.isfile 
                    (os.path.join(os.path.join("data", control_mode, "trajectories"), x)), 
                        os.listdir(os.path.join("data", control_mode, "trajectories")) ) 
            list_of_files = list(list_of_files)
            list_of_files.sort(key = lambda x: int(x[len("trajectory_data"):-5]))
            if len(list_of_files) == 0:
                tstart = 0
            else:
                tstart = int(list_of_files[-1][len("trajectory_data"):-5]) + 1
            num_trajectories = int(input("Enter number of Trajectories: "))
            pth = input("\nEnter Path (nothing for default, 0 for no saving): ")
            pth = os.path.join("data", control_mode, "trajectories/") if pth == "" else pth
            if pth == '0': pth = 0

            images, image = list(), None
            tidx = tstart
            while tidx < tstart + num_trajectories:

                # Setting a reset pose for the robot
                apply_negative_z_force(ctrl, rcv)
                reset_success = ctrl.moveL(reset_pose[0], reset_pose[1], reset_pose[2], False)
                print("reset to initial pose:", reset_success)
                count = 0
                time.sleep(0.7)
                # wait to start moving
                print("Press space to start")
                for j in range(10000):
                    time.sleep(0.01)  # To prevent high CPU usage
                    i += 1
                    if nbc.get_data() == ' ':  # x1b is ESC
                        break
                protected_img_check[0] = 1 and bool(pth)
                time.sleep(0.1)

                reset_success = ctrl.moveL(reset_pose[0], reset_pose[1], reset_pose[2], False)
                print("reset to initial pose:", reset_success)
                count = 0
                time.sleep(0.7)

                print("To exit press 'q'")
                measured_values, frames = list(), list()
                puck_history = [(-1.5,0,0) for i in range(5)] # pretend that the puck starts at the other end of the table, but is occluded, for 5 frames
                total = time.time()
                for j in range(2000):
                    time.sleep(max(0,block_time - runtime))
                    print(time.time() - total, runtime)
                    total = time.time()
                    start = time.time()
                    # ret, image = cap.read()
                    # cv2.imshow('image',image)
                    # cv2.setMouseCallback('image', move_event)
                    # cv2.waitKey(1)
                    pixel_coord = np.array([0, 0])
                    pixel_coord[0] = protected_mouse_pos[0]
                    pixel_coord[1] = protected_mouse_pos[1]
                    # pixel_coord[2] = protected_mouse_pos[2]
                    # print("Consumer Side Pixel Coord: ", pixel_coord)
                    val = nbc.get_data()
                    if val == 'q': 
                        break

                    # force control, need it to keep it on the table
                    apply_negative_z_force(ctrl, rcv)

                    # get image data
                    if cap is not None:
                        image, save_img = save_collect(cap)
                        images.append(save_img)
                    
                    # acquire useful statistics
                    true_pose = rcv.getTargetTCPPose()
                    true_speed = rcv.getTargetTCPSpeed()
                    true_force = rcv.getActualTCPForce()
                    measured_acc = rcv.getActualToolAccelerometer()
                    
                    if control_mode in ["mouse", "mimic"]:
                        x, y = (pixel_coord - offset_constants) * 0.001
                    elif control_mode in ["RL", "BC", 'rnet']:
                        x,y, puck = autonomous_model.take_action(true_pose, true_speed, true_force, measured_acc, rcv.isProtectiveStopped(), image, puck_history, lims, move_lims) # TODO: add image handling
                        puck_history.append(puck)
                    ###### servoL #####
                    if control_type == "pol":
                        polx, poly = compute_pol(x, -y, true_pose, lims, move_lims)
                        srvpose = ([polx, poly, 0.30] + angle, vel,acc)
                    elif control_type == "rect":
                        recx, recy = compute_rect(x, -y, true_pose, lims, move_lims)
                        srvpose = ([recx, recy, 0.30] + angle, vel,acc)
                    elif control_type == "prim":
                        x, y = motion_primitive.compute_primitive(val, true_pose, lims, move_lims)
                        srvpose = ([x, y, 0.30] + angle, vel,acc)

                    values = get_data(time.time(), tidx, count, true_pose, true_speed, true_force, measured_acc, srvpose, rcv.isProtectiveStopped())
                    measured_values.append(values), #frames.append(np.array(protected_img[:]).reshape(640,480,3))
                    
                    # TODO: change of direction is currently very sudden, we need to tune that
                    # print("servl", srvpose[0][1], true_speed, true_force, measured_acc, ctrl.servoL(srvpose[0], vel, acc, block_time, lookahead, gain))
                    
                    # ctrl.servoL(srvpose[0], vel, acc, block_time, lookahead, gain)

                    # print("servl", np.abs(polx - true_pose[0]), np.abs(poly - true_pose[1]), pixel_coord, srvpose[0], rcv.isProtectiveStopped())# , true_speed, true_force, measured_acc, )
                    # print("servl", srvpose[0][:2], x,y, true_pose[:2], rcv.isProtectiveStopped())# , true_speed, true_force, measured_acc, )
                    # print("time", time.time() - start)
                    count += 1
                    runtime = time.time() - start
                protected_img_check[0] = 0
                ctrl.servoStop(6)
                if pth: 
                    store_check = input("\nDo you want to store traj " + str(tidx) + "? (y/n): ")
                    if store_check == 'y': 
                        if store_data(pth, tidx, count, os.path.join("temp", "images"), images, measured_values):
                            tidx += 1
                else:
                    tidx += 1
                if control_mode in "RL": autonomous_model.train(measured_values, images, puck_history=puck_history[5:])
                clear_images()

    finally:
        camera_process.kill()
        ctrl.forceModeStop()
        ctrl.stopScript()




if __name__ == "__main__":
    control_mode = 'rnet' # mouse, mimic, keyboard, RL, BC, rnet
    control_type = 'rect' # rect, pol or prim

    main(control_mode, control_type, "")
