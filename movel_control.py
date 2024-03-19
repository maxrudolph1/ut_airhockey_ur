import rtde_control
import time
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
rtde_frequency = 500.0
from pynput import keyboard

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

vel = 0.8
acc = 0.8

##### TO CALL MOVEL, PLEASE USE THE SINGLE POSE ARGUMENT FORMAT #####

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

def main():
    ctrl = RTDEControl("172.22.22.2", rtde_frequency, RTDEControl.FLAG_USE_EXT_UR_CAP)
    rcv = RTDEReceive("172.22.22.2")

    
    print(rcv.getTargetTCPPose())

    angle = [-0.05153677648744038, -2.9847520618606172, 0.]

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
            print("reset to initial pose:", ctrl.moveL(reset_pose[0], reset_pose[1], reset_pose[2], False))
            for i in range(60):

                print("To exit press 'q'")
                val = nbc.get_data()
                if val == 'q': break

                # force control, need it to keep it on the table
                ### MUST RUN THIS BEFORE MOVEL ###
                apply_negative_z_force(ctrl, rcv) 

                # randomly sampling from a box in the workspace
                x_min = -0.8
                x_max = -0.60
                y_min = -0.30
                y_max = 0.30

                x = np.random.random() * (x_max - x_min) + x_min
                y = np.random.random() * (y_max - y_min) + y_min

                # 0.32 here is a default z value. Force control allows us not to compute the *exact* z value here
                pose = ([x, y, 0.32] + angle, vel,acc)

                print("movel", ctrl.moveL(pose[0], pose[1], pose[2], False))

                # apply force control again just in case the arm leaves the table
                apply_negative_z_force(ctrl, rcv) 

                print(rcv.getTargetTCPPose())


    finally:
        ctrl.forceModeStop()
        ctrl.stopScript()




if __name__ == "__main__":
    main()
