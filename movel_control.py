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

# ...or, in a non-blocking fashion:

vel = 0.8
acc = 0.8

# testing poses
pose_1 = ([-0.6253904507520187, 0.147548517402645964, 0.31455387563231807, 0.13037291932640277, -3.00043870427807, -0.004805326667537209], vel,acc)
pose_2 = ([-0.6253904507520187, -0.067548517402645964, 0.31455387563231807, 0.13037291932640277, -3.00043870427807, -0.004805326667537209], vel,acc)
pose_3 = ([-0.5253904507520187, 0.047548517402645964, 0.31455387563231807, 0.13037291932640277, -3.00043870427807, -0.004805326667537209], vel,acc)

##### TO CALL MOVEL, PLEASE USE THE SINGLE POSE ARGUMENT FORMAT #####

def apply_negative_z_force(ctrl):
    task_frame = [0, 0, 0, 0, 0, 0]
    selection_vector = [0, 0, 1, 0, 0, 0]
    wrench_down = [0.0, 0.0, -5, 0.0, 0.0, 0.0]
    ctrl_type = 2  # Assuming no transformation of the force frame
    limits = [2.0, 2.0, 1.5, 1.0, 1.0, 1.0]

    ctrl.forceMode(task_frame, selection_vector, wrench_down, ctrl_type, limits)

def main():
    ctrl = RTDEControl("172.22.22.2", rtde_frequency, RTDEControl.FLAG_USE_EXT_UR_CAP)
    rcv = RTDEReceive("172.22.22.2")

    
    print(rcv.getTargetTCPPose())

    try:
        with NonBlockingConsole() as nbc:
            i = 0
            for j in range(100000):
                time.sleep(0.01)  # To prevent high CPU usage
                i += 1
                if nbc.get_data() == ' ':  # x1b is ESC
                    break
            for i in range(60):
                # time.sleep(0.5)  # To prevent high CPU usage

                print("Use arrow keys to control the robot, to exit press 'q'")
                
                # force control, but we dont seem to need it anymore.
                # apply_negative_z_force(ctrl) 

                # randomly sampling from a box in the workspace
                x_min = -0.8
                x_max = -0.60
                y_min = -0.30
                y_max = 0.30
                
                # randomly sampling from a circle.
                # mid_x, mid_y = -0.75, 0.             
                # rad = np.random.random() * 0.1
                # angle = np.random.random() * 2 * np.pi
                # x = np.cos(angle) * rad + mid_x
                # y = np.sin(angle) * rad + mid_y

                x = np.random.random() * (x_max - x_min) + x_min
                y = np.random.random() * (y_max - y_min) + y_min

                pose = ([x, y, 0.31455387563231807, 0.13037291932640277, -3.00043870427807, -0.004805326667537209], vel,acc)

                print("movel", ctrl.moveL(pose[0], pose[1], pose[2], False))
                print(rcv.getTargetTCPPose())


    finally:
        # ctrl.forceModeStop()
        ctrl.stopScript()




if __name__ == "__main__":
    main()
