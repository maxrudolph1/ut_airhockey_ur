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

    # Parameters
    speed_magnitude = 0.05
    speed_vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ######  RPY of the Tool Frame when the table is at an angle for reference ####
    # [-0.05153677648744038, -3.0947520618606172, 0.]

    ctrl.jogStart(speed_vector, RTDEControl.FEATURE_TOOL)


    try:
        with NonBlockingConsole() as nbc:
            i = 0
            for j in range(100000):
                time.sleep(0.01)  # To prevent high CPU usage
                i += 1
                if nbc.get_data() == ' ':  # x1b is ESC
                    break
            for i in range(100000):
                time.sleep(0.05)  # To prevent high CPU usage

                print("Use arrow keys to control the robot, to exit press 'q'")
                val = nbc.get_data()
                if val == 'q': break
                if val == 'a':
                    speed_vector = [0.0, -speed_magnitude, 0.0, 0.0, 0.0, 0.0]
                elif val == 'd':
                    speed_vector = [0.0, speed_magnitude, 0.0, 0.0, 0.0, 0.0]
                elif val == 'w':
                    speed_vector = [speed_magnitude, 0.0, 0.0, 0.0, 0.0, 0.0]
                elif val == 's':
                    speed_vector = [-speed_magnitude, 0.0, 0.0, 0.0, 0.0, 0.0]
                elif val == 'q':
                    break
                else:
                    speed_vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                ctrl.jogStart(speed_vector, RTDEControl.FEATURE_TOOL)
                apply_negative_z_force(ctrl, rcv)
                print(rcv.getActualTCPPos())

    finally:
        ctrl.jogStop()
        ctrl.forceModeStop()
        ctrl.stopScript()




if __name__ == "__main__":
    main()
