import ur_rtde
import keyboard
import time

def apply_negative_z_force(rtde_control):
    task_frame = [0, 0, 0, 0, 0, 0]
    selection_vector = [0, 0, 1, 0, 0, 0]
    wrench = [0.0, 0.0, -10.0, 0.0, 0.0, 0.0]
    type = 2  # Assuming no transformation of the force frame
    limits = [0.0, 0.0, 0.2, 0.0, 0.0, 0.0]

    rtde_control.forceMode(task_frame, selection_vector, wrench, type, limits)

def main():
    rtde_control = ur_rtde.RTDEControlInterface("127.0.0.1")

    # Parameters
    speed_magnitude = 0.15
    speed_vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    rtde_control.jogStart(speed_vector, ur_rtde.RTDEControlInterface.FEATURE_TOOL)

    print("Use arrow keys to control the robot, to exit press 'q'")

    try:
        while True:
            time.sleep(0.01)  # To prevent high CPU usage
            if keyboard.is_pressed('up'):
                speed_vector = [0.0, 0.0, -speed_magnitude, 0.0, 0.0, 0.0]
            elif keyboard.is_pressed('down'):
                speed_vector = [0.0, 0.0, speed_magnitude, 0.0, 0.0, 0.0]
            elif keyboard.is_pressed('left'):
                speed_vector = [speed_magnitude, 0.0, 0.0, 0.0, 0.0, 0.0]
            elif keyboard.is_pressed('right'):
                speed_vector = [-speed_magnitude, 0.0, 0.0, 0.0, 0.0, 0.0]
            elif keyboard.is_pressed('q'):
                break
            else:
                speed_vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            # apply_negative_z_force(rtde_control)
            rtde_control.jogStart(speed_vector, ur_rtde.RTDEControlInterface.FEATURE_TOOL)

    finally:
        rtde_control.jogStop()
        rtde_control.stopScript()
        # rtde_control.forceModeStop()




if __name__ == "__main__":
    main()
