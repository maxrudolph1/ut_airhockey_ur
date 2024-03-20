import cv2, time
import numpy as np
import multiprocessing
from servol_homography import NonBlockingConsole, ProtectedArray
from real_world_human_input.input_photo import find_red_hockey_puck

upscale_constant = 3
original_size = np.array([640, 480])
visual_downscale_constant = 2


def mimic_control(shared_array):
    cap = cv2.VideoCapture(0)

    Mimg = np.load('Mimg_tele.npy')

    shared_mouse_pos = multiprocessing.Array("f", 3)
    protected_mouse_pos = ProtectedArray(shared_mouse_pos)

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
        x,y,changed_image = find_red_hockey_puck(showdst)

        # dst = cv2.resize(dst, original_size.astype(int).tolist(), 
        #             interpolation = cv2.INTER_LINEAR)
        # cv2.imshow('image',image)
        cv2.imshow('image',changed_image)
        shared_array[0] = x * visual_downscale_constant
        shared_array[1] = y * visual_downscale_constant
        cv2.waitKey(1)
        # print("showtime", time.time() - start)

if __name__ == '__main__':
    shared_mouse_pos = multiprocessing.Array("f", 3)

    protected_mouse_pos = ProtectedArray(shared_mouse_pos)

    camera_process = multiprocessing.Process(target=mimic_control, args=(protected_mouse_pos,))

    camera_process.start()
    while True:
        pass
    #     print(protected_mouse_pos[0], protected_mouse_pos[1])
