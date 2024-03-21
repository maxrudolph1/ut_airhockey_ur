import cv2
import imageio
import time
import numpy as np
from real_world_human_input.input_photo import find_red_hockey_paddle


mousepos = (0,0,1)
Mimg = np.load('Mimg.npy')

upscale_constant = 3
original_size = np.array([640, 480])
visual_downscale_constant = 2
save_downscale_constant = 2

def camera_callback(shared_array, save_image_check):
    cap = cv2.VideoCapture(1)

    while True:
        start = time.time()
        ret, image = cap.read()
        image = cv2.rotate(image, cv2.ROTATE_180)
        if save_image_check[0] == 1: imageio.imsave("./temp/images/img" + str(time.time()) + ".jpg", cv2.resize(image, (int(640/save_downscale_constant), int(480/save_downscale_constant))))
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

# callback functions for mimic control
def mimic_control(shared_array):
    cap = cv2.VideoCapture(0)

    Mimg_tele = np.load('Mimg_tele.npy')

    while True:
        start = time.time()
        ret, image = cap.read()
        # image = cv2.rotate(image, cv2.ROTATE_180)
        # shared_image[:] = image.flatten()
        image = cv2.resize(image, (int(640*upscale_constant), int(480*upscale_constant)), 
                    interpolation = cv2.INTER_LINEAR)
        dst = cv2.warpPerspective(image,Mimg_tele,original_size * upscale_constant)
        showdst = cv2.resize(dst, (int(640*upscale_constant / visual_downscale_constant), int(480*upscale_constant / visual_downscale_constant)), 
                    interpolation = cv2.INTER_LINEAR)

        # dst = cv2.resize(dst, original_size.astype(int).tolist(), 
        #             interpolation = cv2.INTER_LINEAR)
        # cv2.imshow('image',image)
        x,y,changed_image = find_red_hockey_paddle(showdst)

        # dst = cv2.resize(dst, original_size.astype(int).tolist(), 
        #             interpolation = cv2.INTER_LINEAR)
        # cv2.imshow('image',image)
        cv2.imshow('image',changed_image)
        shared_array[0] = y * visual_downscale_constant
        shared_array[1] = x * visual_downscale_constant
        cv2.waitKey(1)

def save_callback(save_image_check):
    cap = cv2.VideoCapture(1)

    while True:
        start = time.time()
        ret, image = cap.read()
        image = cv2.rotate(image, cv2.ROTATE_180)
        if save_image_check[0] == 1: imageio.imsave("./temp/images/img" + str(time.time()) + ".jpg", cv2.resize(image, (int(640/save_downscale_constant), int(480/save_downscale_constant))))
        image = cv2.resize(image, (int(640*upscale_constant), int(480*upscale_constant)), 
                    interpolation = cv2.INTER_LINEAR)
        dst = cv2.warpPerspective(image,Mimg,original_size * upscale_constant)
        dst = cv2.rotate(dst, cv2.ROTATE_90_CLOCKWISE)
        showdst = cv2.resize(dst, (int(480*upscale_constant / visual_downscale_constant), int(640*upscale_constant / visual_downscale_constant)), 
                    interpolation = cv2.INTER_LINEAR)
        cv2.imshow('showdst',showdst)
        cv2.waitKey(1)


# performs saving without multiprocessing
def save_collect(cap):
    start = time.time()
    ret, image = cap.read()
    image = cv2.rotate(image, cv2.ROTATE_180)
    save_image = cv2.resize(image, (int(640/save_downscale_constant), int(480/save_downscale_constant)))
    image = cv2.resize(image, (int(640*upscale_constant), int(480*upscale_constant)), 
                interpolation = cv2.INTER_LINEAR)
    dst = cv2.warpPerspective(image,Mimg,original_size * upscale_constant)
    dst = cv2.rotate(dst, cv2.ROTATE_90_CLOCKWISE)
    showdst = cv2.resize(dst, (int(480*upscale_constant / visual_downscale_constant), int(640*upscale_constant / visual_downscale_constant)), 
                interpolation = cv2.INTER_LINEAR)
    cv2.imshow('showdst',showdst)
    cv2.waitKey(1)
    return showdst, save_image

