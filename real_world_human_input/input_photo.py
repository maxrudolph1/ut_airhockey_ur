import cv2
import numpy as np
import sys
# sys.path.append('../')
# from transforms import RobosuiteTransforms, compute_affine_transform

import os

# from check_extrinsics import convert_camera_extrinsic

# Function to find the red hockey puck
def find_red_hockey_paddle(image):
    # Load the image
    # image = cv2.imread(image_path)

    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:,:540] = 0
    hsv_image[400:,:] = 0

    # Define the range of red color in HSV
    # These values might need adjustment depending on the image
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create a mask for red color
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask = mask1 + mask2



    # Blob detection parameters
#     params = cv2.SimpleBlobDetector_Params()
#     params.filterByColor = True
#     params.blobColor = 255  # Since the mask will be white where the puck is
#     params.minDistBetweenBlobs = 100


#    # Filter by Area.
#     params.filterByArea = False
#     params.minArea = 100  # Adjust based on the size of the puck in the image

#     # Filter by Circularity
#     params.filterByCircularity = False
#     params.minCircularity = 0.7  # Adjust to better match the puck's shape

#     # Filter by Convexity
#     params.filterByConvexity = False
#     params.minConvexity = 0.8

#     # Filter by Inertia
#     params.filterByInertia = True
#     params.minInertiaRatio = 0.5
    
    # Create a detector with the parameters
    # detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    # keypoints = detector.detect(mask)
    vals = np.where(mask > 0)
    x, y = int(np.round(np.median(vals[0]))),int(np.round(np.median(vals[1])))
    # cv2.imshow('mask',mask)
    # cv2.waitKey(1)

    # # Draw detected blobs as red circles
    # # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    # print(image.shape)
    
    # image_with_keypoints = cv2.drawKeypoints(image, [(x,y)], np.array([]), (0, 0, 255),
    #                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # py = int(keypoints[0].pt[0])
    # px = int(keypoints[0].pt[1])
    # width=100
    image[x-5:x+5, y-5:y+5, :] = 0
    # Save the image with keypoints
    # cv2.imwrite('output.jpg', image_with_keypoints)

    return x,y,image

def find_green_hockey_puck(image):
    # Load the image
    # 

    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # hsv_image[:,:540] = 0

    # Define the range of green color in HSV
    # These values might need adjustment depending on the image
    # lower_green1 = np.array([100, 10, 25])
    # upper_green1 = np.array([120, 50, 50])
    lower_green1 = np.array([100, 10, 25])
    upper_green1 = np.array([120, 255, 255])
    lower_green2 = np.array([170, 120, 70])
    upper_green2 = np.array([180, 255, 255])

    # Create a mask for green color
    mask = cv2.inRange(hsv_image, lower_green1, upper_green1)
    # mask2 = cv2.inRange(hsv_image, lower_green2, upper_green2)
    # mask = mask1 + mask2



    # Detect blobs
    # keypoints = detector.detect(mask)
    while True:
        cv2.imshow('mask',hsv_image)
        cv2.imshow('mask',mask)
        cv2.waitKey(1)
    vals = np.where(mask > 0)
    x, y = int(np.round(np.median(vals[0]))),int(np.round(np.median(vals[1])))

    image[x-5:x+5, y-5:y+5, :] = 0
    # Save the image with keypoints
    # cv2.imwrite('output.jpg', image_with_keypoints)

    return x,y,image


# def convert_to_world_coords(img_coord):
#     extrinsics = convert_camera_extrinsic([180+14, -21, 0], [0.2286, 0.2286, 1.7907])
#     extrinsics = np.concatenate([extrinsics, np.array([[0,0,0,1]])], axis=0)
#     intrinsics = np.load('../camera_matrix.npy')

#     transforms = RobosuiteTransforms(extrinsics, intrinsics)

#     robot_numbers = np.array([[-0.54, -0.39], [-0.8, -.38], [-.78,0.39], [-.45, .39], [-.43,0]])
#     camera_numbers = np.array([[0.029, 0.09], [-0.296, 0.044], [-0.285, -0.421], [0.0224, -0.418], [0.039, -0.152]])

#     A, t = compute_affine_transform(camera_numbers, robot_numbers)
#     world_coord = transforms.pixel_to_world_coord(np.array(img_coord), solve_for_z=False)
#     world_coord = (A @ world_coord[:2] + t)  * np.array([1,1.512])
#     return world_coord

# Replace 'your_image_path.jpg' with the path to your image
# image = cv2.imread(os.path.join("data", 'puck_test.jpg'))
# print(find_green_hockey_puck(image))
# px,py = find_red_hockey_puck(os.path.join("real_world_human_input", 'test_input.JPG')) 
# px,py = find_red_hockey_puck(os.path.join("real_world_human_input", 'test_input.JPG')) 
# world_coord = convert_to_world_coords((px,py,1))
# print(world_coord)
