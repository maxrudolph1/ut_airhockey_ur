# import pygame
import sys

import numpy as np
# import robosuite as suite
# from air_hockey_challenge_robosuite.robosuite.controllers import load_controller_config
# from air_hockey_challenge_robosuite.robosuite.wrappers.visualization_wrapper import VisualizationWrapper
# from air_hockey_challenge_robosuite.robosuite.utils.camera_utils import get_camera_extrinsic_matrix, \
#     get_camera_intrinsic_matrix
from transforms import RobosuiteTransforms, compute_affine_transform
# from robosuite.wrappers import GymWrapper

import cv2
from check_extrinsics import convert_camera_extrinsic


def image_to_pygame(image):
    """
    Convert an image to a pygame surface and scale it to fit the window
    """
    # pg_image = image.clip(0, 1)
    # pg_image = (image * 255).astype(np.uint8)
    # pg_image = np.transpose(image, (1, 0, 2))  # Transpose the image array
    pg_image = pygame.surfarray.make_surface(image)
    pg_image = pygame.transform.scale(pg_image, size)  # Scale the image to fit the window
    return pg_image


def update_window(image):
    global pixel_coord
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEMOTION:
            pixel_coord[:2] = event.pos  # update the shared list with the mouse coordinates

    # Draw the image onto the screen
    screen.blit(image_to_pygame(image), (0, 0))

    # Update the window
    pygame.display.flip()


def get_observation_data(obs):
    return cv2.rotate(cv2.resize(np.uint8(obs[:-3].reshape(env.modality_dims['agentview_image'])), size), cv2.ROTATE_90_CLOCKWISE), obs[-3:]

mousepos = (0,0,1)

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
        


if __name__ == '__main__':
    # Initialize Pygame
    # pygame.init()

    # Set the size of the window
    size = (1000, 1000)

    screen_image = np.zeros((size[0], size[1], 3), np.float32)
    pixel_coord = [0, 0, 1]

    # Create a window
    # screen = pygame.display.set_mode(size)

    # config = {'env_name': 'AirHockey',
    #           'robots': ['UR5e'],
    #           'controller_configs':
    #               {'type': 'OSC_POSITION',
    #                'interpolation': None,
    #                "impedance_mode": "variable_kp",
    #                "control_delta": False},
    #           'gripper_types': 'Robotiq85Gripper', }

    # env = suite.make(
    #     **config,
    #     has_renderer=True,
    #     has_offscreen_renderer=True,
    #     render_camera="sideview",
    #     use_camera_obs=True,
    #     use_object_obs=False,
    #     control_freq=40,
    # )
    cap = cv2.VideoCapture(0)
    # if cap.isOpened():
    #     print(f'Camera index available: {0}')
    #     cap.release()
    # while(True):
    #     ret, frame = cap.read()

    #     cv2.imshow('frame',frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # env = VisualizationWrapper(env)
    # env = GymWrapper(env, keys=['agentview_image', 'robot0_eef_pos'])

    # intrinsic_mat = get_camera_intrinsic_matrix(env.sim, 'agentview', *size)
    # f = 700
    # intrinsic_mat = np.array([[f, 0, 500],
    #                           [0, f, 500],
    #                           [0, 0, 1]])
    
    

    # extrinsics = convert_camera_extrinsic([180+14, -21, 0], [0.2286, 0.2286, 1.7907])
    extrinsics = convert_camera_extrinsic([0, 180+30, 0])
    extrinsics = np.concatenate([extrinsics, np.array([[0,0,0,1]])], axis=0)
    print(extrinsics.shape)
    intrinsics = np.load('camera_matrix.npy')

    transforms = RobosuiteTransforms(extrinsics, intrinsics)

    robot_numbers = np.array([[-0.54, -0.39], [-0.8, -.38], [-.78,0.39], [-.45, .39], [-.43,0]])
    camera_numbers = np.array([[0.029, 0.09], [-0.296, 0.044], [-0.285, -0.421], [0.0224, -0.418], [0.039, -0.152]])

    A, t = compute_affine_transform(camera_numbers, robot_numbers)

    # Start the main loop
    counter = 0
    while True:
        # screen_image, _ = get_observation_data(obs)
        # update_window(screen_image)
        ret, image = cap.read()
        cv2.imshow('image',image)
        cv2.setMouseCallback('image', move_event)
        cv2.waitKey(10)
        pixel_coord = mousepos
        relative_coord = transforms.get_relative_coord(pixel_coord)
        world_coord = transforms.pixel_to_world_coord(np.array(pixel_coord), solve_for_z=False)
        world_coord = (A @ world_coord[:2] + t)  * np.array([1,1.512])
        # error = np.array(world_coord[:-1]) - obs[-3:]
        # print(obs[-3:], world_coord[:-1], error)
        # action = np.append(np.ones(6) * 300, world_coord[:-1], axis=0)
        # obs, reward, done, info, _ = env.step(action)
        ret, image = cap.read()
