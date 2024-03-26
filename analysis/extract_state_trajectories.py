import cv2
import numpy as np
from matplotlib import pyplot as plt
import h5py
# Load the image
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)


def find_hsv_puck(image, hsv_low=[0,0,0], hsv_high=[255, 255, 255], hsv_alt=None):
    # hsv_alt should e a lit
    h, w, _ = image.shape
    
    # Convert the left half of the image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # We'll lower the saturation and value thresholds to possibly capture a darker green
    refined_lower = np.array(hsv_low)  # Lower saturation and value
    refined_upper = np.array(hsv_high)
    # print(hsv_image[:,:,2])
    # Create a mask for green color in the left half with the refined thresholds
    refined_mask = cv2.inRange(hsv_image, refined_lower, refined_upper)
    remove_table_edges_mask = np.zeros((h,w), dtype=np.uint8)
    remove_table_edges_mask[0:175, 30:290] = 1
    # remove_table_edges_mask[, :] = 1
    # refined_mask *= remove_table_edges_mask
    refined_mask[320:,:] = 0
    puck_idx = np.where(refined_mask)
    refined_result = cv2.bitwise_and(image, image, mask=refined_mask)
    
    return cv2.cvtColor(refined_result,cv2.COLOR_HSV2RGB), puck_idx, np.stack([refined_mask,refined_mask,refined_mask]).transpose(1,2,0)


def load_hdf5_to_dict(datapath):
    """
    Load a hdf5 dataset into a dictionary.

    :param datapath: Path to the hdf5 file.
    :return: Dictionary with the dataset contents.
    """
    data_dict = {}
    
    # Open the hdf5 file
    with h5py.File(datapath, 'r') as hdf:
        # Loop through groups and datasets
        def recursively_save_dict_contents_to_group(h5file, current_dict):
            """
            Recursively traverse the hdf5 file to save all contents to a Python dictionary.
            """
            for key, item in h5file.items():
                if isinstance(item, h5py.Dataset):  # if it's a dataset
                    current_dict[key] = item[()]  # load the dataset into the dictionary
                elif isinstance(item, h5py.Group):  # if it's a group (which can contain other groups or datasets)
                    current_dict[key] = {}
                    recursively_save_dict_contents_to_group(item, current_dict[key])

        # Start the recursive function
        recursively_save_dict_contents_to_group(hdf, data_dict)

    return data_dict

mousepos = (0,0,1)
Mimg = np.load('../Mimg.npy')

upscale_constant = 3
original_size = np.array([640, 480])
visual_downscale_constant = 2
save_downscale_constant = 1
offset_constants = np.array((2100, 500))


def homography_transform(image, get_save=False):
    # image = cv2.rotate(image, cv2.ROTATE_180)
    save_image = None
    if get_save:
        save_image = cv2.resize(image, (int(640/save_downscale_constant), int(480/save_downscale_constant)))
    image = cv2.resize(image, (int(640*upscale_constant), int(480*upscale_constant)), 
                interpolation = cv2.INTER_LINEAR)
    dst = cv2.warpPerspective(image,Mimg,original_size * upscale_constant)
    dst = cv2.rotate(dst, cv2.ROTATE_90_CLOCKWISE)
    showdst = cv2.resize(dst, (int(480*upscale_constant / visual_downscale_constant), int(640*upscale_constant / visual_downscale_constant)), 
                interpolation = cv2.INTER_LINEAR)
    return showdst, save_image


for traj in range(0,500):
    try:
        path = f'/datastor1/calebc/public/data/mouse/trajectories/trajectory_data{traj}.hdf5'
        dataset_dict = load_hdf5_to_dict(path)
        xs,ys = [], []
        for img in dataset_dict['train_img']:
            # train_img = dataset_dict['train_img'][120]
            train_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            dst, img = homography_transform(train_img)
            refined_img, idx,mask = find_hsv_puck(img, [0,100,100], [50,255,255])
            x,y = np.median(idx[0]), np.median(idx[1])
            xs.append(x)
            ys.append(y)
            xy_pixel = np.array([xs,ys])
        np.save(f'/datastor1/calebc/public/data/mouse/trajectories/state_trajectories/state_trajectory_data{traj}.hdf5', xy_pixel)
        print(xs)
    except:
        pass
