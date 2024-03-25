import imageio
import cv2
import time
import h5py
import os
import shutil
import numpy as np
from rtde_receive import RTDEReceiveInterface as RTDEReceive

# Received Package data types
# https://www.universal-robots.com/articles/ur/interface-communication/real-time-data-exchange-rtde-guide/

# Might be useful for robot proptioception recording
# https://github.com/UniversalRobots/RTDE_Python_Client_Library/blob/main/examples/record.py

def get_data(cur_time, tidx, i, pose, speed, force, acc, desired_pose, estop):
	# rcv = RTDEReceive("172.22.22.2")
	# ret, image = cap.read()
	# timestamp = time.time()
	# print(image.shape)
	# # cv2.imshow('capture',image)
	# # cv2.waitKey(1)
	# image = None

	# print(np.array([tidx]), np.array([i]), pose, speed, force, acc, desired_pose)

	val = np.concatenate([np.array([cur_time]), np.array([tidx]), np.array([i]), np.array([estop]).astype(float), np.array(pose), np.array(speed), np.array(force), np.array(acc), np.array(desired_pose[0])])
	return val#, image

def store_data(pth, tidx, count, image_path, images, vals):

	if len(images) == 0:
		list_of_files = filter( lambda x: os.path.isfile 
							(os.path.join(image_path, x)), 
								os.listdir(image_path) ) 
		list_of_files = list(list_of_files)
		list_of_files.sort()

		vidx = 0
		imgs = list()
		for fil, nextfil in zip(list_of_files, list_of_files[1:]):
			tfil, tnextfil = float(fil[3:-4]), float(nextfil[3:-4])
			tcur = vals[vidx][0]
			if tfil < tcur < tnextfil:
				if np.abs(tfil-tcur) >= np.abs(tnextfil - tcur):
					print(nextfil, tcur)
					imgs.append(imageio.imread(os.path.join(image_path, nextfil)))
				else:
					print(fil, tcur)
					imgs.append(imageio.imread(os.path.join(image_path, fil)))
				vidx += 1
				# cv2.imshow('hsv',imgs[-1])
				# cv2.waitKey(1)
				if vidx == len(vals):
					break
	else:
		imgs = images

	imgs = np.stack(imgs, axis=0)
	
	vals = np.stack(vals, axis=0)

	print(imgs.shape, vals.shape)
	write_trajectory(pth, tidx, imgs, vals)

def write_trajectory(pth, tidx, imgs, vals):
	with h5py.File(os.path.join(pth, 'trajectory_data' + str(tidx) + '.hdf5'), 'w') as hf:
		hf.create_dataset("train_img",
						shape=imgs.shape,
						compression="gzip",
						compression_opts=9,
						data = imgs)
		print(vals)

		hf.create_dataset("train_vals",
						shape=vals.shape,
						compression="gzip",
						compression_opts=9,
						data = vals)
		print(tidx, hf)

def clear_images():
    folder = './temp/images/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))




if __name__ == "__main__":
	store_data()
	
