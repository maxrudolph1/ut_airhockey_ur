import imageio
import cv2
import time
import h5py
import os
import numpy as np
from rtde_receive import RTDEReceiveInterface as RTDEReceive

# Received Package data types
# https://www.universal-robots.com/articles/ur/interface-communication/real-time-data-exchange-rtde-guide/

# Might be useful for robot proptioception recording
# https://github.com/UniversalRobots/RTDE_Python_Client_Library/blob/main/examples/record.py

def get_data(tidx, i, pose, speed, force, acc, desired_pose):
	# rcv = RTDEReceive("172.22.22.2")
	# ret, image = cap.read()
	# timestamp = time.time()
	# print(image.shape)
	# # cv2.imshow('capture',image)
	# # cv2.waitKey(1)
	# image = None

	# print(np.array([tidx]), np.array([i]), pose, speed, force, acc, desired_pose)

	val = np.concatenate([np.array([tidx]), np.array([i]), np.array(pose), np.array(speed), np.array(force), np.array(acc), np.array(desired_pose[0])])
	return val#, image

def store_data(pth, tidx, count, imgs, vals):
	hf=h5py.File(os.path.join(pth, 'trajectory_data' + str(tidx) + '.hdf5'), 'w')

	imgs, vals = np.stack(imgs, axis=0), np.stack(vals, axis=0)

	print(imgs.shape, vals.shape)

	hf.create_dataset("train_img",
					shape=imgs.shape,
					compression="gzip",
					compression_opts=9,
					data = imgs)

	hf.create_dataset("train_vals",
					shape=vals.shape,
					compression="gzip",
					compression_opts=9,
					data = vals)



if __name__ == "__main__":
	save_data()
	
