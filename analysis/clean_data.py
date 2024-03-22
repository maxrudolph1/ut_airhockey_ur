import h5py
import os
import cv2

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


def clean_data(pth, target):

    filenames = list()
    for filename in os.listdir(pth):
        file_path = os.path.join(pth, filename)
        try:
            if os.path.isfile(file_path):
                filenames.append(filename)
        except Exception as e:
            print('Failed to recover %s. Reason: %s' % (file_path, e))

    target_filenames = list()
    for filename in os.listdir(target):
        file_path = os.path.join(pth, filename)
        try:
            if os.path.isfile(file_path):
                target_filenames.append(filename)
        except Exception as e:
            print('Failed to recover %s. Reason: %s' % (file_path, e))
    cleaned_trajectories = [int(fn[len("trajectory_data"):-5]) for fn in target_filenames]

    for fn in filenames:
        tidx = int(fn[len("trajectory_data"):-5])
        if tidx in cleaned_trajectories:
            print("skipping", tidx)
            continue
        dataset_dict = load_hdf5_to_dict(os.path.join(pth, fn))
        idx = 0
        frames = dataset_dict['train_img']
        keep = False
        start, end = 0, 2000
        while True:
            frame = frames[idx]
            # cv2.imshow('frame',cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cv2.imshow('frame',frame)
            key = cv2.waitKey(100)
            print("traj", tidx, "frame", idx, key)
            if key == 81:
                idx = max(idx - 1,0)
            if key == 83:
                idx = min(idx + 1, len(frames) - 1)
            if key == 121:
                keep = True
                break
            if key == 110:
                keep = False
                break
            if key == 101:
                start = idx
            if key == 114:
                end = idx
        print("final statistics", tidx, keep, start, end)
        if keep: write_trajectory(target, tidx, frames[start:end], dataset_dict['train_vals'][start:end])

def visualize_clean_data(target, tidx):
    dataset_dict = load_hdf5_to_dict(os.path.join(target, "trajectory_data" + str(tidx) + ".hdf5"))
    frames = dataset_dict['train_img']
    idx = 0
    while True:
        frame = frames[idx]
        # cv2.imshow('frame',cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cv2.imshow('frame',frame)
        key = cv2.waitKey(100)
        print("traj", tidx, "frame", idx, key)
        if key == 81:
            idx = max(idx - 1,0)
        if key == 83:
            idx = min(idx + 1, len(frames) - 1)
        if key == 121: 
            break



# things to look for: start when the puck is dropped and the human hand is no longer over the white part of the table
    # invisible puck
    # end after the robot finishes the last hit and the puck is coming back down when the human misses the puck
    # the camera might say it is not responding, you don't need to close it
        

# usage: left, right advance and regress the frames respectively
    # n will reject the trajectory
    # y will keep the trajectory
    # e will set the start point
    # r will set the end point

if __name__ == '__main__':
# CHANGE THE FOLLOWING TO YOUR PATH AND TARGET
    pth = "data/mouse/trajectories/"
    target = "data/mouse/cleaned/"
    # clean_data(pth, target)
# UNCOMMENT and COMMENT OUT ABOVE if you just want to visualize some cleaned data
    # I left some examples in data/mouse/cleaned
    # press y to stop visualizing cleaned data
    tidx = 34
    visualize_clean_data(target, tidx)