import os
import h5py
import numpy as np

MIN_MAG = 0.03
MIN_LEN = 5


def produce_reaching(data_mode, dataset_path):
    # takes in a dataset path and data mode and then produces a sequence of goals
    # for the paddle position, where between goals there is a contiguous sequence of actions
    # in the general direction
    for mode in data_mode:
        data_dir = os.path.join(dataset_path, mode)
        data_dir = os.path.join(data_dir, 'cleaned')
        print('Loading data from:', data_dir)
        obs = []
        imgs = []
        acts = []
        for file in os.listdir(data_dir):
            if file.find('trajectory_data') != -1:
                with h5py.File(os.path.join(data_dir, file), 'r') as f:
                    traj_num = int(file[len('trajectory_data'):-5])
                    measured_vals = np.array(f['train_vals'])
                    print("loading", traj_num, measured_vals.shape)
                    start, sv = 0, None
                    cur_spd = None
                    end, ev = 0, None
                    subt_idxes = list()
                    true_posses = list()
                    true_vals = list()
                    tvs = list()
                    for i, val in enumerate(measured_vals):
                        estop, true_pos, true_speed, act = val[3], val[4:6], val[10:12], val[-6:-4]
                        tvs.append(val)
                        true_posses.append(true_pos)
                        if end == start: # starting a new trajectory
                            start, sv = i, true_pos
                            cur_spd = true_speed
                            end += 1
                        else:
                            cur_mag = (np.abs(cur_spd) > MIN_MAG).astype(int) * cur_spd
                            true_mag = (np.abs(true_speed) > MIN_MAG).astype(int) * true_speed
                            if np.sum(np.abs(true_mag)) == 0 or (np.sign(cur_mag[0]) != np.sign(true_mag[0]) or np.sign(cur_mag[1]) != np.sign(true_mag[1])):
                                # if we change direction or have zero velocity signal end of trajectory
                                if end - start > MIN_LEN:# if we have enough points, keep the trajectory
                                    subt_idxes.append((start, end))
                                    print("added full", start, end, true_posses)
                                    goals = np.stack([val.copy() for _ in range(end-start)], axis= 0)
                                    dones = np.zeros(end-start)
                                    dones[-1] = 1
                                    true_vals.append((np.stack(tvs,axis=0), goals, dones))
                                else:
                                    print("skipped", start, end, true_posses)
                                if end - start > MIN_LEN * 2: # if we have a lot of points, keep a sub trajectory fo at least MIN_LEN
                                    length = np.random.randint(MIN_LEN, end-start)
                                    sidx = np.random.randint(start, end - length)
                                    subt_idxes.append((sidx, sidx + length))
                                    goals = np.stack([val.copy() for _ in range(length)], axis= 0)
                                    dones = np.zeros(length)
                                    dones[-1] = 1
                                    true_vals.append((np.stack(tvs[sidx-start:sidx + length - start], axis=0), goals, dones))
                                    print("added sub", sidx, sidx + length, true_posses[sidx - start:sidx+length - start])
                                start = end = i + 1 # start a new trajectory where this one ended
                                true_posses = list()
                                tvs = list() 
                            else:
                                end += 1
                if len(true_vals) == 0: continue
                all_vals = (
                    np.concatenate([tv[0] for tv in true_vals], axis=0),
                    np.concatenate([tv[1] for tv in true_vals], axis=0),
                    np.concatenate([tv[2] for tv in true_vals], axis=0),
                )
                with h5py.File(os.path.join(dataset_path, mode, 'reaching', 'reaching_data' + str(traj_num) + '.hdf5'), 'w') as hf:
                    hf.create_dataset("train_vals",
                                    shape=all_vals[0].shape,
                                    compression="gzip",
                                    compression_opts=9,
                                    data = all_vals[0])
                    hf.create_dataset("goals",
                                    shape=all_vals[1].shape,
                                    compression="gzip",
                                    compression_opts=9,
                                    data = all_vals[1])
                    hf.create_dataset("dones",
                                    shape=all_vals[2].shape,
                                    compression="gzip",
                                    compression_opts=9,
                                    data = all_vals[2])


if __name__ == '__main__':
    produce_reaching(['mouse'], 'data')