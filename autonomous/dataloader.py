# Create a torch dataloader for the BCBuffer class in autonomous/buffer.py

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import os
import h5py
from autonomous.agent import ClipTable

class BCBufferDataset(Dataset):
    def __init__(self, dataset_path, modes, img_size, frame_stack=4, puck_history_len=3, input_mode='img'):
        self.dataset_path = dataset_path
        self.data_modes = modes
        self.img_size = img_size
        self.frame_stack = frame_stack
        self.input_mode = input_mode
        self.puck_history_len = puck_history_len

        self.transform_img = torchvision.transforms.Compose([
            ClipTable(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(self.img_size),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.trajectory_path_list = []
        for mode in self.data_modes:
            data_dir = os.path.join(self.dataset_path, mode)
            data_dir = os.path.join(data_dir, 'cleaned')
            for file in os.listdir(data_dir):
                self.trajectory_path_list.append(os.path.join(data_dir, file))

        # Compute the length of the dataset
        self.dataset_len = 0
        for trajectory_path in self.trajectory_path_list:
            with h5py.File(trajectory_path, 'r') as f:
                if self.input_mode == 'img':
                    self.dataset_len += len(f['train_img']) - self.frame_stack + 1
                else:
                    self.dataset_len += len(f['train_img']) - self.puck_history_len + 1

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        # Sample a trajectory
        trajectory_idx = np.random.choice(len(self.trajectory_path_list))
        trajectory_path = self.trajectory_path_list[trajectory_idx]
        with h5py.File(trajectory_path, 'r') as f:
            # Sample a random starting index
            if self.input_mode == 'img':
                start_idx = np.random.randint(low=self.frame_stack-1, high=len(f['train_img']))
                imgs = f['train_img'][start_idx-self.frame_stack+1:]
                measured_val = f['train_vals'][start_idx]
                # obs_from_measured_val = measured_val[3:-6]
                act = measured_val[-6:-4] - measured_val[4:6] # delta x, delta y
                act/=[0.26, 0.12]

                img_stack  = []
                for i in range(self.frame_stack):
                    img_stack.append(self.transform_img(imgs[i]))
                # print(img_stack[0].shape)
                img_stack = torch.cat(img_stack, axis=0)
                # print(img_stack.shape)

                return dict(img=img_stack, act=torch.tensor(act))
            
            else:
                NotImplementedError("Only image input mode is supported")

    def __sample__(self, batch_size):
        # Create a pool of processes calling __getitem__
        p = Pool(4)
        batch = p.map(self.__getitem__, range(batch_size))
        p.close()
        p.join()


def create_dataloader(dataset_path, modes, img_size, frame_stack=4, puck_history_len=3, input_mode='img', batch_size=32, num_workers=64):
    dataset = BCBufferDataset(dataset_path, modes, img_size, frame_stack, puck_history_len, input_mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    return dataloader

