import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import numpy as np
import h5py
from networks.network_utils import pytorch_model

from autonomous.agent import Agent, ClipTable
from autonomous.models import mlp, resnet
from autonomous.buffer import BCBuffer
from autonomous.dataloader import BCBufferDataset, create_dataloader
import cv2
from PIL import Image

class BehaviorCloning(Agent):
    def __init__(self, hidden_sizes, device, learning_rate, batch_size, num_iter, frame_stack=4, img_size=(224, 224), puck_history_len = 3, input_mode='img', target_config='train_ppo.yaml', dataset_path='/datastor1/calebc/public/data', data_mode=['mimic', 'mouse'], save_freq=500, save_dir='frame_stack', log_freq=1, puck_detector=None):
        super().__init__(img_size, puck_history_len, device, target_config, puck_detector)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_iter = num_iter
        
        if frame_stack > 1:
            self.buffer = BCBuffer(self.obs_dim, self.act_dim, [frame_stack, *img_size, 3], device, 5000)
        else:
            self.buffer = BCBuffer(self.obs_dim, self.act_dim, [*img_size, 3], device, 5000)
        self.dataset_path = dataset_path
        self.data_mode = data_mode
        self.save_freq = save_freq
        self.save_dir = save_dir
        self.log_freq = log_freq
        self.frame_stack = frame_stack

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.transform_img = torchvision.transforms.Compose([
            ClipTable(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(self.img_size),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.input_mode = input_mode
        if self.input_mode == 'img':
            self.policy = resnet(self.act_dim, num_input_channels=3*self.frame_stack, output_activation=nn.Tanh, pretrained=True).to(self.device)
        else:
            self.policy = mlp([self.obs_dim] + hidden_sizes + [self.act_dim], activation=nn.ReLU, output_activation=nn.Tanh).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

    def populate_buffer(self):
        '''
        Load datasets from hdf5 file and store in buffer
        '''
        for mode in self.data_mode:
            data_dir = os.path.join(self.dataset_path, mode)
            data_dir = os.path.join(data_dir, 'cleaned')
            print('Loading data from:', data_dir)
            obs = []
            imgs = []
            acts = []
            for file in os.listdir(data_dir):
                
                with h5py.File(os.path.join(data_dir, file), 'r') as f:
                    # try:
                    dat_imgs = np.array(f['train_img'])
                    measured_vals = np.array(f['train_vals'])

                    puck_history = [(-1.5,0,0) for i in range(self.puck_history_len)]

                    # if self.frame_stack > 1:
                    #     dat_imgs = np.concatenate([np.zeros([self.frame_stack-1, *dat_imgs.shape[1:]]), dat_imgs], axis=0)
                    # for i in range(self.frame_stack-1, len(dat_imgs)):
                    #     img = dat_imgs[i-self.frame_stack+1:i+1]
                    #     measured_val = measured_vals[i - self.frame_stack + 1]
                    # cv2.imshow('img', new_img)
                    # cv2.waitKey(0) #170
                    frame_stacked = []
                    for i in range(self.frame_stack-1):
                        frame_stacked.append(self.transform_img(dat_imgs[i]))

                    for img, measured_val in zip(dat_imgs[self.frame_stack-1:], measured_vals[self.frame_stack-1:]):
                        obs_from_measured_val = measured_val[3:-6]
                        act = measured_val[-6:-4] - measured_val[4:6] # delta x, delta y
                        act/=[0.26, 0.12]
                        if self.puck_detector is not None:
                            puck = self.puck_detector(img, puck_history)
                            puck_history = puck_history[1:] + [puck]
                        # print(obs_from_measured_val.shape, act.shape, img.shape, np.array(puck_history).shape)
                        obs_from_measured_val = np.concatenate([obs_from_measured_val, *[puck_history[i] for i in range(self.puck_history_len)]])
                        obs.append(obs_from_measured_val)
                        acts.append(act)
                        # print('Image transformation')
                        # print(img)
                        transformed_img = self.transform_img(img)
                        frame_stacked.append(transformed_img)
                        # print(transformed_img)
                        # print('--------------------------------')
                        # print('stacked_img_shape: ', torch.cat(frame_stacked, axis=0).shape)
                        imgs.append(torch.cat(frame_stacked, axis=0))
                        frame_stacked.pop(0)
                    # except Exception as e:
                        # print('Error in file:', file, e)
                        # continue
            
        print('Storing data in buffer')
        print(len(imgs), imgs[0].shape)
        print(np.max(acts), np.min(acts))
        self.buffer.store_all(obs, acts, imgs)

    def load_data(self):
        self.dataloader = create_dataloader(self.dataset_path, self.data_mode, self.img_size, self.frame_stack, self.puck_history_len, self.input_mode, self.batch_size)

    def train_offline(self):
        mean_loss = 0
        running_mean = 0
        iter_num = 0
        for ep in range(self.num_iter):
            for i, batch in enumerate(self.dataloader):
                # batch = self.buffer.sample_batch(4)
                self.optimizer.zero_grad()
                if self.input_mode == 'img':
                    action_pred = self.policy(batch['img'].to(self.device))
                else:
                    action_pred = self.policy(batch['obs'])

                loss = ((action_pred - batch['act'].to(self.device)) ** 2).mean()
                mean_loss += loss.item()
                running_mean += loss.item()
                loss.backward()
                self.optimizer.step()

                if (iter_num+1) % self.log_freq == 0:
                    print('Iteration:', iter_num+1, 'Loss:', running_mean / self.log_freq)
                    running_mean = 0

                if iter_num % self.save_freq == 0:
                    torch.save(self.policy.state_dict(), os.path.join(self.save_dir, 'bc_model_' + str(iter_num) + '.pt'))
                    print('Model saved at iteration:', iter_num)
                iter_num += 1
        torch.save(self.policy.state_dict(), os.path.join(self.save_dir, 'bc_model_final.pt'))
        # print('Model saved at iteration:', i)
        return {'loss': mean_loss / iter_num}
    
    def load_model(self, model_path):
        self.policy.load_state_dict(torch.load(model_path))
        self.policy.eval()
        print('Model loaded from:', model_path)

    def take_action(self, pose, speed, force, acc, estop, image,images, puck_history, lims, move_lims):
        if self.input_mode == 'img': # TODO: images would have to be stakced for frame stacking
            puck = (puck_history[-1][0],puck_history[-1][1],0)
            zero_stack = max(0, self.frame_stack - len(images))
            frame_stack = np.concatenate([np.zeros((images[-1].shape[0], images[-1].shape[1], zero_stack * 3)).astype(np.uint8)] + [images[-i] for i in range(1,self.frame_stack + 1 - zero_stack)], axis =-1)
            
            image = pytorch_model.wrap(self.transform_img(frame_stack).float(), device=self.device).unsqueeze(0)
            # print(self.frame_stack, zero_stack, len(images), self.transform_img(frame_stack))
            netout = self.policy(image)
            delta_x, delta_y = pytorch_model.unwrap(netout[0])
            move_vector = np.array((delta_x,delta_y)) * np.array(move_lims)
            x, y = move_vector + pose[:2]
            # x, y = clip_limits(delta_vector[0], delta_vector[1],lims)
            print(netout, move_vector, delta_x, delta_y, pose[:2],  x,y)
            return x, y, puck
        else:
            super().take_action(pose, speed, force, acc, estop, image, puck_history, lims, move_lims)
        return x, y, puck


# class BehaviorCloning():
#     def __init__(self, obs_dim, act_dim, hidden_sizes, activation, learning_rate, batch_size, num_iter, device):
#         self.device = device
#         self.model = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, nn.Identity).to(device)
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
#         self.batch_size = batch_size
#         self.num_iter = num_iter

#     def take_action(self, true_pose, true_speed, true_force, measured_acc, srvpose, estop, image):
#         state_vals = torch.tensor([true_pose, true_speed, true_force, measured_acc], dtype=torch.float32).to(self.device)
#         action = self.model(state_vals)
#         return action.detach().cpu().numpy()
    
#     def train(self, measured_values, images):
#         # time.time(), tidx, count, true_pose, true_speed, true_force, measured_acc, srvpose, rcv.isProtectiveStopped() = measured_values
#         states = measured_values[:, 3:7]
#         actions = measured_values[:, 7]
#         actions = actions[:, :2]

#         bc_buffer = BCBuffer(states.shape[1], actions.shape[1], len(states))
#         bc_buffer.store_all(states, actions)

#         mean_loss = 0

#         for _ in range(self.num_iter):
#             batch = bc_buffer.sample_batch(self.batch_size)
#             self.optimizer.zero_grad()
#             action_pred = self.model(batch['obs'])
#             loss = ((action_pred - batch['act']) ** 2).mean()
#             mean_loss += loss.item()
#             loss.backward()
#             self.optimizer.step()
        
#         return {'loss': mean_loss / self.num_iter}
    

# if __name__ == '__main__':
#     bc = BehaviorCloning([64, 64], 'cuda', 3e-4, 128, 10000)
#     bc.populate_buffer()
#     # bc.load_model('models/bc_model_900.pt')
#     bc.train_offline()