import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import numpy as np
import h5py

from autonomous.agent import Agent
from autonomous.models import mlp, resnet
from autonomous.buffer import BCBuffer

class BehaviorCloning(Agent):
    def __init__(self, hidden_sizes, device, learning_rate, batch_size, num_iter, img_size=(224, 224), puck_history_len = 5, input_mode='img', target_config='train_ppo.yaml', dataset_path='/datastor1/calebc/public/data', data_mode='mouse'):
        super().__init__(img_size, puck_history_len, device, target_config)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_iter = num_iter
        
        self.buffer = BCBuffer(self.obs_dim, self.act_dim, [*img_size, 3], device, 5000)
        self.dataset_path = dataset_path
        self.data_mode = data_mode

        self.transform_img = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(self.img_size),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.input_mode = input_mode
        if self.input_mode == 'img':
            self.policy = resnet(self.act_dim).to(self.device)
        else:
            self.policy = mlp([self.obs_dim] + [64, 64] + [self.act_dim], activation=nn.ReLU, output_activation=nn.Tanh).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate).to(self.device)

    def populate_buffer(self):
        '''
        Load datasets from hdf5 file and store in buffer
        '''

        data_dir = os.path.join(self.dataset_path, self.data_mode)
        obs = []
        imgs = []
        acts = []
        for file in os.listdir(data_dir):
            
            with h5py.File(os.path.join(data_dir, file), 'r') as f:
                imgs = np.array(f['train_img'])
                measured_vals = np.array(f['train_vals'])

                puck_history = [(-1.5,0,0) for i in range(self.puck_history_len)]

                for img, measured_val in zip(img, measured_vals):
                    obs_from_measured_val = measured_val[3:-6]
                    act = measured_val[-6:-4] - measured_val[4:6] # delta x, delta y
                    puck = self.puck_detector(img, puck_history)
                    puck_history = puck_history[1:] + [puck]
                    obs_from_measured_val = np.concatenate([obs_from_measured_val, puck_history])
                    obs.append(obs_from_measured_val)
                    acts.append(act)
                    transformed_img = self.transform_image(img)
                    imgs.append(transformed_img)

        self.buffer.store_all(obs, acts, imgs)


    def train_offline(self):
        mean_loss = 0

        for _ in range(self.num_iter):
            batch = self.buffer.sample_batch(self.batch_size)
            self.optimizer.zero_grad()
            if self.input_mode == 'img':
                action_pred = self.policy(batch['img'])
            else:
                action_pred = self.policy(batch['obs'])
            loss = ((action_pred - batch['act']) ** 2).mean()
            mean_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        
        return {'loss': mean_loss / self.num_iter}

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