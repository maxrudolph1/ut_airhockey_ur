import torch
import torch.nn as nn
import torch.optim as optim

from models import mlp
from buffer import BCBuffer

class BehaviorCloning():
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, learning_rate, batch_size, num_iter, device):
        self.device = device
        self.model = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation, nn.Identity).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.num_iter = num_iter

    def take_action(self, true_pose, true_speed, true_force, measured_acc, srvpose, estop, image):
        state_vals = torch.tensor([true_pose, true_speed, true_force, measured_acc], dtype=torch.float32).to(self.device)
        action = self.model(state_vals)
        return action.detach().cpu().numpy()
    
    def train(self, measured_values, images):
        # time.time(), tidx, count, true_pose, true_speed, true_force, measured_acc, srvpose, rcv.isProtectiveStopped() = measured_values
        states = measured_values[:, 3:7]
        actions = measured_values[:, 7]
        actions = actions[:, :2]

        bc_buffer = BCBuffer(states.shape[1], actions.shape[1], len(states))
        bc_buffer.store_all(states, actions)

        mean_loss = 0

        for _ in range(self.num_iter):
            batch = bc_buffer.sample_batch(self.batch_size)
            self.optimizer.zero_grad()
            action_pred = self.model(batch['obs'])
            loss = ((action_pred - batch['act']) ** 2).mean()
            mean_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        
        return {'loss': mean_loss / self.num_iter}