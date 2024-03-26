import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from models import ActorCritic
from buffer import PPOBuffer

class PPO:
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, device, learning_rate, clip_ratio, gamma, lam, batch_size, num_ppo_updates, buffer_size):
        self.device = device
        self.ac = ActorCritic(obs_dim, act_dim, hidden_sizes, activation, device).to(device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=learning_rate)
        self.buffer = PPOBuffer(obs_dim, act_dim, buffer_size, gamma, lam)
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.lam = lam
        self.batch_size = batch_size
        self.num_ppo_updates = num_ppo_updates
        

    def take_action(self, true_pose, true_speed, true_force, measured_acc, srvpose, estop, image):
        state_vals = torch.tensor([true_pose, true_speed, true_force, measured_acc], dtype=torch.float32).to(self.device)
        action = self.ac.act(state_vals)
        return action
    
    def add_transition(self, data):
        self.buffer.store(*data)
    
    def train(self):
        mean_pi_loss = 0
        mean_entropy = 0
        mean_v_loss = 0

        for _ in range(self.num_ppo_updates):
            batch = self.buffer.sample_batch(self.batch_size)
            self.optimizer.zero_grad()
            pi, v = self.ac(batch['obs'])
            logp = self.ac.log_prob(batch['obs'], batch['act'])
            ratio = torch.exp(logp - batch['logp'])
            clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch['adv']
            pi_loss = -(torch.min(ratio * batch['adv'], clip_adv)).mean()
            v_loss = ((v - batch['ret']) ** 2).mean()
            loss = pi_loss + v_loss
            entropy = self.ac.entropy(batch['obs']).mean()
            mean_pi_loss += pi_loss.item()
            mean_entropy += entropy.item()
            mean_v_loss += v_loss.item()
            loss.backward()
            self.optimizer.step()
        
        return {'pi_loss': mean_pi_loss / self.num_ppo_updates, 'entropy': mean_entropy / self.num_ppo_updates, 'v_loss': mean_v_loss / self.num_ppo_updates}