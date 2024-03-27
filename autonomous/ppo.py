import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

from autonomous.models import ActorCritic
from autonomous.buffer import PPOBuffer
from autonomous.agent import Agent

class PPO(Agent):
    def __init__(self, hidden_sizes, device, learning_rate, clip_ratio, gamma, lam, batch_size, num_trajs, num_ppo_updates, buffer_size, img_size=(224, 224), puck_history_len=5, target_config='train_ppo.yaml'):
        super().__init__(img_size, puck_history_len, device, target_config)
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.lam = lam
        self.batch_size = batch_size
        self.num_trajs = num_trajs
        self.num_ppo_updates = num_ppo_updates
        self.policy = ActorCritic(self.obs_dim, self.act_dim, hidden_sizes, nn.ReLU, device).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.buffer = PPOBuffer(self.obs_dim, self.act_dim, buffer_size, self.device, gamma, lam)
        self.traj_counter = 0

    def train(self, measured_vals, images, puck_history):
        curr_puck_history = [(-1.5,0,0) for i in range(self.puck_history_len)]
        for i, (img, measured_val) in enumerate(zip(images, measured_vals)):
            pose, speed = measured_val[4:10], measured_val[10:16]
            next_state = self._compute_state(pose, speed, i, puck_history)
            _, reward, _, _, _ = self.single_agent_step(next_state)
            curr_puck_history = curr_puck_history[1:] + [puck_history[i]]
            print(np.concatenate(curr_puck_history), measured_val[3:-6])
            obs = np.concatenate([measured_val[3:-6], np.concatenate(curr_puck_history)])
            print(obs.shape)

            with torch.no_grad():
                obs = torch.tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)
                val = self.policy.compute(obs)[1]
                logp = self.policy.log_prob(obs, torch.tensor(measured_val[-6:-4], dtype=torch.float32).to(self.device).unsqueeze(0))

            self.buffer.store(obs[0].cpu().numpy(), copy.deepcopy(measured_val[-6:-4]), copy.deepcopy(reward), val[0].cpu().numpy(), copy.deepcopy(logp.cpu().numpy()))

        # To Check: Might be wrong
        last_val = self.policy.compute(obs)[1].detach().cpu().numpy()
        self.buffer.finish_path(last_val[0])
        self.traj_counter += 1

        if self.traj_counter == self.num_trajs:
            mean_pi_loss = 0
            mean_entropy = 0
            mean_v_loss = 0
            print(self.buffer.ptr, self.batch_size)
            for _ in range(self.num_ppo_updates):
                batch = self.buffer.sample_batch(self.batch_size)
                self.optimizer.zero_grad()
                pi, v = self.policy.compute(batch['obs'])
                logp = self.policy.log_prob(batch['obs'], batch['act'])
                ratio = torch.exp(logp - batch['logp'])
                clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch['adv']
                pi_loss = -(torch.min(ratio * batch['adv'], clip_adv)).mean()
                v_loss = ((v - batch['ret']) ** 2).mean()
                loss = pi_loss + v_loss
                entropy = self.policy.entropy(batch['obs']).mean()
                mean_pi_loss += pi_loss.item()
                mean_entropy += entropy.item()
                mean_v_loss += v_loss.item()
                loss.backward()
                self.optimizer.step()
        
            self.traj_counter = 0
            self.buffer.reset()
            return {'pi_loss': mean_pi_loss / self.num_ppo_updates, 'entropy': mean_entropy / self.num_ppo_updates, 'v_loss': mean_v_loss / self.num_ppo_updates}

        return None

            
            # if self.puck_detector is not None: puck = self.puck_detector(im, puck_history)
            # else: puck = (puck_history[-1][0],puck_history[-1][1],0)
            
            # puck_history = np.concatenate((np.array(puck), puck_history[-2:]), axis=0)

# class PPO:
#     def __init__(self, obs_dim, act_dim, hidden_sizes, activation, device, learning_rate, clip_ratio, gamma, lam, batch_size, num_ppo_updates, buffer_size):
#         self.device = device
#         self.ac = ActorCritic(obs_dim, act_dim, hidden_sizes, activation, device).to(device)
#         self.optimizer = optim.Adam(self.ac.parameters(), lr=learning_rate)
#         self.buffer = PPOBuffer(obs_dim, act_dim, buffer_size, gamma, lam)
#         self.clip_ratio = clip_ratio
#         self.gamma = gamma
#         self.lam = lam
#         self.batch_size = batch_size
#         self.num_ppo_updates = num_ppo_updates
        

#     def take_action(self, true_pose, true_speed, true_force, measured_acc, srvpose, estop, image):
#         state_vals = torch.tensor([true_pose, true_speed, true_force, measured_acc], dtype=torch.float32).to(self.device)
#         action = self.ac.act(state_vals)
#         return action
    
#     def add_transition(self, data):
#         self.buffer.store(*data)
    
#     def train(self):
#         mean_pi_loss = 0
#         mean_entropy = 0
#         mean_v_loss = 0

#         for _ in range(self.num_ppo_updates):
#             batch = self.buffer.sample_batch(self.batch_size)
#             self.optimizer.zero_grad()
#             pi, v = self.ac(batch['obs'])
#             logp = self.ac.log_prob(batch['obs'], batch['act'])
#             ratio = torch.exp(logp - batch['logp'])
#             clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch['adv']
#             pi_loss = -(torch.min(ratio * batch['adv'], clip_adv)).mean()
#             v_loss = ((v - batch['ret']) ** 2).mean()
#             loss = pi_loss + v_loss
#             entropy = self.ac.entropy(batch['obs']).mean()
#             mean_pi_loss += pi_loss.item()
#             mean_entropy += entropy.item()
#             mean_v_loss += v_loss.item()
#             loss.backward()
#             self.optimizer.step()
        
#         return {'pi_loss': mean_pi_loss / self.num_ppo_updates, 'entropy': mean_entropy / self.num_ppo_updates, 'v_loss': mean_v_loss / self.num_ppo_updates}