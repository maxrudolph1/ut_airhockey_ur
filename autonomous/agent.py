from networks.network_utils import ObjDict, pytorch_model
from autonomous.models import mlp
from networks.general.mlp import MLPNetwork
from robot_operation.coordinate_transform import clip_limits
import numpy as np
import torch.nn as nn
from autonomous.autonomous import AutonomousModel
from typing import Tuple

class ClipTable(object):
    '''
    Takes numpy image [H, W, C] and returns clipped image [H, W, C] with the same shape
    '''
    def __call__(self, X):
        X = np.copy(X)
        X[170:, :, :] = 0
        return X
    
    def __repr__(self):
        return self.__class__.__name__ + '()'


class Agent(AutonomousModel):
    def __init__(self, img_size, puck_history_len, device, target_config = 'train_ppo.yaml', puck_detector=None):
        super().__init__(target_config)
        
        # a randomly initialized neural network that takes in all the paddle components and estop and computes an action
        # optional puck detection
        # self.args = ObjDict({
        #     "num_inputs": 1 + 6 + 6 + 6 + 3 + 3 * 3, # estop + pose + speed + force + acc + last 3 puck pos (with missing indicator)
        #     "num_outputs": 2,
        #     "use_layer_norm": False,
        #     "hidden_sizes": hidden_sizes,
        #     "gpu": 0,
        #     "scale_final": 1,
        #     "activation": 'relu',
        #     "activation_final": 'tanh',
        #     "use_bias": False,
        #     "dropout": 0,
        #     "init_form": 'knorm'
        # })
        # self.network = MLPNetwork(args)
        self.obs_dim = 1 + 6 + 6 + 3 * puck_history_len + 2 + 2 # estop + pose + speed + last $puck_history_len puck pos (with missing indicator) + puck_vel + paddle_puck_rel
        print('obs_dim', self.obs_dim)
        self.act_dim = 2
        self.img_size = img_size
        self.puck_history_len = puck_history_len

        # if args.gpu >= 0: self.network.cuda(device=args.gpu)
        self.device = device
        self.puck_detector = puck_detector
        self.dataset_state_mean = np.load('/datastor1/calebc/public/data/mouse_mimic_mean.npy')
        self.dataset_state_std = np.load('/datastor1/calebc/public/data/mouse_mimic_std.npy')

    def _compute_state(self, pose, speed, i, puck_history):
        state_info = dict()
        state_info['paddles'] = dict()
        state_info['paddles']['paddle_ego'] = dict()
        state_info['paddles']['paddle_ego']['position'] = pose[:2]
        state_info['paddles']['paddle_ego']['velocity'] = speed[:2]
        state_info["pucks"] = list()
        state_info["pucks"].append({"position": np.array(puck_history[i])[:2], "velocity": np.array(puck_history[i])[:2] - np.array(puck_history[i-1])[:2], "occluded": np.array(puck_history[i])[-1]})
        return state_info

    def take_action(self, pose, speed, force, acc, estop, image, images, puck_history, lims, move_lims):
        if self.puck_detector is not None: puck = self.puck_detector(image, puck_history)
        else: puck = (puck_history[-1][0],puck_history[-1][1],0)
        puck_vals = np.concatenate( [np.array(puck_history[-i]) for i in range(1,self.puck_history_len)] + [np.array(puck)])
        puck_vel = np.array(puck)[:2] - np.array(puck_history[-1])[:2]
        paddle_puck_rel = np.array(pose[:2]) - np.array(puck)
        prop_input = np.expand_dims(np.concatenate([np.array([estop]).astype(float), np.array(pose), np.array(speed), np.array(force) / 50, np.array(acc),
                                     puck_vals, puck_vel, paddle_puck_rel]), axis=0)
        prop_input -= self.dataset_state_mean
        prop_input /= self.dataset_state_std
        prop_input = pytorch_model.wrap(prop_input, device=self.device)
        netout = self.policy(prop_input)
        delta_x, delta_y = pytorch_model.unwrap(netout[0])
        move_vector = np.array((delta_x,delta_y)) * np.array(move_lims) / 5
        x, y = move_vector + pose[:2]
        # x, y = clip_limits(delta_vector[0], delta_vector[1],lims)
        print(netout, move_vector, delta_x, delta_y, pose[:2],  x,y)
        return x, y, puck

    def single_agent_step(self, next_state) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.env.current_timestep > 0:
            self.env.old_state = self.env.current_state
        self.env.current_state = next_state

        hit_a_puck = False
        is_finished, truncated, puck_within_home, puck_within_alt_home, puck_within_goal, _ = self.env.has_finished(next_state)
        if not truncated:
            reward = self.env.get_base_reward(next_state, hit_a_puck, puck_within_home, 
                                     puck_within_alt_home, puck_within_goal,
                                     self.env.ego_goal_pos, self.env.ego_goal_radius)
        else:
            reward = self.env.truncate_rew
        reward += self.env.get_reward_shaping(next_state)
        self.env.current_timestep += 1
        
        obs = self.env.get_observation(next_state)
        return obs, reward, is_finished, truncated, {}


    # def train(self, images, poses, speeds, puck_history):
    #     for i, (im, pos, spd) in enumerate(zip(images, poses, speeds)):
    #         next_state = self._compute_state(pos, spd, i, puck_history)
    #         obs, reward, done, truncated, info = self.single_agent_step(next_state)
    #     return {}
    