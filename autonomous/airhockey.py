from gymnasium import Env
import numpy as np
from gymnasium.spaces import Box
from gymnasium import spaces
import math
from typing import Tuple


class AirHockeyEnv(Env):
    def __init__(self,
                 simulator, # box2d or robosuite
                 simulator_params,
                 task, 
                 n_training_steps,
                 wall_bumping_rew,
                 direction_change_rew,
                 horizontal_vel_rew,
                 diagonal_motion_rew,
                 stand_still_rew,
                 terminate_on_out_of_bounds, 
                 terminate_on_enemy_goal, 
                 terminate_on_puck_stop,
                 truncate_rew,
                 goal_max_x_velocity, 
                 goal_min_y_velocity, 
                 goal_max_y_velocity,
                 seed,
                 max_timesteps=1000):
        
            
        self.simulator_params = simulator_params

        self.max_timesteps = max_timesteps
        self.current_timestep = 0
        self.n_training_steps = n_training_steps
        self.n_timesteps_so_far = 0
        self.seed = seed
        np.random.seed(seed)
        
        # termination conditions
        self.terminate_on_out_of_bounds = terminate_on_out_of_bounds
        self.terminate_on_enemy_goal = terminate_on_enemy_goal
        self.terminate_on_puck_stop = terminate_on_puck_stop
        
        # reward function
        self.goal_conditioned = True if 'goal' in task else False
        self.goal_radius_type = 'home'
        self.goal_min_x_velocity = -goal_max_x_velocity
        self.goal_max_x_velocity = goal_max_x_velocity
        self.goal_min_y_velocity = goal_min_y_velocity
        self.goal_max_y_velocity = goal_max_y_velocity
        self.reward_type = task
        self.multiagent = self.simulator_params['num_paddles'] == 2
        self.truncate_rew = truncate_rew
        self.wall_bumping_rew = wall_bumping_rew
        self.direction_change_rew = direction_change_rew
        self.horizontal_vel_rew = horizontal_vel_rew
        self.diagonal_motion_rew = diagonal_motion_rew
        self.stand_still_rew = stand_still_rew
        
        self.width = simulator_params['width']
        self.length = simulator_params['length']
        self.paddle_radius = simulator_params['paddle_radius']
        self.puck_radius = simulator_params['puck_radius']
        
        self.table_x_top = -self.length / 2
        self.table_x_bot = self.length / 2
        self.table_y_right = self.width / 2
        self.table_y_left = -self.width / 2
        self.max_paddle_vel = 1 # TODO: adjust
        self.max_puck_vel = 2
        
        self.initialize_spaces()
        
        self.metadata = {}
        self.reset()

    def initialize_spaces(self):
        # setup observation / action / reward spaces
        low = np.array([self.table_x_top, self.table_y_left, -self.max_paddle_vel, -self.max_paddle_vel, 
                        self.table_x_top, self.table_y_left, -self.max_puck_vel, -self.max_puck_vel])

        high = np.array([self.table_x_bot, self.table_y_right, self.max_paddle_vel, self.max_paddle_vel, 
                         self.table_x_bot, self.table_y_right, self.max_puck_vel, self.max_puck_vel])
        
        if not self.goal_conditioned:
            self.observation_space = Box(low=low, high=high, shape=(8,), dtype=float)
        else:
            
            if self.reward_type == 'goal_position':
                # y, x
                goal_low = np.array([self.table_x_top, self.table_y_left])#, -self.max_paddle_vel, self.max_paddle_vel])
                goal_high = np.array([0, self.table_y_right])#, self.max_paddle_vel, self.max_paddle_vel])
                
                self.observation_space = spaces.Dict(dict(
                    observation=Box(low=low, high=high, shape=(8,), dtype=float),
                    desired_goal=Box(low=goal_low, high=goal_high, shape=(2,), dtype=float),
                    achieved_goal=Box(low=goal_low, high=goal_high, shape=(2,), dtype=float)
                ))
            
            elif self.reward_type == 'goal_position_velocity':
                goal_low = np.array([self.table_x_top, self.table_y_left, -self.max_puck_vel, -self.max_puck_vel])
                goal_high = np.array([0, self.table_y_right, self.max_puck_vel, self.max_puck_vel])
                self.observation_space = spaces.Dict(dict(
                    observation=Box(low=low, high=high, shape=(8,), dtype=float),
                    desired_goal=Box(low=goal_low, high=goal_high, shape=(4,), dtype=float),
                    achieved_goal=Box(low=goal_low, high=goal_high, shape=(4,), dtype=float)
                ))
            
            self.min_goal_radius = self.width / 16
            self.max_goal_radius = self.width / 4
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32) # 2D action space
        self.reward_range = Box(low=-1, high=1) # need to make sure rewards are between 0 and 1

    @staticmethod
    def from_dict(state_dict):
        return AirHockeyEnv(**state_dict)

    def get_observation(self, state_info):
        ego_paddle_x_pos = state_info['paddles']['paddle_ego']['position'][0]
        ego_paddle_y_pos = state_info['paddles']['paddle_ego']['position'][1]
        ego_paddle_x_vel = state_info['paddles']['paddle_ego']['velocity'][0]
        ego_paddle_y_vel = state_info['paddles']['paddle_ego']['velocity'][1]
        puck_x_pos = state_info['pucks'][0]['position'][0]
        puck_y_pos = state_info['pucks'][0]['position'][1]
        puck_x_vel = state_info['pucks'][0]['velocity'][0]
        puck_y_vel = state_info['pucks'][0]['velocity'][1]

        if not self.multiagent:
            obs = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel, puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
        else:
            alt_paddle_x_pos = state_info['paddles']['paddle_alt']['position'][0]
            alt_paddle_y_pos = state_info['paddles']['paddle_alt']['position'][1]
            alt_paddle_x_vel = state_info['paddles']['paddle_alt']['velocity'][0]
            alt_paddle_y_vel = state_info['paddles']['paddle_alt']['velocity'][1]
            
            obs_ego = np.array([ego_paddle_x_pos, ego_paddle_y_pos, ego_paddle_x_vel, ego_paddle_y_vel,  puck_x_pos, puck_y_pos, puck_x_vel, puck_y_vel])
            obs_alt = np.array([-alt_paddle_x_pos, -alt_paddle_y_pos, alt_paddle_x_vel, alt_paddle_y_vel, -puck_x_pos, -puck_y_pos, -puck_x_vel, -puck_y_vel])
            obs = (obs_ego, obs_alt)
        return obs
    
    def set_goals(self, goal_radius_type, ego_goal_pos=None, alt_goal_pos=None):
        if self.goal_conditioned:
            if goal_radius_type == 'fixed':
                # ego_goal_radius = np.random.uniform(low=self.min_goal_radius, high=self.max_goal_radius)
                base_radius = (self.min_goal_radius + self.max_goal_radius) / 2 * (0.75)
                # linearly decrease radius, should start off at 3*base_radius then decrease to base_radius
                ratio = 2 * (1 - self.n_timesteps_so_far / self.n_training_steps) + 1
                ego_goal_radius = ratio * base_radius
                if self.multiagent:
                    alt_goal_radius = ego_goal_radius      
                self.ego_goal_radius = ego_goal_radius
                if self.multiagent:
                    self.alt_goal_radius = alt_goal_radius
            elif goal_radius_type == 'home':
                self.ego_goal_radius = 0.16 * self.width
                if self.multiagent:
                    self.alt_goal_radius = 0.16 * self.width
            if ego_goal_pos is None:
                min_x = self.table_x_min + self.ego_goal_radius
                max_x = self.table_x_max - self.ego_goal_radius
                min_y = 0 + self.ego_goal_radius
                max_y = self.length / 2 - self.ego_goal_radius
                self.ego_goal_pos = np.random.uniform(low=(min_y, min_x), high=(max_y, max_x))
                
                min_x_vel = self.goal_min_x_velocity
                max_x_vel = self.goal_max_x_velocity
                min_y_vel = self.goal_min_y_velocity
                max_y_vel = self.goal_max_y_velocity
                
                self.ego_goal_vel = np.random.uniform(low=(min_x_vel, min_y_vel), high=(max_x_vel, max_y_vel))
                
                if self.multiagent:
                    self.alt_goal_pos = np.random.uniform(low=(-self.length / 2, self.table_x_min), high=(0, self.table_x_max))
            else:
                self.ego_goal_pos = ego_goal_pos
                if self.multiagent:
                    self.alt_goal_pos = alt_goal_pos
        else:
            self.ego_goal_pos = None
            self.ego_goal_radius = None
            self.alt_goal_pos = None
            self.alt_goal_radius = None
    
    # def convert_to_box2d_coords(self, x, y):
    #     return (x, -y)

    def has_finished(self, state_info, multiagent=False):
        truncated = False
        terminated = False
        puck_within_alt_home = False
        puck_within_home = False

        if self.current_timestep > self.max_timesteps:
            terminated = True
        else:
            if self.terminate_on_out_of_bounds:
                # check if we hit any walls or are above the middle of the board
                if state_info['paddles']['paddle_ego']['position'][0] < 0 + self.paddle_radius or \
                    state_info['paddles']['paddle_ego']['position'][0] > self.table_x_bot - self.paddle_radius or \
                    state_info['paddles']['paddle_ego']['position'][1] > self.table_y_right - self.paddle_radius or \
                    state_info['paddles']['paddle_ego']['position'][1] < self.table_y_left + self.paddle_radius:
                    truncated = True

        # confusing, but we need to swap x and y for this function
        bottom_center_point = np.array([self.table_x_bot, 0])
        top_center_point = np.array([self.table_x_top, 0])
        puck_within_home = self.is_within_home_region(bottom_center_point, state_info['pucks'][0]['position'])
        puck_within_alt_home = self.is_within_home_region(top_center_point, state_info['pucks'][0]['position'])
        
        if self.terminate_on_enemy_goal:
            if not terminated and puck_within_home:
                truncated = True

        if multiagent:
            terminated = terminated or truncated or puck_within_alt_home or puck_within_home
            truncated = False
            
        if self.terminate_on_puck_stop:
            if not truncated and np.linalg.norm(state_info['pucks'][0]['velocity']) < 0.01:
                truncated = True

        puck_within_ego_goal = False
        puck_within_alt_goal = False

        if self.goal_conditioned:
            if self.is_within_goal_region(self.ego_goal_pos, state_info['pucks'][0]['position'], self.ego_goal_radius):
                puck_within_ego_goal = True
            if multiagent:
                if self.is_within_goal_region(self.alt_goal_pos, state_info['pucks'][0]['position'], self.alt_goal_radius):
                    puck_within_alt_goal = True

        return terminated, truncated, puck_within_home, puck_within_alt_home, puck_within_ego_goal, puck_within_alt_goal
    
    def get_goal_region_reward(self, point, position, radius, discrete=True) -> float:
        point = np.array([point[0], point[1]])
        dist = np.linalg.norm(position - point)
        
        if discrete:
            return 1.0 if dist < radius else 0.0
        # also return float from [0, 1] 0 being far 1 being the point
        # use sigmoid function because being closer is much more important than being far
        sigmoid_scale = 2
        reward_raw = 1 - (dist / radius)
        reward = 1 / (1 + np.exp(-reward_raw * sigmoid_scale))
        reward = 0 if dist >= radius else reward
        return reward

    def get_home_region_reward(self, point, position, discrete=True) -> float:
        # this is for the two base regions of each side of the eboard
        # TODO: this may need to be tuned :) let's provide a rough estimate of where the goal is
        # 90 / 560 = 0.16 <- normalized dist in pixels
        return self.get_goal_region_reward(point, position, 0.16 * self.width, discrete=discrete)
    
    def is_within_goal_region(self, point, position, radius) -> bool:
        point = np.array([point[0], point[1]])
        dist = np.linalg.norm(position - point)
        return dist < radius
    
    def is_within_home_region(self, point, position) -> bool:
        return self.is_within_goal_region(point, position, 0.16 * self.width)

    def get_base_reward(self, state_info, hit_a_puck, puck_within_home, 
                       puck_within_alt_home, puck_within_goal,
                       goal_pos, goal_radius):
        if self.reward_type == 'goal_discrete':
            return self.get_goal_region_reward(goal_pos, state_info['pucks'][0]['position'], 
                                                 goal_radius, discrete=True)
        elif self.reward_type == 'goal_position' or self.reward_type == 'goal_position_velocity':
            # return self.get_goal_region_reward(goal_pos, self.pucks[self.puck_names[0]][0], 
            #                                      goal_radius, discrete=False)
            return self.compute_reward(self.get_achieved_goal(self.current_state), self.get_desired_goal(), {})
        elif self.reward_type == 'puck_height':
            reward = -state_info['pucks'][0]['position'][0]
            # min acceptable reward is 0 height and above
            reward = max(reward, 0)
            # let's normalize reward w.r.t. the top half length of the table
            # aka within the range [0, self.length / 2]
            max_rew = self.length / 2
            min_rew = 0
            reward = (reward - min_rew) / (max_rew - min_rew)
            return reward
        elif self.reward_type == 'puck_vel':
            # reward for positive velocity towards the top of the board
            reward = -state_info['pucks'][0]['velocity'][0]
            
            max_rew = 2 # estimated max vel
            min_rew = 0  # min acceptable good velocity
            
            if reward < min_rew:
                return 0
            
            reward = min(reward, max_rew)
            reward = (reward - min_rew) / (max_rew - min_rew)
            return reward
        elif self.reward_type == 'puck_touch':
            reward = 1 if hit_a_puck else 0
            return reward
        elif self.reward_type == 'alt_home':
            reward = 1 if puck_within_alt_home else 0
            return reward
        else:
            raise ValueError("Invalid reward type defined in config.")
        
    def get_reward_shaping(self, state_info):
        additional_rew = 0.0
        
        # small negative reward for changing direction
        if self.current_timestep > 0:
            old_vel = self.old_state['paddles']['paddle_ego']['velocity']
            new_vel = state_info['paddles']['paddle_ego']['velocity']
            vel_unit = old_vel / (np.linalg.norm(old_vel) + 1e-8)
            new_vel_unit = new_vel / (np.linalg.norm(new_vel) + 1e-8)
            cosine_sim = np.dot(vel_unit, new_vel_unit) / (np.linalg.norm(vel_unit) * np.linalg.norm(new_vel_unit) + 1e-8)
            norm_cosine_sim = (cosine_sim + 1) / 2
            max_change_dir_rew = self.direction_change_rew
            direction_rew = max_change_dir_rew * (1 - norm_cosine_sim)
            additional_rew += direction_rew
            
        # small negative reward for moving too fast in horizontal direction
        max_vel = self.max_paddle_vel
        max_vel_rew = self.horizontal_vel_rew
        normalized_y_vel = np.abs(state_info['paddles']['paddle_ego']['velocity'][1]) / max_vel
        additional_rew += max_vel_rew * normalized_y_vel
        
        # negative penalty for diagonal motion
        # angle of vector will be close to % 45 degrees if moving diagonally
        angle = np.arctan2(state_info['paddles']['paddle_ego']['velocity'][1], state_info['paddles']['paddle_ego']['velocity'][0])
        angle = np.abs(angle)
        # check if sufficiently close to pi/4, 3pi/4, 5pi/4, 7pi/4
        threshold = np.pi / 12
        # check if between (pi/4 - pi/12, pi/4 + pi/12), ...
        if np.abs(angle - -np.pi / 4) < threshold or np.abs(angle - 3 * -np.pi / 4) < threshold or \
            np.abs(angle - np.pi / 4) < threshold or np.abs(angle - 3 * np.pi / 4) < threshold:
            additional_rew += self.diagonal_motion_rew
        
        # small positive reward for keeping still
        if np.linalg.norm(state_info['paddles']['paddle_ego']['velocity']) < 0.01:
            additional_rew += self.stand_still_rew
            
        # determine if close to walls
        if self.wall_bumping_rew != 0:
            bump_right = state_info['paddles']['paddle_ego']['position'][1] > self.table_y_right - 2 * self.paddle_radius
            bump_left = state_info['paddles']['paddle_ego']['position'][1] < self.table_y_left + 2 * self.paddle_radius
            bump_top = state_info['paddles']['paddle_ego']['position'][0] < 0 + 4 * self.paddle_radius
            bump_bottom = state_info['paddles']['paddle_ego']['position'][0] > self.table_x_bot - 4 * self.paddle_radius
            if bump_left or bump_right or bump_top or bump_bottom:
                additional_rew += self.wall_bumping_rew
        
        # todo: figure out how to determine if puck was hit by object.
        # contacts, contact_names = self.get_contacts()
        # hit_a_puck = self.respond_contacts(contact_names)
        # # hacky way of determing if puck was hit below TODO: fix later!
        # hit_a_puck = np.any(contacts) # check if any are true
        return additional_rew
        
    
    def get_joint_reward(self, ego_hit_a_puck, alt_hit_a_puck, 
                         puck_within_ego_home, puck_within_alt_home,
                         puck_within_ego_goal, puck_within_alt_goal) -> Tuple[float, float]:
        ego_reward = self.get_reward(ego_hit_a_puck, puck_within_ego_home, 
                                     puck_within_alt_home, puck_within_ego_goal,
                                     self.ego_goal_pos, self.ego_goal_radius)
        alt_reward = self.get_reward(alt_hit_a_puck, puck_within_alt_home,
                                     puck_within_ego_home, puck_within_alt_goal,
                                     self.alt_goal_pos, self.alt_goal_radius)
        return ego_reward, alt_reward
    
    def step(self, action):
        if not self.multiagent:
            obs, reward, is_finished, truncated, info = self.single_agent_step(action)
            if not self.goal_conditioned:
                return obs, reward, is_finished, truncated, info
            else:
                return {"observation": obs, "desired_goal": self.get_desired_goal(), "achieved_goal": self.get_achieved_goal(self.current_state)}, reward, is_finished, truncated, info
        else:
            return self.multi_step(action)

    def single_agent_step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        next_state = self.simulator.get_transition(action)
        if self.current_timestep > 0:
            self.old_state = self.current_state
        self.current_state = next_state

        hit_a_puck = False
        is_finished, truncated, puck_within_home, puck_within_alt_home, puck_within_goal, _ = self.has_finished(next_state)
        if not truncated:
            reward = self.get_base_reward(next_state, hit_a_puck, puck_within_home, 
                                     puck_within_alt_home, puck_within_goal,
                                     self.ego_goal_pos, self.ego_goal_radius)
        else:
            reward = self.truncate_rew
        reward += self.get_reward_shaping(next_state)
        self.current_timestep += 1
        
        obs = self.get_observation(next_state)
        return obs, reward, is_finished, truncated, {}
    
    def multi_step(self, joint_action):
        raise NotImplementedError("Multi-agent step function not implemented yet. But shouldn't take much work, it is mostly copy-pasting. But need to do specific rewards per player")