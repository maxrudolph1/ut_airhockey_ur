import os, yaml
from airhockey import AirHockeyEnv

class AutonomousModel():
    def __init__(self, target_config = 'train_ppo.yaml'):
        with open(os.path.join("autonomous", target_config), 'r') as f:
            air_hockey_cfg = yaml.safe_load(f)
            air_hockey_params = air_hockey_cfg['air_hockey']
            air_hockey_params['n_training_steps'] = air_hockey_cfg['n_training_steps']
            air_hockey_params['seed'] = air_hockey_cfg['seed']
            self.env = AirHockeyEnv.from_dict(air_hockey_params)


    def take_action(self, true_pose, true_speed, true_force, measured_acc, estop, image):
        return 0,0
    
    def train(self, images, poses):
        return {}

    def load(self, load_path):
        # loads the network for the agent from the path
        return self