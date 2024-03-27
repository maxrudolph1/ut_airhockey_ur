from autonomous.random_agent import RandomAgent
from autonomous.bc import BehaviorCloning
from autonomous.ppo import PPO
from real_world_human_input.input_photo import find_red_hockey_puck

def initialize_agent(control_mode, load_path, additional_args):
    puck_detector = find_red_hockey_puck
    agent = None
    if control_mode == "RL":
        if additional_args["algo"] == "ppo":
            agent = PPO([ 512], "cuda:0", 0.0004, clip_ratio=0.2, gamma=0.99, lam=0.9, 
                        batch_size=128, num_trajs=1, num_ppo_updates=10, buffer_size=10000, 
                        img_size=(224, 224), puck_history_len=5)
    if control_mode == "BC":
        if additional_args["image_input"]:
            agent = BehaviorCloning(list(), "cuda:0", 0.0004, 512, 1, input_mode='img', frame_stack=additional_args["frame_stack"])
        else:
            agent = BehaviorCloning([512], "cuda:0", 0.0004, 512, 1, puck_history_len = 5, input_mode='img', target_config='train_ppo.yaml', dataset_path='/datastor1/calebc/public/data', data_mode='mouse')
    if control_mode == 'rnet':
        agent = RandomAgent([512], puck_detector=puck_detector)
    if len(load_path) > 0:
        agent.load_model(load_path)
    return agent
    