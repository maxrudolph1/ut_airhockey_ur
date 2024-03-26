from autonomous.random_agent import RandomAgent
from autonomous.bc import BehaviorCloning
from real_world_human_input.input_photo import find_red_hockey_puck

def initialize_agent(control_mode, load_path, additional_args):
    puck_detector = find_red_hockey_puck
    agent = None
    if control_mode == "RL":
        pass
    if control_mode == "BC":
        if additional_args["image_input"]:
            agent = BehaviorCloning(list(), "cuda:0", 0.0004, 512, 1, input_mode='img')
        else:
            agent = BehaviorCloning([512], "cuda:0", 0.0004, 512, 1, puck_history_len = 5, input_mode='img', target_config='train_ppo.yaml', dataset_path='/datastor1/calebc/public/data', data_mode='mouse')
    if control_mode == 'rnet':
        agent = RandomAgent([512], puck_detector=puck_detector)
    if len(load_path) > 0:
        agent.load(load_path)
    return agent
    