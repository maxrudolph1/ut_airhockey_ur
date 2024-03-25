from autonomous.random_agent import RandomAgent
from real_world_human_input.input_photo import find_red_hockey_puck

def initialize_agent(control_mode, load_path):
    puck_detector = find_red_hockey_puck
    agent = None
    if control_mode == "RL":
        pass
    if control_mode == "BC":
        pass
    if control_mode == 'rnet':
        agent = RandomAgent([512], puck_detector=None)
    if len(load_path) > 0:
        agent.load(load_path)
    return agent
    