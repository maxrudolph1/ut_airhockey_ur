class AutonomousModel():
    def __init__(self):
        pass

    def take_action(self, true_pose, true_speed, true_force, measured_acc, srvpose, estop, image):
        return 0,0
    
    def train(self, images, poses):
        return {}