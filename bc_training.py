from autonomous.bc import BehaviorCloning

bc = BehaviorCloning([256, 256], 'cuda', 3e-4, 128, 50000, input_mode='state', save_dir='/datastor1/siddhant/air-hockey/state_no_force_no_acc_relpos_vel/')
# bc.populate_buffer()
# bc.load_state_data()
# bc.load_data()
# bc.load_model('/datastor1/siddhant/air-hockey/frame_stack/bc_model_5000.pt')
bc.train_offline()