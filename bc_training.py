from autonomous.bc import BehaviorCloning

bc = BehaviorCloning([256, 256], 'cuda', 3e-4, 128, 50000, input_mode='goal', save_dir='/datastor1/siddhant/air-hockey/goal/')
# bc.populate_buffer()
# bc.load_state_data()
# bc.load_data()
# bc.load_model('/datastor1/siddhant/air-hockey/frame_stack/bc_model_5000.pt')
bc.train_offline()