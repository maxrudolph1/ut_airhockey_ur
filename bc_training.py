from autonomous.bc import BehaviorCloning

bc = BehaviorCloning([64, 64], 'cuda', 3e-4, 128, 50000, save_dir='/datastor1/mrudolph')
# bc.populate_buffer()
# bc.load_state_data()
# bc.load_data()
# bc.load_model('/datastor1/siddhant/air-hockey/frame_stack/bc_model_5000.pt')
bc.train_offline()