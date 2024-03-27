from autonomous.bc import BehaviorCloning

bc = BehaviorCloning([64, 64], 'cuda', 3e-4, 128, 10)
# bc.populate_buffer()
bc.load_data()
# bc.load_model('models/bc_model_900.pt')
bc.train_offline()