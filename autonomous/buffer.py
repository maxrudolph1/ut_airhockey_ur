import torch
import numpy as np

class BCBuffer:
    def __init__(self, obs_dim, act_dim, img_size, device, size=5000):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.img_buf = np.zeros((size, *img_size), dtype=np.float32)

        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, img):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.img_buf[self.ptr] = img
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def store_all(self, obs, act, img):
        self.obs_buf = obs
        self.act_buf = act
        self.img_buf = img
        self.size = len(obs)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=torch.tensor(self.obs_buf[idxs]).to(self.device),
            img=torch.tensor(self.img_buf[idxs]).to(self.device),
            act=torch.tensor(self.act_buf[idxs]).to(self.device)
            )
    
class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.mean_ret = 0

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp

    def reset(self):
        self.ptr, self.path_start_idx = 0, 0

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def discount_cumsum(self, x, discount):
        return torch.tensor([sum([discount ** i * x[j + i] for i in range(len(x) - j)]) for j in range(len(x))])

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.ptr, size=batch_size)
        return dict(obs=torch.as_tensor(self.obs_buf[idxs], dtype=torch.float32),
                    act=torch.as_tensor(self.act_buf[idxs], dtype=torch.float32),
                    adv=torch.as_tensor(self.adv_buf[idxs], dtype=torch.float32),
                    ret=torch.as_tensor(self.ret_buf[idxs], dtype=torch.float32),
                    logp=torch.as_tensor(self.logp_buf[idxs], dtype=torch.float32))