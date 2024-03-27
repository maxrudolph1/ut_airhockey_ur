import torch
import numpy as np

class BCBuffer:
    def __init__(self, obs_dim, act_dim, device, val_split = 0.2, size=5000):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)

        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device
        self.val_split = val_split

    def store(self, obs, act):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def store_all(self, obs, act):
        self.obs_buf = np.array(obs)
        self.act_buf = np.array(act)
        self.size = len(obs)
        self.train_ids = np.random.randint(low=0, high=self.size, size=int(self.size*(1-self.val_split)), replace=False)
        self.val_ids = np.array([i for i in range(self.size) if i not in self.train_ids])

    def sample_batch_train(self, batch_size=32):
        idxs = np.random.choice(self.train_ids, size=batch_size)
        return dict(
            obs=torch.tensor(self.obs_buf[idxs], dtype=torch.float32).to(self.device),
            act=torch.tensor(self.act_buf[idxs], dtype=torch.float32).to(self.device)
            )
    def sample_batch_val(self, batch=32):
        idxs = np.random.choice(self.val_ids, size=batch)
        return dict(
            obs=torch.tensor(self.obs_buf[idxs], dtype=torch.float32).to(self.device),
            act=torch.tensor(self.act_buf[idxs], dtype=torch.float32).to(self.device)
            )



class OfflineBuffer:
    def __init__(self, obs_dim, act_dim, img_size, device, size=5000):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.img_obs_buf = np.zeros((size, *img_size), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_img_obs_buf = np.zeros((size, *img_size), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, img_obs, act, reward, next_obs, next_img_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.img_obs_buf[self.ptr] = img_obs
        self.next_obs_buf[self.ptr] = next_obs
        self.next_img_obs_buf[self.ptr] = next_img_obs
        self.rew_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def store_all(self, obs, img_obs, act, reward, next_obs, next_img_obs, done):
        self.obs_buf = np.array(obs)
        self.act_buf = np.array(act)
        self.img_obs_buf = torch.stack(img_obs)
        self.next_obs_buf = np.array(next_obs)
        self.next_img_obs_buf = torch.stack(next_img_obs)
        self.rew_buf = np.array(reward)
        self.done_buf = np.array(done)
        self.size = len(obs)

    def sample(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=torch.tensor(self.obs_buf[idxs]).to(self.device),
            img_obs=self.img_buf[idxs].to(self.device),
            act=torch.tensor(self.act_buf[idxs]).to(self.device),
            reward=torch.tensor(self.rew_buf[idxs]).to(self.device),
            next_obs=torch.tensor(self.next_obs_buf[idxs]).to(self.device),
            next_img_obs=self.next_img_obs_buf[idxs].to(self.device),
            done=torch.tensor(self.done_buf[idxs]).to(self.device)
            )

class PPOBuffer:
    def __init__(self, obs_dim, act_dim, size, device, gamma=0.99, lam=0.95):
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
        self.device = device

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp

        self.ptr = (self.ptr + 1) % self.max_size

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
        return dict(obs=torch.as_tensor(self.obs_buf[idxs], dtype=torch.float32).to(self.device),
                    act=torch.as_tensor(self.act_buf[idxs], dtype=torch.float32).to(self.device),
                    adv=torch.as_tensor(self.adv_buf[idxs], dtype=torch.float32).to(self.device),
                    ret=torch.as_tensor(self.ret_buf[idxs], dtype=torch.float32).to(self.device),
                    logp=torch.as_tensor(self.logp_buf[idxs], dtype=torch.float32).to(self.device))