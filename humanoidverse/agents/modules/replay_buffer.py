import torch
import numpy as np
class AMPReplayBuffer:
    def __init__(self, buffer_size, obs_dim, device='cpu'):
        self.buffer_size = buffer_size
        self.device = device
        self.states = torch.zeros((buffer_size, obs_dim), device=device)
        self.next_states = torch.zeros((buffer_size, obs_dim), device=device)
        self.step = 0
        self.num_samples = 0
        self.full = False

    def insert(self, states, next_states):
        num_states = states.shape[0]
        start_idx = self.step
        end_idx = self.step + num_states

        if self.step + num_states > self.buffer_size:
            self.full = True
            self.states[self.step:self.buffer_size] = states[:self.buffer_size - self.step]
            self.next_states[self.step:self.buffer_size] = next_states[:self.buffer_size - self.step]
            self.states[:end_idx - self.buffer_size] = states[self.buffer_size - self.step:]
            self.next_states[:end_idx - self.buffer_size] = next_states[self.buffer_size - self.step:]
        else:
            self.states[start_idx:end_idx] = states
            self.next_states[start_idx:end_idx] = next_states
        # 更新实际样本数和下一个插入位置
        self.num_samples = min(self.buffer_size, max(end_idx, self.num_samples))
        self.step = (self.step + num_states) % self.buffer_size
            
    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            sample_idxs = np.random.choice(self.num_samples, size=mini_batch_size)
            yield (self.states[sample_idxs].to(self.device),
                   self.next_states[sample_idxs].to(self.device))
