import torch
import numpy as np
class AMPReplayBuffer:
    def __init__(self, buffer_size, obs_dim, device='cpu'):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.device = device

        self.states = torch.zeros((buffer_size, obs_dim), device=device)
        self.next_states = torch.zeros((buffer_size, obs_dim), device=device)
        self.pos = 0
        self.full = False

    def insert(self, state, next_state):
        batch_size = state.shape[0]
        
        if self.pos + batch_size > self.buffer_size:
            self.full = True
            remaining = self.buffer_size - self.pos
            self.states[self.pos:self.pos+remaining] = state[:remaining]
            self.next_states[self.pos:self.pos+remaining] = next_state[:remaining]
            self.pos = 0
            
            # Insert the remaining samples
            remaining = batch_size - remaining
            self.states[self.pos:self.pos+remaining] = state[-remaining:]
            self.next_states[self.pos:self.pos+remaining] = next_state[-remaining:]
            self.pos = remaining
        else:
            self.states[self.pos:self.pos+batch_size] = state
            self.next_states[self.pos:self.pos+batch_size] = next_state
            self.pos += batch_size
            
    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        num_samples = self.buffer_size if self.full else self.pos
        indices = torch.randperm(num_samples, device=self.device)
        
        for i in range(num_mini_batch):
            start_idx = i * mini_batch_size
            end_idx = min((i + 1) * mini_batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            yield (self.states[batch_indices], self.next_states[batch_indices])
