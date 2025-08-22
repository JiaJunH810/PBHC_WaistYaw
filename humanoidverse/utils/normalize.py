from typing import Tuple

import torch
import numpy as np
class AMPNormalizer:
    def __init__(self, size, eps=1e-8, clip_range=5.0):
        self.size = size
        self.eps = eps
        self.clip_range = clip_range
        
        self.sum = torch.zeros(size)
        self.sumsq = torch.zeros(size)
        self.count = torch.zeros(1)
        
        self.mean = torch.zeros(size)
        self.std = torch.ones(size)
        
    def update(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
            
        x = x.reshape(-1, self.size)
        self.sum += x.sum(axis=0)
        self.sumsq += (x ** 2).sum(axis=0)
        self.count += x.shape[0]
        
        self.mean = self.sum / self.count
        self.std = np.sqrt(np.maximum(self.sumsq / self.count - self.mean ** 2, self.eps))
        
    def normalize(self, x):
        if isinstance(x, torch.Tensor):
            x_np = x.cpu().numpy()
            normalized = (x_np - self.mean) / (self.std + self.eps)
            normalized = np.clip(normalized, -self.clip_range, self.clip_range)
            return torch.from_numpy(normalized).to(x.device)
        else:
            normalized = (x - self.mean) / (self.std + self.eps)
            return np.clip(normalized, -self.clip_range, self.clip_range)
        
    def normalize_torch(self, x, device):
        x_np = x.cpu().numpy()
        normalized = (x_np - self.mean) / (self.std + self.eps)
        normalized = np.clip(normalized, -self.clip_range, self.clip_range)
        return torch.from_numpy(normalized).to(device)
