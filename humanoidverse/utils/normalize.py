from typing import Tuple
import torch
import numpy as np

class AMPNormalizer:
    def __init__(self, size, eps=1e-8, clip_range=5.0, device='cpu'):
        self.size = size
        self.eps = eps
        self.clip_range = clip_range
        self.device = device
        
        # 使用 PyTorch 张量而不是 NumPy 数组
        self.sum = torch.zeros(size, device=device)
        self.sumsq = torch.zeros(size, device=device)
        self.count = torch.zeros(1, device=device)
        
        self.mean = torch.zeros(size, device=device)
        self.std = torch.ones(size, device=device)
        
    def update(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)
            
        x = x.reshape(-1, self.size)
        self.sum += x.sum(dim=0)
        self.sumsq += (x ** 2).sum(dim=0)
        self.count += x.shape[0]
        
        self.mean = self.sum / self.count
        self.std = torch.sqrt(torch.clamp(self.sumsq / self.count - self.mean ** 2, min=self.eps))
        
    def normalize(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self.device)
            normalized = (x - self.mean) / (self.std + self.eps)
            normalized = torch.clamp(normalized, -self.clip_range, self.clip_range)
            return normalized.cpu().numpy() if not torch.is_tensor(x) else normalized
        else:
            normalized = (x - self.mean) / (self.std + self.eps)
            return torch.clamp(normalized, -self.clip_range, self.clip_range)
        
    def normalize_torch(self, x, device=None):
        if device is None:
            device = self.device
            
        # 检查并转换输入数据类型
        if isinstance(x, np.ndarray):
            # 如果数据类型是object，尝试转换为float32
            if x.dtype == np.object_:
                x = x.astype(np.float32)
            x = torch.from_numpy(x).to(device)
        elif isinstance(x, list):
            x = np.array(x, dtype=np.float32)  # 确保转换为数值类型
            x = torch.from_numpy(x).to(device)
        elif isinstance(x, torch.Tensor):
            # 如果已经是Tensor，确保它在正确的设备上
            x = x.to(device)
        else:
            raise TypeError(f"Unsupported input type: {type(x)}")
            
        normalized = (x - self.mean.to(device)) / (self.std.to(device) + self.eps)
        normalized = torch.clamp(normalized, -self.clip_range, self.clip_range)
        return normalized