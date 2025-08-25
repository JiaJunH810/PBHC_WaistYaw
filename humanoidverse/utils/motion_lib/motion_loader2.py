import os
import glob
import json
import random
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from scipy.spatial.transform import Slerp
from pathlib import Path
from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
from humanoidverse.utils.motion_lib.skeleton import SkeletonTree
from humanoidverse.utils.torch_utils import *
import joblib
import time
def to_torch(tensor, device='cpu'):
    if torch.is_tensor(tensor):
        return tensor.to(device)
    else:
        if isinstance(tensor, list):
            tensor = np.array(tensor)
        return torch.from_numpy(tensor).to(device)
class AMPLoader:
    def __init__(self, config, num_envs, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # 初始化阶段使用 CPU
        self.cpu_device = 'cpu'
        self.device = device  # 用于后续计算
        
        self.num_envs = num_envs
        self.config = config
        
        # 在 CPU 上初始化 mesh_parsers
        self.mesh_parsers = Humanoid_Batch(self.config.robot.motion, device=self.cpu_device)
        
        skeleton_file = Path(self.config.robot.motion.asset.assetRoot) / self.config.robot.motion.asset.assetFileName
        self.skeleton_tree = SkeletonTree.from_mjcf(skeleton_file)
        self.up_axis_idx = 2
        
        # 在 CPU 上加载数据
        self.load_data(self.config.robot.motion.motion_file)
        
        # 将 obs_scales 转换为 GPU 张量
        self.obs_scales_tensor = {}
        for key, value in self.config.obs.obs_scales.items():
            self.obs_scales_tensor[key] = torch.tensor(value, device=self.device, dtype=torch.float32)

    def load_data(self, motion_file):
        # 在 CPU 上加载数据
        motion_data = joblib.load(motion_file)
        if len(motion_data) > 1:
            raise NotImplementedError("只支持单个 motion 文件")

        trajectories = []
        trajectory_lens = []
        trajectory_idxs = []
        trajectory_num_frames = []
        trajectory_frame_durations = []
        
        for i, key in enumerate(motion_data):
            traj = self.load_motion(motion_data[key])
            trajectories.append(traj)
            time_between_frames = 1.0 / traj['fps']
            num_frames = traj['dof_pos'].shape[1]
            trajectory_idxs.append(i)
            trajectory_lens.append((num_frames - 1) * time_between_frames)
            trajectory_num_frames.append(num_frames)
            trajectory_frame_durations.append(time_between_frames)
        
        # 将轨迹数据转移到 GPU
        self.trajectories = []
        for traj in trajectories:
            gpu_traj = {}
            for key, value in traj.items():
                if torch.is_tensor(value):
                    gpu_traj[key] = value.to(self.device)
                else:
                    gpu_traj[key] = value
            self.trajectories.append(gpu_traj)
        
        # 将其他数据也转移到 GPU
        self.trajectory_lens = torch.tensor(trajectory_lens, dtype=torch.float32, device=self.device)
        self.trajectory_idxs = torch.tensor(trajectory_idxs, dtype=torch.int64, device=self.device)
        self.trajectory_num_frames = torch.tensor(trajectory_num_frames, dtype=torch.int64, device=self.device)
        self.trajectory_frame_durations = torch.tensor(trajectory_frame_durations, dtype=torch.float32, device=self.device)

    def load_motion(self, motion):
        # 在 CPU 上加载运动数据
        dt = 1.0 / motion['fps']
        trans = to_torch(motion['root_trans_offset'], self.cpu_device).clone()
        pose_aa = to_torch(motion['pose_aa'], self.cpu_device).clone()
        num_frame = trans.shape[0]
        
        # 在 CPU 上计算运动
        curr_motion = self.mesh_parsers.fk_batch(pose_aa[None, ], trans[None, ], return_full=True, dt=dt)
        
        # 将结果转移到 GPU
        for key in curr_motion:
            if torch.is_tensor(curr_motion[key]):
                curr_motion[key] = curr_motion[key].to(self.device)
        
        # 计算 projected_gravity 并确保在 GPU 上
        gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((num_frame, 1)).float()
        root_rotation = curr_motion['global_rotation_extend'][:, :, 0, :].squeeze()
        curr_motion['projected_gravity'] = quat_rotate_inverse(root_rotation, gravity_vec).unsqueeze(dim=0)
        
        return curr_motion

    # 其他方法保持不变，但确保它们在 GPU 上执行
    def _blend_frame(self, traj, frame0, frame1, blend):
        match = {
            'global_translation_extend': 'root_trans_offset',
            'global_rotation_extend': 'root_rot',
            'global_root_velocity': 'base_lin_vel',
            'global_root_angular_velocity': 'base_ang_vel',
            'dof_pos': 'dof_pos',
            'dof_vels': 'dof_vel',
        }
        blended_frame = {}
        
        # 对线性数据进行线性插值（在 GPU 上）
        linear_keys = [
            'global_root_velocity', 'global_root_angular_velocity', 
            'dof_pos', 'global_translation_extend', 'dof_vels', 'global_rotation_extend',
        ]
        
        for key in linear_keys:
            if key in match.keys():
                # 数据已经在 GPU 上
                data0 = traj[key][0][frame0]
                data1 = traj[key][0][frame1]
                blended_frame[match[key]] = (1.0 - blend) * data0 + blend * data1
        
        blended_frame['root_trans_offset'] = blended_frame['root_trans_offset'][0]
        blended_frame['root_rot'] = blended_frame['root_rot'][0]
        
        # 计算 projected_gravity（在 GPU 上）
        gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).float()
        blended_frame['projected_gravity'] = quat_rotate_inverse(
            blended_frame['root_rot'].unsqueeze(0), 
            gravity_vec.unsqueeze(0)
        ).squeeze()
        
        return blended_frame
    
    def get_obs_dim(self, frame):
        batch_obs = []
        for key in self.config.obs.obs_dict.discriminator_obs:
            # 数据已经在 GPU 上
            data = frame[key]
            batch_obs.append(data * self.obs_scales_tensor[key])
        batch_obs_tensor = torch.cat(batch_obs, dim=0)
        return batch_obs_tensor

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):        
            s, s_next = [], []
            traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
            times = self.traj_time_sample_batch(traj_idxs)
            
            # 批量处理所有轨迹（在 GPU 上）
            for traj_idx, t in zip(traj_idxs, times):
                state = self.get_full_frame_at_time(traj_idx, t)
                next_state = self.get_full_frame_at_time(traj_idx, t + self.trajectory_frame_durations[traj_idx])
                s.append(self.get_obs_dim(state))
                s_next.append(self.get_obs_dim(next_state))
            
            # 在 GPU 上堆叠张量
            s = torch.stack(s, dim=0)
            s_next = torch.stack(s_next, dim=0)
            
            yield s, s_next