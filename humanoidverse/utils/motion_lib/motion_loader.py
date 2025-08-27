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
from isaac_utils.rotations import slerp
from collections import OrderedDict

def to_torch(tensor, device='cpu'):
    if torch.is_tensor(tensor):
        return tensor.to(device)
    else:
        if isinstance(tensor, list):
            tensor = np.array(tensor)
        return torch.from_numpy(tensor).to(device)

class AMPLoader:

    def __init__(self, config, num_envs, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_envs = num_envs
        self.config = config
        self.mesh_parsers = Humanoid_Batch(self.config.robot.motion, device='cpu')
        skeleton_file = Path(self.config.robot.motion.asset.assetRoot) / self.config.robot.motion.asset.assetFileName
        self.skeleton_tree = SkeletonTree.from_mjcf(skeleton_file)
        # 向上的轴是z轴，所以索引是2
        self.up_axis_idx = 2
        self.load_data(self.config.robot.motion.motion_file)

    # 设置只加载单个 motion 文件，但是会当做有多个motion进行处理
    def load_data(self, motion_file):

        motion_data = joblib.load(motion_file)
        if len(motion_data) > 1:
            raise NotImplementedError("只支持单个 motion 文件")

        trajectories = []
        trajectories_fps = []
        trajectory_idxs = []
        # 一个traj包括
        # global_velocity_extend,global_angular_velocity_extend,global_translation_extend,
        # global_rotation_mat_extend,global_rotation_extend,global_translation,global_rotation_mat
        # global_rotation,local_rotation,global_root_velocity,global_root_angular_velocity,
        # global_velocity,dof_pos,dof_vels
        for i, key in enumerate(motion_data):
            traj = self.load_motion(motion_data[key])
            trajectories.append(traj)
            trajectories_fps.append(motion_data[key]['fps'])
            trajectory_idxs.append(i)

        self.num_motions = len(trajectories)
        num_frames_list = [t['dof_pos'].shape[0] for t in trajectories]
        self.trajectory_idxs = torch.tensor(trajectory_idxs, dtype=torch.int).to(self.device)
        self.trajectory_num_frames = torch.tensor(num_frames_list, dtype=torch.int, device=self.device)
        self.trajectory_frame_durations = torch.tensor([1.0 / fps for fps in trajectories_fps], dtype=torch.float32, device=self.device)
        self.trajectory_lens = (self.trajectory_num_frames.float() - 1) * self.trajectory_frame_durations

        trajectories_start = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.int), self.trajectory_num_frames[:-1].cpu()]), dim=0)
        self.traj_start_offsets = trajectories_start.to(self.device)
        self.trajectories = dict()
        for key in traj.keys():
            self.trajectories[key] = torch.cat([t[key] for t in trajectories], dim=0).to(self.device)

    def load_motion(self, motion):
        # 读取数据
        dt = 1.0 / motion['fps']
        trans = to_torch(motion['root_trans_offset'], device='cpu').clone()
        pose_aa = to_torch(motion['pose_aa'], device='cpu').clone()
        curr_motion = self.mesh_parsers.fk_batch(pose_aa[None, ], trans[None, ], return_full= True, dt = dt)
        curr_single_motion = dict()
        curr_single_motion['root_trans_offset'] = curr_motion['global_translation_extend'].squeeze(0)[:, 0, :]
        curr_single_motion['base_lin_vel'] = curr_motion['global_root_velocity'].squeeze(0)
        curr_single_motion['base_ang_vel'] = curr_motion['global_root_angular_velocity'].squeeze(0)
        curr_single_motion['dof_pos'] = curr_motion['dof_pos'].squeeze(0)
        curr_single_motion['dof_vel'] = curr_motion['dof_vels'].squeeze(0)
        curr_single_motion['root_rot'] = curr_motion['global_rotation_extend'].squeeze(0)[:, 0, :]
        return curr_single_motion

    def weighted_traj_idx_sample(self):
        return torch.randint(0, len(self.trajectory_idxs), (1,), device=self.device).item()

    def weighted_traj_idx_sample_batch(self, size):
        return torch.randint(0, len(self.trajectory_idxs), (size,), device=self.device)

    def traj_time_sample(self, traj_idx):
        subst = self.trajectory_frame_durations[traj_idx]
        traj_len = self.trajectory_lens[traj_idx]
        return max(0, traj_len * torch.rand(size=(1,), device=self.device) - subst)

    def traj_time_sample_batch(self, traj_idxs):
        subst = self.trajectory_frame_durations[traj_idxs]
        traj_lens = self.trajectory_lens[traj_idxs]
        time_samples = traj_lens * torch.rand(size=(len(traj_idxs),), device=self.device) - subst
        return torch.clamp_min(time_samples, min=0.0)

    def get_full_frame_at_time(self, traj_idx, time):
        start = self.traj_start_offsets[traj_idx]
        p = float(time) / self.trajectory_lens[traj_idx].item()
        n = self.trajectory_num_frames[traj_idx].item()
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        blend = p * n - idx_low

        # 确保索引在有效范围内
        idx_low = min(max(0, idx_low), n - 1)
        idx_high = min(max(0, idx_high), n - 1)
        blend_frame = self._blend_frame(start + idx_low, start + idx_high, blend)
        return blend_frame

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        starts = self.traj_start_offsets[traj_idxs]
        n = self.trajectory_num_frames[traj_idxs].float()
        T = self.trajectory_lens[traj_idxs]
        p = times / T
        idx_low = torch.floor(p * n).int()
        idx_high = torch.clamp(idx_low + 1, max=(n.int() - 1))
        blend = p * n - idx_low
        blend_frame = self._blend_frame_batch(starts + idx_low, starts + idx_high, blend)
        return blend_frame

    # 需要的观测数据有base_lin_vel、base_ang_vel、projected_gravity、dof_pos、dof_vel
    # 还需要的数据有root_trans_offset、root_rot
    def _blend_frame(self, frame0, frame1, blend):
        blended_frame = {}
        # 对线性数据进行线性插值
        linear_keys = ['root_trans_offset', 'base_lin_vel', 'base_ang_vel', 'dof_pos', 'dof_vel']
        
        for key in linear_keys:
            blended_frame[key] = (1.0 - blend) * self.trajectories[key][frame0] + blend * self.trajectories[key][frame1]
        # 对旋转数据进行球面线性插值
        rotation_keys = ['root_rot']
        for key in rotation_keys:
            rot0 = self.trajectories[key][frame0].unsqueeze(0)
            rot1 = self.trajectories[key][frame1].unsqueeze(0)
            blended_frame[key] = slerp(rot0, rot1, torch.tensor([blend], device=self.device))

        # 计算 projected_gravity（使用球形插值）
        gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).float()
        blended_frame['projected_gravity'] = quat_rotate_inverse(blended_frame['root_rot'], gravity_vec.unsqueeze(0)).squeeze()
        return blended_frame
    
    def _blend_frame_batch(self, frame0, frame1, blend):
        blended_frame = {}
        # 对线性数据进行线性插值
        linear_keys = ['root_trans_offset', 'base_lin_vel', 'base_ang_vel', 'dof_pos', 'dof_vel']
        
        for key in linear_keys:
            blended_frame[key] = torch.lerp(self.trajectories[key][frame0], self.trajectories[key][frame1], blend.unsqueeze(-1))
        # 对旋转数据进行球面线性插值
        rotation_keys = ['root_rot']
        for key in rotation_keys:
            rot0 = self.trajectories[key][frame0]
            rot1 = self.trajectories[key][frame1]
            blended_frame[key] = slerp(rot0, rot1, blend.unsqueeze(-1))

        # 计算 projected_gravity（使用球形插值）
        gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((frame0.shape[0], 1)).float()
        blended_frame['projected_gravity'] = quat_rotate_inverse(blended_frame['root_rot'], gravity_vec).squeeze()
        return blended_frame
    
    def get_obs_dim(self, frame):
        batch_obs = []
        obs_scales = self.config.obs.obs_scales
        # print(obs_scales)
        for key in sorted(self.config.obs.obs_dict.discriminator_obs):
            batch_obs.append(frame[key] * obs_scales[key])
        batch_obs_tensor = torch.cat(batch_obs, dim=0)
        return batch_obs_tensor

    def get_obs_dim_batch(self, frames):
        batch_obs = []
        obs_scales = self.config.obs.obs_scales
        
        for key in sorted(self.config.obs.obs_dict.discriminator_obs):
            if key in frames:
                scaled_obs = frames[key] * obs_scales[key]
                batch_obs.append(scaled_obs)
        
        # 沿着最后一个维度连接所有观测
        batch_obs_tensor = torch.cat(batch_obs, dim=-1)
        return batch_obs_tensor

    def get_full_frame(self):
        traj_idx = self.weighted_traj_idx_sample()
        t = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, t)

    def get_full_frame_batch(self, num_frames):
        traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
        times = self.traj_time_sample_batch(traj_idxs)
        return self.get_full_frame_at_time_batch(traj_idxs, times)

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            s, s_next = [], []
            traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
            times = self.traj_time_sample_batch(traj_idxs)
            states = self.get_full_frame_at_time_batch(traj_idxs, times)
            next_states = self.get_full_frame_at_time_batch(traj_idxs, times + self.trajectory_frame_durations[traj_idxs])
            s = self.get_obs_dim_batch(states)
            s_next = self.get_obs_dim_batch(next_states)
            yield s, s_next

    @staticmethod
    def get_joint_pose_batch(frames):
        return frames['dof_pos']

    @staticmethod
    def get_joint_vel_batch(frames):
        return frames['dof_vel']

    @staticmethod
    def get_root_trans_offset_batch(frames):
        return frames['root_trans_offset']

    @staticmethod
    def get_root_rot_batch(frames):
        return frames['root_rot']

    @staticmethod
    def get_linear_vel_batch(frames):
        return frames['base_lin_vel']

    @staticmethod
    def get_angular_vel_batch(frames):
        return frames['base_ang_vel']

