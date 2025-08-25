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
        trajectory_lens = []
        trajectory_idxs = []
        trajectory_num_frames = []
        trajectory_frame_durations = []
        # 一个traj包括
        # global_velocity_extend,global_angular_velocity_extend,global_translation_extend,
        # global_rotation_mat_extend,global_rotation_extend,global_translation,global_rotation_mat
        # global_rotation,local_rotation,global_root_velocity,global_root_angular_velocity,
        # global_velocity,dof_pos,dof_vels
        for i, key in enumerate(motion_data):
            traj = self.load_motion(motion_data[key])
            trajectories.append(traj)
            time_between_frames = 1.0 / traj['fps']
            num_frames = traj['dof_pos'].shape[1]
            trajectory_idxs.append(i)
            trajectory_lens.append((num_frames - 1) * time_between_frames)
            trajectory_num_frames.append(num_frames)
            trajectory_frame_durations.append(time_between_frames)

        # 将数据放到gpu上
        trajectories_gpu = []
        for traj in trajectories:
            traj_gpu = {}
            for key, value in traj.items():
                if isinstance(value, torch.Tensor):
                    traj_gpu[key] = value.to(self.device)
                else:
                    traj_gpu[key] = value  # 非张量保持原样
            trajectories_gpu.append(traj_gpu)
            

        self.trajectories = trajectories_gpu
        self.trajectory_lens = torch.tensor(trajectory_lens, dtype=torch.float32).to(self.device)
        self.trajectory_idxs = torch.tensor(trajectory_idxs, dtype=torch.int64).to(self.device)  # 索引用int64
        self.trajectory_num_frames = torch.tensor(trajectory_num_frames, dtype=torch.int64).to(self.device)
        self.trajectory_frame_durations = torch.tensor(trajectory_frame_durations, dtype=torch.float32).to(self.device)

    def load_motion(self, motion):
        # 读取数据
        dt = 1.0 / motion['fps']
        trans = to_torch(motion['root_trans_offset'], device='cpu').clone()
        pose_aa = to_torch(motion['pose_aa'], device='cpu').clone()
        num_frame = trans.shape[0]
        curr_motion = self.mesh_parsers.fk_batch(pose_aa[None, ], trans[None, ], return_full= True, dt = dt)
        gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device='cpu').repeat((num_frame, 1)).float()
        root_rotation = curr_motion['global_rotation_extend'][:, :, 0, :].squeeze()
        curr_motion['projected_gravity'] = quat_rotate_inverse(root_rotation, gravity_vec).unsqueeze(dim=0)
        return curr_motion

    def weighted_traj_idx_sample(self):
        return torch.randint(0, len(self.trajectory_idxs), (1,), device=self.device).item()

    def weighted_traj_idx_sample_batch(self, size):
        return torch.randint(0, len(self.trajectory_idxs), (size,), device=self.device)

    def traj_time_sample(self, traj_idx):
        subst = self.trajectory_frame_durations[traj_idx]
        traj_len = self.trajectory_lens[traj_idx]
        return max(0, traj_len * torch.size(size=(1,), device=self.device) - subst)

    def traj_time_sample_batch(self, traj_idxs):
        subst = self.trajectory_frame_durations[traj_idxs]
        traj_lens = self.trajectory_lens[traj_idxs]
        time_samples = traj_lens * torch.rand(size=(len(traj_idxs),), device=self.device) - subst
        return torch.clamp_min(time_samples, min=0.0)

    def get_full_frame_at_time(self, traj_idx, time):
        traj = self.trajectories[traj_idx]
        p = float(time) / self.trajectory_lens[traj_idx].item()
        n = self.trajectory_num_frames[traj_idx].item()
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        blend = p * n - idx_low

        # 确保索引在有效范围内
        idx_low = min(max(0, idx_low), n - 1)
        idx_high = min(max(0, idx_high), n - 1)
        blend_frame = self._blend_frame(traj, idx_low, idx_high, blend)
        return blend_frame

    # 需要的观测数据有base_lin_vel、base_ang_vel、projected_gravity、dof_pos、dof_vel
    # 还需要的数据有root_trans_offset、root_rot
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
        # 对线性数据进行线性插值
        linear_keys = [
            'global_root_velocity', 'global_root_angular_velocity', 
            'dof_pos', 'global_translation_extend', 'dof_vels',
        ]
        
        for key in linear_keys:
                if key in match.keys():
                    blended_frame[match[key]] = (1.0 - blend) * traj[key][0][frame0] + blend * traj[key][0][frame1]
        blended_frame['root_trans_offset'] = blended_frame['root_trans_offset'][0]
        # 对旋转数据进行球面线性插值
        rotation_keys = ['global_rotation_extend']
        for key in rotation_keys:
            # 将四元数转换为scipy的Rotation对象进行插值
            rot0 = traj[key][0][frame0][0].view(-1, 4).cpu().numpy()
            rot1 = traj[key][0][frame1][0].view(-1, 4).cpu().numpy()
            
            blended_rot = np.zeros_like(rot0)
            for i in range(rot0.shape[0]):
                r0 = sRot.from_quat(rot0[i])
                r1 = sRot.from_quat(rot1[i])
                slerp = Slerp([0, 1], sRot.concatenate([r0, r1]))
                blended_rot[i] = slerp([blend]).as_quat()
            if key in match.keys():
                blended_frame[match[key]] = to_torch(blended_rot, self.device)

        # 计算 projected_gravity（使用球形插值）
        gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).float()
        blended_frame['projected_gravity'] = quat_rotate_inverse(blended_frame['root_rot'], gravity_vec.unsqueeze(0)).squeeze()
        return blended_frame
    
    def get_obs_dim(self, frame):
        batch_obs = []
        obs_scales = self.config.obs.obs_scales
        # print(obs_scales)
        for key in self.config.obs.obs_dict.discriminator_obs:
            batch_obs.append(frame[key] * obs_scales[key])
        batch_obs_tensor = torch.cat(batch_obs, dim=0)
        return batch_obs_tensor

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        frames = []
        for traj_idx, t in zip(traj_idxs, times):
            frames.append(self.get_full_frame_at_time(traj_idx, t))
        return frames

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
            for traj_idx, t in zip(traj_idxs, times):
                state = self.get_full_frame_at_time(traj_idx, t)
                next_state = self.get_full_frame_at_time(traj_idx, t + self.trajectory_frame_durations[traj_idx])
                s.append(self.get_obs_dim(state))
                s_next.append(self.get_obs_dim(next_state))

            s = torch.stack(s, dim=0)
            s_next = torch.stack(s_next, dim=0)
            yield s, s_next

    @property
    def num_motions(self):
        return len(self.trajectories)

    @staticmethod
    def get_joint_pose_batch(frames):
        # DOF位置
        dof_pos_list = []
        for frame in frames:
            dof_pos = frame['dof_pos']
            dof_pos_list.append(dof_pos.squeeze())  # [num_dofs]
        return torch.stack(dof_pos_list, dim=0)  # [num_envs, num_dofs]

    @staticmethod
    def get_joint_vel_batch(frames):
        # DOF速度
        dof_vel_list = []
        for frame in frames:
            dof_vel = frame['dof_vel']
            dof_vel_list.append(dof_vel.squeeze())
        return torch.stack(dof_vel_list, dim=0)

    @staticmethod
    def get_root_trans_offset_batch(frames):
        root_trans_offset_list = []
        for frame in frames:
            if 'root_trans_offset' in frame:
                root_trans_offset = frame['root_trans_offset']
            else:
                raise KeyError("frame 中没有 root_trans_offset，请在 get_full_frame_at_time 中加入 root_trans_offset")
            root_trans_offset_list.append(root_trans_offset.squeeze(0))
        return torch.stack(root_trans_offset_list, dim=0)

    @staticmethod
    def get_root_rot_batch(frames):
        # 根部旋转（四元数）
        root_rot_list = []
        for frame in frames:
            if 'root_rot' in frame:
                root_rot = frame['root_rot']
            else:
                raise KeyError("frame 中没有 root_rot，请在 get_full_frame_at_time 中加入 root_rot")
            root_rot_list.append(root_rot.squeeze())
        return torch.stack(root_rot_list, dim=0)

    @staticmethod
    def get_linear_vel_batch(frames):
        # 基座线速度
        base_lin_vel_list = []
        for frame in frames:
            base_lin_vel = frame['base_lin_vel']
            base_lin_vel_list.append(base_lin_vel.squeeze())
        return torch.stack(base_lin_vel_list, dim=0)

    @staticmethod
    def get_angular_vel_batch(frames):
        # 基座角速度
        base_ang_vel_list = []
        for frame in frames:
            base_ang_vel = frame['base_ang_vel']
            base_ang_vel_list.append(base_ang_vel.squeeze())
        return torch.stack(base_ang_vel_list, dim=0)

