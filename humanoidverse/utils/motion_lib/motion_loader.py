import os
import glob
import json
import random
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from pathlib import Path
from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
from humanoidverse.utils.motion_lib.skeleton import SkeletonTree
import joblib

def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)

class AMPLoader:

    def __init__(self, motion_cfg, num_envs, device):
        self.device = device
        self.num_envs = num_envs
        self.m_cfg = motion_cfg
        self.mesh_parsers = Humanoid_Batch(motion_cfg)
        skeleton_file = Path(self.m_cfg.asset.assetRoot) / self.m_cfg.asset.assetFileName
        self.skeleton_tree = SkeletonTree.from_mjcf(skeleton_file)
        
        self.load_data(self.m_cfg.motion_file)

    # 设置只加载单个 motion 文件，但是会当做有多个motion进行处理
    def load_data(self, motion_file):

        motion_data = joblib.load(motion_file)
        if len(motion_data) > 1:
            raise NotImplementedError("只支持单个 motion 文件")

        self.trajectories = []
        self.trajectory_lens = []
        self.trajectory_idxs = []
        self.trajectory_num_frames = []
        self.trajectory_frame_durations = []
        # 一个traj包括
        # global_velocity_extend,global_angular_velocity_extend,global_translation_extend,
        # global_rotation_mat_extend,global_rotation_extend,global_translation,global_rotation_mat
        # global_rotation,local_rotation,global_root_velocity,global_root_angular_velocity,
        # global_velocity,dof_pos,dof_vels
        for i, motion in enumerate(motion_data):
            traj = self.load_motion(motion)
            self.trajectories.append(traj)
            time_between_frames = 1.0 / traj['fps']
            num_frames = traj['root_trans_offset'].shape[0]
            self.trajectory_idxs.append(i)
            self.trajectory_lens.append((num_frames - 1) * time_between_frames)
            self.trajectory_num_frames.append(num_frames)
            self.trajectory_frame_durations.append(time_between_frames)

        self.trajectories = [torch.tensor(trajectory, device=self.device) for trajectory in self.trajectories]
        self.trajectory_lens = torch.tensor(self.trajectory_lens, device=self.device)

    def load_motion(self, motion):
        # 读取数据
        dt = 1.0 / motion['fps']
        trans = to_torch(motion['root_trans_offset']).clone()
        pose_aa = to_torch(motion['pose_aa']).clone()
        curr_motion = {
            'pose_aa': pose_aa,
            'dt': dt,
            'fps': motion['fps'],
            'root_trans_offset': trans,
        }
        
        return curr_motion

    def weighted_traj_idx_sample(self):
        return np.random.choice(self.trajectory_idxs)

    def weighted_traj_idx_sample_batch(self, size):
        return np.random.choice(self.trajectory_idxs, size=size)

    def traj_time_sample(self, traj_idx):
        subst = self.trajectory_frame_durations[traj_idx]
        return max(0, self.trajectory_lens[traj_idx] * np.random.uniform() - subst)

    def traj_time_sample_batch(self, traj_idxs):
        subst = self.trajectory_frame_durations[traj_idxs]
        time_samples = self.trajectory_lens[traj_idxs] * np.random.uniform(size=len(traj_idxs)) - subst
        return np.maximum(np.zeros_like(time_samples), time_samples)

    def get_full_frame_at_time(self, traj_idx, time):
        p = float(time) / self.trajectory_lens[traj_idx]
        n = self.trajectory_num_frames[traj_idx]
        idx_low, idx_high = int(np.floor(p * n)), int(np.ceil(p * n))
        blend = p * n - idx_low


        frame_start = self.trajectories[traj_idx][idx_low]
        frame_end = self.trajectories[traj_idx][idx_high]

        blend_root_trans_offset, blend_pose_aa = self._blend_frame(frame_start, frame_end, blend)
        dt = 1.0 / self.trajectories[traj_idx]['fps']
        blend_frame = self.mesh_parsers.fk_batch(
            blend_pose_aa[None, ], 
            blend_root_trans_offset[None, ], 
            return_full=True, 
            dt=dt
        )
        return blend_frame

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        frames = []
        for traj_idx, t in zip(traj_idxs, times):
            frames.append(self.get_full_frame_at_time(traj_idx, t))
        return frames

    def _blend_frame(self, frame0, frame1, blend):
        root_trans_offset0 = frame0['root_trans_offset']
        root_trans_offset1 = frame1['root_trans_offset']
        blend_root_trans_offset = (1.0 - blend) * root_trans_offset0 + blend * root_trans_offset1
        pose_aa0 = frame0['pose_aa']
        pose_aa1 = frame1['pose_aa']
        blend_pose_aa = self._quat_slerp(pose_aa0, pose_aa1, blend)

        return blend_root_trans_offset, blend_pose_aa
    
    def _quat_slerp(self, q0, q1, t):
        r0 = sRot.from_quat(q0)
        r1 = sRot.from_quat(q1)
        return sRot.slerp(0, 1, [r0, r1])(t).as_quat()

    def get_full_frame(self):
        traj_idx = self.weighted_traj_idx_sample()
        t = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, t)

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        for _ in range(num_mini_batch):
            s, s_next = [], []
            traj_idxs = self.weighted_traj_idx_sample_batch(mini_batch_size)
            times = self.traj_time_sample_batch(traj_idxs)
            for traj_idx, t in zip(traj_idxs, times):
                s.append(self.get_full_frame_at_time(traj_idx, t))
                s_next.append(self.get_full_frame_at_time(traj_idx, t + self.trajectory_frame_durations[traj_idx]))
            yield s, s_next

    @property
    def num_motions(self):
        return len(self.motions)
