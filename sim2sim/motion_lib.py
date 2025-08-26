from torch_humanoid_batch import Humanoid_Batch
from pathlib import Path
from humanoidverse.utils.motion_lib.skeleton import SkeletonTree
import joblib
import torch
from easydict import EasyDict
from isaac_utils.rotations import(
    slerp,
    quat_to_exp_map,
    calc_heading_quat_inv
)
def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)
    
class MotionLib:
    def __init__(self, config, device):
        self.motion_config = config
        
        self._device = device
        skeleton_file = Path(self.motion_config.assetRoot) / self.motion_config.assetFileName
        self.skeleton_tree = SkeletonTree.from_mjcf(skeleton_file)
        self.mesh_parsers = Humanoid_Batch(config)
        print(f"Loaded skeleton from {skeleton_file}")
        print(f"Loading motion data from {self.motion_config.motion_file}...")

        self.load_data(self.motion_config.motion_file)

        
    

    def load_data(self, motion_file):
        self.num_bodies = len(self.skeleton_tree.node_names)

        motion = joblib.load(motion_file)
        for key in list(motion.keys()):
            motion = motion[key]
        
        motion_file_data, curr_motion = self.load_motion_with_skeleton(motion)
        
        # process data
        motion_fps = curr_motion.fps
        self._motion_dt = 1.0 / motion_fps
        self.num_frames = torch.tensor(curr_motion.global_rotation.shape[0])
        self._motion_lengths = torch.tensor(1.0 / motion_fps * (self.num_frames - 1)).to(self._device)

        self.root_pos = curr_motion.global_translation_extend[..., 0, :].to(self._device)
        self.root_rot = curr_motion.global_rotation_extend[..., 0, :].to(self._device)
        self.root_vel = curr_motion.global_velocity_extend[..., 0, :].to(self._device)
        self.root_ang_vel = curr_motion.global_angular_velocity_extend[..., 0, :].to(self._device)
        self.dof_pos = curr_motion.dof_pos.to(self._device)
        

    
    def load_motion_with_skeleton(self, motion):
        seq_len = motion['root_trans_offset'].shape[0]
        start, end = 0, seq_len
        
        trans = to_torch(motion['root_trans_offset']).clone()[start:end]
        pose_aa = to_torch(motion['pose_aa'][start:end]).clone()

        dt = 1 / motion['fps']

        curr_motion = self.mesh_parsers.fk_batch(pose_aa[None, ], trans[None, ], return_full=True, dt=dt)
        curr_motion = EasyDict({k: v.squeeze() if torch.is_tensor(v) else v for k, v in curr_motion.items() })

        return (motion, curr_motion)

    
    def _calc_frame_blend(self, motion_times, motion_len, num_frames, dt):
        time = motion_times.clone()
        phase = time / motion_len
        phase = torch.clip(phase, 0., 1.)
        time[time < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0)
        return frame_idx0, frame_idx1, blend

    def _calc_motion_frame(self, motion_times):
        f0, f1, blend = self._calc_frame_blend(motion_times, self._motion_lengths,
                                                               self.num_frames, self._motion_dt)
        blend = blend.unsqueeze(-1)
        root_pos_t0 = self.root_pos[f0]
        root_pos_t1 = self.root_pos[f1]
        root_pos = (1.0 - blend) * root_pos_t0 + blend * root_pos_t1
        
        root_rot_t0 = self.root_rot[f0]
        root_rot_t1 = self.root_rot[f1]
        root_rot = slerp(root_rot_t0, root_rot_t1, blend)
        
        root_vel_t0 = self.root_vel[f0]
        root_vel_t1 = self.root_vel[f1]
        root_vel = (1.0 - blend) * root_vel_t0 + blend * root_vel_t1

        root_ang_vel_t0 = self.root_ang_vel[f0]
        root_ang_vel_t1 = self.root_ang_vel[f1]
        root_ang_vel = (1.0 - blend) * root_ang_vel_t0 + blend * root_ang_vel_t1

        dof_pos_t0 = self.dof_pos[f0]
        dof_pos_t1 = self.dof_pos[f1]
        dof_pos = (1.0 - blend) * dof_pos_t0 + blend * dof_pos_t1

        return root_pos, root_rot, root_vel, root_ang_vel, dof_pos