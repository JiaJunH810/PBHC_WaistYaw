
from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch

from pathlib import Path
from humanoidverse.utils.motion_lib.skeleton import SkeletonTree
from loguru import logger
import os
from enum import Enum
import glob
import joblib
import numpy as np
import torch
from easydict import EasyDict
from rich.progress import track
import random
from typing import Dict, Any, Optional

from isaac_utils.rotations import(
    quat_angle_axis,
    quat_inverse,
    quat_mul_norm,
    get_euler_xyz,
    normalize_angle,
    slerp,
    quat_to_exp_map,
    quat_to_angle_axis,
    quat_mul,
    quat_conjugate,
    calc_heading_quat_inv
)

class MotionlibMode(Enum):
    file = 1
    directory = 2

class FixHeightMode(Enum):
    no_fix = 0
    full_fix = 1
    ankle_fix = 2


def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)
    
def forbidden(fn):
    def wrapper(*args, **kwargs):
        raise RuntimeError("You are NOT ALLOWED to call it.")
    return wrapper

# 这边在对blend的计算稍微一些差异, 这里先不改成gmt的样子
def _calc_frame_blend(time, len, num_frames, dt):
    time = time.clone()
    phase = time / len
    phase = torch.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
    time[time < 0] = 0

    frame_idx0 = (phase * (num_frames - 1)).long()
    frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
    blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0) # clip blend to be within 0 and 1
    
    return frame_idx0, frame_idx1, blend

def _local_rotation_to_dof_smpl(local_rot):
    B, J, _ = local_rot.shape
    dof_pos = quat_to_exp_map(local_rot[:, 1:])
    return dof_pos.reshape(B, -1)

class MotionLibBase():
    
    def __init__(self, motion_lib_cfg, num_envs, device):
        
        def setup_constants(self, fix_height = FixHeightMode.full_fix, multi_thread = True):
            self.fix_height = fix_height
            self.multi_thread = multi_thread

            self._sampling_prob = torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions  # For use in sampling batches

        self.motion_config = motion_lib_cfg
        self._sim_fps = 1/self.motion_config.get("step_dt", 1/50)

        self.num_envs = num_envs
        self._device = device
        self.mesh_parsers = None
        self.has_action = False
        self.has_contact_mask = self.motion_config.has_contact_mask

        skeleton_file = Path(self.motion_config.asset.assetRoot) / self.motion_config.asset.assetFileName
        self.skeleton_tree = SkeletonTree.from_mjcf(skeleton_file)

        logger.info(f"Loaded skeleton from {skeleton_file}")
        logger.info(f"Loading motion data from {self.motion_config.motion_file}...")

        self.load_data(self.motion_config.motion_file)
        setup_constants(self, fix_height=FixHeightMode.no_fix, multi_thread=False)
    
    def load_data(self, motion_file, min_length=-1, im_eval=False):
        if os.path.isfile(motion_file):
            self.mode = MotionlibMode.file
            self._motion_data_load = [motion_file]
        else:
            self.mode = MotionlibMode.directory
            self._motion_data_load = glob.glob(os.path.join(motion_file, "*.pkl"))
        
        data_list = [joblib.load(file_path) for file_path in self._motion_data_load]

        self._motion_data_values = np.array([list(data.values()) for data in data_list])
        self._motion_data_keys = np.array([list(data.keys()) for data in data_list])

        self._num_unique_motions = len(self._motion_data_values)
        logger.info(f"Loaded {self._num_unique_motions} motions")
    
    def load_motions(self, random_sample=True, start_idx=0, max_len=-1, target_heading=None):
        assert target_heading is None, "Not Allowed to use target_heading!"

        
        total_len = 0.0
        self.num_bodies = len(self.skeleton_tree.node_names)
        num_envs = self.num_envs

        if random_sample:
            sample_idxes = torch.multinomial(self._sampling_prob, num_samples=num_envs, replacement=True).to(self._device)
        else:
            sample_idxes = torch.remainder(torch.arange(num_envs) + start_idx, self._num_unique_motions ).to(self._device)

        self._curr_motion_ids = sample_idxes
        
        logger.info(f"Loading {num_envs} motions...")
        logger.info(f"Sampling motion: {self._curr_motion_ids[:5]}, ....")

        res_all_motions = []
        for motion_data_value  in self._motion_data_values:
            res_all_motions.append(self.load_motion_with_skeleton(motion_data_value, self.fix_height, target_heading, max_len)[0])

        
        motions = []
        _motion_lengths = []
        _motion_fps = []
        _motion_dt = []
        _motion_num_frames = []
        _motion_bodies = []
        _motion_aa = []
        has_action = False
        _motion_actions = []
        _motion_contact_masks = []

        for f in track(range(len(res_all_motions)), description="Loading motions..."):
            motion_file_data, curr_motion = res_all_motions[f]
            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps
            num_frames = curr_motion.global_rotation.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            if "beta" in motion_file_data:
                pose_aa = motion_file_data['pose_aa'].reshape(-1, self.num_bodies * 3)
                pose_aa_tensor = torch.tensor(pose_aa, device=self._device, dtype=torch.float32)
                _motion_aa.append(pose_aa_tensor)
                _motion_bodies.append(torch.tensor(curr_motion.gender_beta, device=self._device, dtype=torch.float32))
            else:
                _motion_aa.append(torch.zeros((num_frames, self.num_bodies * 3)))
                _motion_bodies.append(torch.zeros(17))
            
            _motion_fps.append(motion_fps)
            _motion_dt.append(curr_dt)
            _motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            _motion_lengths.append(curr_len)
            if self.has_action:
                _motion_actions.append(torch.tensor(curr_motion.action, device=self._device, dtype=torch.float32))
            if self.has_contact_mask:
                _motion_contact_masks.append(curr_motion.contact_mask)
            
            del curr_motion
        
        self._motion_lengths = torch.tensor(_motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_fps = torch.tensor(_motion_fps, device=self._device, dtype=torch.float32)
        self._motion_bodies = torch.stack(_motion_bodies).to(self._device).type(torch.float32)

        self._motion_dt = torch.tensor(_motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(_motion_num_frames, device=self._device)
        
        self._motion_aa = torch.cat(_motion_aa, dim=0).to(self._device).type(torch.float32)
        if self.has_action:
            self._motion_actions = torch.cat(_motion_actions, dim=0).to(self._device).type(torch.float32)
        if self.has_contact_mask:
            self._motion_contact_masks = torch.cat(_motion_contact_masks, dim=0).to(self._device).type(torch.float32)


        self._num_motions = len(motions)

        self.gts = []
        self.grs = []
        self.lrs = []
        self.grvs = []
        self.gravs = []
        self.gavs = []
        self.gvs = []
        self.dvs = []

        for i in range(self._num_motions):
            self.gts.append(motions[i].global_translation.float().to(self._device))
            self.grs.append(motions[i].global_rotation.float().to(self._device))
            self.lrs.append(motions[i].local_rotation.float().to(self._device))
            self.grvs.append(motions[i].global_root_velocity.float().to(self._device))
            self.gravs.append(motions[i].global_root_angular_velocity.float().to(self._device))
            self.gavs.append(motions[i].global_angular_velocity.float().to(self._device))
            self.gvs.append(motions[i].global_velocity.float().to(self._device))
            self.dvs.append(motions[i].dof_vels.float().to(self._device))
        
        self.gts = torch.cat(self.gts, dim=0).to(self._device).type(torch.float32)
        self.grs = torch.cat(self.grs, dim=0).to(self._device).type(torch.float32)
        self.lrs = torch.cat(self.lrs, dim=0).to(self._device).type(torch.float32)
        self.grvs = torch.cat(self.grvs, dim=0).to(self._device).type(torch.float32)
        self.gravs = torch.cat(self.gravs, dim=0).to(self._device).type(torch.float32)
        self.gavs = torch.cat(self.gavs, dim=0).to(self._device).type(torch.float32)
        self.gvs = torch.cat(self.gvs, dim=0).to(self._device).type(torch.float32)
        self.dvs = torch.cat(self.dvs, dim=0).to(self._device).type(torch.float32)


        if "global_translation_extend" in motions[0].__dict__:
            self.gts_t = []
            self.grs_t = []
            self.gvs_t = []
            self.gavs_t = []
            for i in range(self._num_motions):
                self.gts_t.append(motions[i].global_translation_extend.float().to(self._device))
                self.grs_t.append(motions[i].global_rotation_extend.float().to(self._device))
                self.gvs_t.append(motions[i].global_velocity_extend.float().to(self._device))
                self.gavs_t.append(motions[i].global_angular_velocity_extend.float().to(self._device))
            
            self.gts_t = torch.cat(self.gts_t, dim=0).to(self._device).type(torch.float32)
            self.grs_t = torch.cat(self.grs_t, dim=0).to(self._device).type(torch.float32)
            self.gvs_t = torch.cat(self.gvs_t, dim=0).to(self._device).type(torch.float32)
            self.gavs_t = torch.cat(self.gavs_t, dim=0).to(self._device).type(torch.float32)

        if "dof_pos" in motions[0].__dict__:
            self.dof_pos = []
            for i in range(self._num_motions):
                self.dof_pos.append(motions[i].dof_pos.float().to(self._device))
            self.dof_pos = torch.cat(self.dof_pos, dim=0).to(self._device).type(torch.float32)
        
        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)
        self.motion_ids = self._curr_motion_ids

        total_len = self.get_total_length()
        logger.info(f"Loaded {self._num_motions:d} motions with a total length of {total_len:.3f}s and {self.gts[0].shape[0]} frames.")
        
        


    def load_motion_with_skeleton(self,
                                  motion_data_list: np.ndarray,
                                  fix_height,
                                  target_heading,
                                  max_len):
        @forbidden
        def fix_trans_height(self, pose_aa, trans, fix_height_mode):
            if fix_height_mode == FixHeightMode.no_fix:
                return trans, 0
            with torch.no_grad():
                mesh_obj = self.mesh_parsers.mesh_fk(pose_aa[None, :1], trans[None, :1])
                height_diff = np.asarray(mesh_obj.vertices)[..., 2].min()
                trans[..., 2] -= height_diff
                
                return trans, height_diff
            
            
        # loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
        res = {}
        
        for f in track(range(len(motion_data_list)), description="Loading motions..."):
            curr_file:Dict[str, Any] = motion_data_list[f]
            if not isinstance(curr_file, dict) and os.path.isfile(curr_file):
                forbidden(lambda :0)()
                key = motion_data_list[f].split("/")[-1].split(".")[0]
                curr_file = joblib.load(curr_file)[key]

            if False: 
            # if True: 
                print("DEBUG: !!!! MotionLibBase: rebase root_trans_offset & root_rot_offset")
                curr_file['root_trans_offset'][:] = np.array([0, 0, 0.8], dtype=np.float64)
                target_heading = np.array([0, 0, 0, 1.0])
                # curr_file['root_rot']= rebase_yaw(curr_file['root_rot'])
                # breakpoint()
            
            seq_len = curr_file['root_trans_offset'].shape[0]
            if max_len == -1 or seq_len < max_len:
                start, end = 0, seq_len
            else:
                start = random.randint(0, seq_len - max_len)
                end = start + max_len

            trans = to_torch(curr_file['root_trans_offset']).clone()[start:end]
            pose_aa = to_torch(curr_file['pose_aa'][start:end]).clone()
            # import ipdb; ipdb.set_trace()
            if "action" in curr_file.keys():
                self.has_action = True
            if "contact_mask" in curr_file.keys():
                contact_shape = curr_file['contact_mask'].shape
                assert len(contact_shape) ==2 and contact_shape[0] == seq_len
                self._contact_size = contact_shape[1]
                if contact_shape[1] == 2:
                    self.has_contact_mask = "point"
                else:
                    raise ValueError(f"Contact mask shape {contact_shape} is not supported")
            
            dt = 1/curr_file['fps']

            B, J, N = pose_aa.shape

            if not target_heading is None:
                from scipy.spatial.transform import Rotation as sRot
                # forbidden(lambda :0)()
                start_root_rot = sRot.from_rotvec(pose_aa[0, 0])
                heading_inv_rot = sRot.from_quat(calc_heading_quat_inv(torch.from_numpy(start_root_rot.as_quat()[None, ]),True))
                heading_delta = sRot.from_quat(target_heading) * heading_inv_rot 
                pose_aa[:, 0] = torch.tensor((heading_delta * sRot.from_rotvec(pose_aa[:, 0])).as_rotvec())

                trans = torch.matmul(trans.to(torch.float64), torch.from_numpy(heading_delta.as_matrix().squeeze().T))

            if self.mesh_parsers is None:
                logger.error("No mesh parser found")
            # trans, trans_fix = fix_trans_height(self, pose_aa, trans, mesh_parsers, fix_height_mode = fix_height)
            curr_motion = self.mesh_parsers.fk_batch(pose_aa[None, ], trans[None, ], return_full= True, dt = dt)
            curr_motion = EasyDict({k: v.squeeze() if torch.is_tensor(v) else v for k, v in curr_motion.items() })
            # add "action" to curr_motion
            if self.has_action:
                curr_motion.action = to_torch(curr_file['action']).clone()[start:end]
            if self.has_contact_mask:
                curr_motion.contact_mask = to_torch(curr_file['contact_mask']).clone()[start:end]
                
            res[f] = (curr_file, curr_motion)
        return res

    ################################### GET ###################################

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]

        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = _calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = (frame_idx0 + self.length_starts[motion_ids]).long()
        f1l = (frame_idx1 + self.length_starts[motion_ids]).long()

        motion_res = self.get_frame_motion_state(f0l, f1l, blend, motion_ids.long(), offset)
        
        return motion_res
        
    def get_frame_motion_state(self, f0l, f1l, blend, motion_ids, offset=None):
        if "dof_pos" in self.__dict__:
            local_rot0 = self.dof_pos[f0l]
            local_rot1 = self.dof_pos[f1l]
        else:
            local_rot0 = self.lrs[f0l]
            local_rot1 = self.lrs[f1l]
            
        body_vel0 = self.gvs[f0l]
        body_vel1 = self.gvs[f1l]

        body_ang_vel0 = self.gavs[f0l]
        body_ang_vel1 = self.gavs[f1l]

        # breakpoint()
        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        vals = [local_rot0, local_rot1, body_vel0, body_vel1, body_ang_vel0, body_ang_vel1, rg_pos0, rg_pos1, dof_vel0, dof_vel1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        if offset is None:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset
        else:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1 + offset[..., None, :]  # ZL: apply offset

        body_vel = (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1

        if "dof_pos" in self.__dict__: # Robot Joints
            dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1
            dof_pos = (1.0 - blend) * local_rot0 + blend * local_rot1
        else:
            dof_vel = (1.0 - blend_exp) * dof_vel0 + blend_exp * dof_vel1
            local_rot = slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
            dof_pos = _local_rotation_to_dof_smpl(local_rot)

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]
        rb_rot = slerp(rb_rot0, rb_rot1, blend_exp)
        return_dict = {}
        
        if "gts_t" in self.__dict__:
            rg_pos_t0 = self.gts_t[f0l]
            rg_pos_t1 = self.gts_t[f1l]
            
            rg_rot_t0 = self.grs_t[f0l]
            rg_rot_t1 = self.grs_t[f1l]
            
            body_vel_t0 = self.gvs_t[f0l]
            body_vel_t1 = self.gvs_t[f1l]
            
            body_ang_vel_t0 = self.gavs_t[f0l]
            body_ang_vel_t1 = self.gavs_t[f1l]
            if offset is None:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1  
            else:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1 + offset[..., None, :]
            rg_rot_t = slerp(rg_rot_t0, rg_rot_t1, blend_exp)
            body_vel_t = (1.0 - blend_exp) * body_vel_t0 + blend_exp * body_vel_t1
            body_ang_vel_t = (1.0 - blend_exp) * body_ang_vel_t0 + blend_exp * body_ang_vel_t1
        else:
            rg_pos_t = rg_pos
            rg_rot_t = rb_rot
            body_vel_t = body_vel
            body_ang_vel_t = body_ang_vel
        

        if self.has_contact_mask:
            contact0, contact1 = self._motion_contact_masks[f0l], self._motion_contact_masks[f1l]
            contact = (1.0 - blend) * contact0 + blend * contact1
            
            return_dict["contact_mask"] = contact
            

        return_dict.update({
            "root_pos": rg_pos[..., 0, :].clone(),
            "root_rot": rb_rot[..., 0, :].clone(),
            "dof_pos": dof_pos.clone(),
            "root_vel": body_vel[..., 0, :].clone(),
            "root_ang_vel": body_ang_vel[..., 0, :].clone(),
            "dof_vel": dof_vel.view(dof_vel.shape[0], -1),
            "motion_aa": self._motion_aa[f0l],
            "motion_bodies": self._motion_bodies[motion_ids],
            "rg_pos": rg_pos,
            "rb_rot": rb_rot,
            "body_vel": body_vel,
            "body_ang_vel": body_ang_vel,
            "rg_pos_t": rg_pos_t,
            "rg_rot_t": rg_rot_t,
            "body_vel_t": body_vel_t,
            "body_ang_vel_t": body_ang_vel_t,
        })
        return return_dict

    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion_length(self, motion_ids=None):
        if motion_ids is None:
            return self._motion_lengths
        else:
            return self._motion_lengths[motion_ids]
    
    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if (truncate_time is not None):
            assert (truncate_time >= 0.0)
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time.to(self._device)

class MotionLibRobotJJH(MotionLibBase):
    def __init__(self, motion_lib_cfg, num_envs, device):
        super().__init__(motion_lib_cfg=motion_lib_cfg, num_envs=num_envs, device=device)
        self.mesh_parsers = Humanoid_Batch(motion_lib_cfg)