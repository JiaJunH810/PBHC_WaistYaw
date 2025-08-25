import argparse, os, time
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
import torch
import onnxruntime as ort
from motion_lib import MotionLib
from config import Config
import numpy as np

def quatToEuler(quat):
    eulerVec = np.zeros(3)
    qw = quat[0] 
    qx = quat[1] 
    qy = quat[2]
    qz = quat[3]
    # roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    eulerVec[0] = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        eulerVec[1] = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
    else:
        eulerVec[1] = np.arcsin(sinp)

    # yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    eulerVec[2] = np.arctan2(siny_cosp, cosy_cosp)
    
    return eulerVec


class HumanoidEnv:
    def __init__(self, policy_path, motion_file, model_path, device):
        self.config = Config(device)
        self.config.motion_file = motion_file
        self.device = device

        self.sim_duration = 60.0
        self.sim_dt = 0.005
        self.sim_decimation = 4
        self.control_dt = self.sim_dt * self.sim_decimation

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.cam.distance = 5.0

        self.last_action = torch.zeros(self.config.num_actions).to(self.device)

        # prepare tar_obs_mimic
        tar_obs_mimic_step = 5
        tar_obs_mimic_counter = 20
        self.tar_obs_steps = torch.tensor([i * tar_obs_mimic_step for i in range(tar_obs_mimic_counter)], dtype=torch.float32).to(self.device)
        
        # prepare history
        history_len = 20
        self.n_proprio = 3 + 2 + 3 * self.config.num_actions
        self.proprio_history_buf = deque(maxlen=history_len)
        for _ in range(history_len):
            self.proprio_history_buf.append(torch.zeros(self.n_proprio).to(self.device))

        # prepare motion lib
        self._motion_lib = MotionLib(self.config, device)

        print("Loading onnx: ", policy_path)
        self.session = ort.InferenceSession(policy_path)
        self.input_name = self.session.get_inputs()[0].name
    
    def data_from_mujoco(self):
        dof_pos = self.data.qpos[7:]
        dof_vel = self.data.qvel[6:]
        quat = self.data.qpos[3:7]
        base_ang_vel = self.data.qvel[3:6]
        return dof_pos, dof_vel, quat, base_ang_vel

    def _get_mimic_obs(self, curr_timestep):
        num_steps = len(self.tar_obs_steps)
        motion_times = torch.tensor(curr_timestep * self.control_dt, device=self.device).unsqueeze(-1)
        obs_motion_time = self.tar_obs_steps * self.control_dt + motion_times
        
        root_pos, root_rot, root_vel, root_ang_vel, dof_pos = self._motion_lib._calc_motion_frame(obs_motion_time)

        roll, pitch, yaw = self.config.euler_from_quaternion(root_rot)
        roll = roll.reshape(1, num_steps, 1)
        pitch = pitch.reshape(1, num_steps, 1)
        yaw = yaw.reshape(1, num_steps, 1)

        root_vel = self.config.quat_rotate_inverse(root_rot, root_vel)
        root_ang_vel = self.config.quat_rotate_inverse(root_rot, root_ang_vel)

        root_pos = root_pos.reshape(1, num_steps, 3)
        root_vel = root_vel.reshape(1, num_steps, 3)
        root_ang_vel = root_ang_vel.reshape(1, num_steps, 3)
        dof_pos = dof_pos.reshape(1, num_steps, -1)

        mimic_obs_buf = torch.cat((
                root_pos[..., 2:3],
                roll, pitch,
                root_vel,
                root_ang_vel[..., 2:3],
                dof_pos,
            ), dim=-1)
        mimic_obs_buf = mimic_obs_buf.reshape(1, -1)

        return mimic_obs_buf.detach().squeeze().to(self.device)
        

    def run(self):
        for i in tqdm(range(int(self.sim_duration / self.sim_dt)), desc="Running simulation..."):
            dof_pos, dof_vel, quat, ang_vel = self.data_from_mujoco()

            dof_pos = torch.from_numpy(dof_pos).float().to(self.device)
            dof_vel = torch.from_numpy(dof_vel).float().to(self.device)

            if i % self.sim_decimation == 0:
                curr_timestep = i // self.sim_decimation
                mimic_obs = self._get_mimic_obs(curr_timestep)

                ang_vel = torch.from_numpy(ang_vel).float().to(self.device)

                rpy = quatToEuler(quat)
                rpy = torch.from_numpy(rpy).float().to(self.device)
                
                obs_prop = torch.cat([
                    self.last_action,
                    ang_vel * self.config.obs_scales['base_ang_vel'],
                    rpy[[1, 0]],
                    dof_pos - self.config.default_dof_pos,
                    dof_vel * self.config.obs_scales['dof_vel'],
                ]).to(self.device)
                
                assert obs_prop.shape[0] == self.n_proprio, f"Expected {self.n_proprio} but got {obs_prop.shape[0]}"
                obs_hist = torch.cat(list(self.proprio_history_buf)).to(self.device)

                obs_tensor = torch.cat([obs_prop, mimic_obs, obs_hist])
                
                obs_numpy = obs_tensor.unsqueeze(0).cpu().numpy()
                actions = np.squeeze(self.session.run(None, {self.input_name: obs_numpy})[0])
                
                actions = torch.from_numpy(actions).to(self.device)
                self.last_action = actions.clone()

                target_pos = self.config.default_dof_pos

                self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
                self.viewer.render()
                
                self.proprio_history_buf.appendleft(obs_prop)   # append to front
            
            torque = (target_pos - dof_pos) * self.config.kp - dof_vel * self.config.kd
            torque = torque.cpu().numpy()
            torque = np.clip(torque, -self.config.torque_limits, self.config.torque_limits)

            self.data.ctrl = torque

            mujoco.mj_step(self.model, self.data)
        
        self.viewer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--motion', type=str)
    args = parser.parse_args()

    assert args.checkpoint != None, ValueError("Please input --checkpoint")
    assert args.motion != None, ValueError("Please input --motion")
    assert os.path.exists(args.motion), ValueError("Please input ture motion path")
    
    policy_path = args.checkpoint
    motion_file = args.motion
    model_path = "description/robots/g1/g1_23dof.xml"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = HumanoidEnv(policy_path, motion_file, model_path, device)
    env.run()