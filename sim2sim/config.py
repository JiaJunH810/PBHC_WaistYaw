import torch
import numpy as np

class Config:
    def __init__(self, device):
        self.dof_names = [
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
            'waist_yaw_joint',
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint',
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint'
        ]
        self.stiffness = {
            "hip_yaw": 100,
            "hip_roll": 100,
            "hip_pitch": 100,
            "knee": 200,
            "ankle_pitch": 20,
            "ankle_roll": 20,
            "waist_yaw": 400,
            "shoulder_pitch": 90,
            "shoulder_roll": 60,
            "shoulder_yaw": 20,
            "elbow": 60,
            "wrist_roll": 60
        }
        self.damping = {
            "hip_yaw": 2.5,
            "hip_roll": 2.5,
            "hip_pitch": 2.5,
            "knee": 5.0,
            "ankle_pitch": 0.2,
            "ankle_roll": 0.1,
            "waist_yaw": 5.0,
            "shoulder_pitch": 2.0,
            "shoulder_roll": 1.0,
            "shoulder_yaw": 0.4,
            "elbow": 1.0,
            "wrist_roll": 1.0
        }
        self.obs_scales = {
            "base_lin_vel": 2.0,
            "base_ang_vel": 0.25,
            "dof_vel": 0.05,
        }
        self.default_dof_pos = torch.tensor([
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
            -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
            0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0
        ]).to(device)
        self.torque_limits = np.array([
            88.0, 88.0, 88.0, 139.0, 50.0, 50.0, 
            88.0, 88.0, 88.0, 139.0, 50.0, 50.0, 
            88.0,
            25.0, 25.0, 25.0, 25.0, 25.0, 
            25.0, 25.0, 25.0, 25.0, 25.0
        ])

        self.num_actions = len(self.dof_names)
        self.action_scale = 0.25

        # prepare kp, kd
        self.kp = torch.zeros(self.num_actions).to(device)
        self.kd = torch.zeros(self.num_actions).to(device)
        for i, name in enumerate(self.dof_names):
            for key in self.stiffness:
                if key in name:
                    self.kp[i] = self.stiffness[key]
                    self.kd[i] = self.damping[key]
        

        # prepare Humanoid_Batch
        self.assetRoot = "description/robots/g1/"
        self.assetFileName = "g1_23dof.xml"

        self.extend_config = [
            {"joint_name": "left_hand_link", "parent_name": "left_elbow_link", "pos": [0.25, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0]},
            {"joint_name": "right_hand_link", "parent_name": "right_elbow_link", "pos": [0.25, 0.0, 0.0], "rot": [1.0, 0.0, 0.0, 0.0]},
            {"joint_name": "head_link", "parent_name": "torso_link", "pos": [0.0, 0.0, 0.42], "rot": [1.0, 0.0, 0.0, 0.0]}
        ]
    
    def euler_from_quaternion(self, quat_angle):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = torch.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = torch.clip(t2, -1, 1)
        pitch_y = torch.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = torch.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

    def quat_rotate_inverse(self, q, v):
        shape = q.shape
        q_w = q[:, -1]
        q_vec = q[:, :3]
        a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * \
            torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
                shape[0], 3, 1)).squeeze(-1) * 2.0
        return a - b + c