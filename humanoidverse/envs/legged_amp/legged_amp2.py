from time import time
import numpy as np
import matplotlib.pyplot as plt
import os

from humanoidverse.utils.torch_utils import *
# from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from isaac_utils.rotations import get_euler_xyz_in_tensor
from isaac_utils.rotations import quat_apply_yaw, wrap_to_pi
from humanoidverse.envs.base_task.base_task import BaseTask
from humanoidverse.utils.noise_tool import noise_process_dict

from humanoidverse.envs.env_utils.history_handler import HistoryHandler
from termcolor import colored
from humanoidverse.utils.helpers import parse_observation
from humanoidverse.envs.env_utils.visualization import Point

from loguru import logger
import copy

from isaac_utils.customize import (
    batch_local_to_world_com
)

# 添加AMP加载器
from humanoidverse.utils.motion_lib.motion_loader import AMPLoader


class LeggedAMP(BaseTask):
    def __init__(self, config, device):
        self.init_done = False
        super().__init__(config, device)
        self.num_envs = self.config.num_envs
        self._domain_rand_config()
        self._prepare_reward_function()
        self.history_handler = HistoryHandler(self.num_envs, config.obs.obs_auxiliary, config.obs.obs_dims, device)
        self.is_evaluating = False

        # 加载运动序列
        self.amp_loader = AMPLoader(config=config, num_envs=self.num_envs)

        # 初始化运动跟踪相关的变量
        self._init_motion_tracking()

        self.init_done = True

    def _init_motion_tracking(self):
        """初始化运动跟踪相关的变量"""
        # 运动ID和时间
        self.motion_ids = torch.arange(self.num_envs).to(self.device)
        self.motion_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        
        # 参考状态缓冲区
        self.ref_dof_pos = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.ref_dof_vel = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.ref_root_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.ref_root_rot = torch.zeros(self.num_envs, 4, device=self.device)
        self.ref_root_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.ref_root_ang_vel = torch.zeros(self.num_envs, 3, device=self.device)
        
        # 差异缓冲区
        self.dif_dof_pos = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.dif_dof_vel = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.dif_root_pos = torch.zeros(self.num_envs, 3, device=self.device)
        
        # 运动长度
        self.motion_lengths = self.amp_loader.trajectory_lens[self.motion_ids]
        
        # 如果是评估模式，不随机化运动开始时间
        if self.is_evaluating and not self.config.enforce_randomize_motion_start_eval:
            self.motion_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        else:
            self.motion_times = self.amp_loader.traj_time_sample_batch(self.motion_ids)


    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        super()._init_buffers()

        self.base_quat = self.simulator.base_quat # XYZW
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)

        # initialize some data used later on
        self._init_counters()
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions_after_delay = torch.zeros(self.num_envs, self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.dim_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_pos = torch.zeros_like(self.simulator.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.simulator.dof_vel)
        self.last_root_vel = torch.zeros_like(self.simulator.robot_root_states[:, 7:13])
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        
        self.contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)
        self.contacts_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)  
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)
        
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.config.robot.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.config.robot.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.config.robot.control.stiffness[dof_name]
                    self.d_gains[i] = self.config.robot.control.damping[dof_name]
                    found = True
                    logger.debug(f"PD gain of joint {name} were defined, setting them to {self.p_gains[i]} and {self.d_gains[i]}")
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.config.robot.control.control_type in ["P", "V"]:
                    logger.warning(f"PD gain of joint {name} were not defined, setting them to zero")
                    raise ValueError(f"PD gain of joint {name} were not defined. Should be defined in the yaml file.")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self._init_domain_rand_buffers()

        # for reward penalty curriculum
        self.average_episode_length = 0. # num_compute_average_epl last termination episode length
        self.last_episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.num_compute_average_epl = self.config.rewards.num_compute_average_epl

        self.need_to_refresh_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        self.add_noise_currculum = self.config.obs.add_noise_currculum
        self.current_noise_curriculum_value = self.config.obs.noise_initial_value
        
        if 'noise_process' in self.config.obs and self.config.obs.noise_process.enable:
            self.use_noise_process = True
            self.noise_process = noise_process_dict[self.config.obs.noise_process.type](shape=(self.num_envs, 3+3), 
                                                                                        tensor_type="torch_"+str(self.device),
                                                                                        dt = self.dt,
                                                                                        **self.config.obs.noise_process.kwargs)
        else:
            self.use_noise_process = False
        
        

    def _domain_rand_config(self):
        if self.config.domain_rand.push_robots:
            self.push_interval_s = torch.randint(self.config.domain_rand.push_interval_s[0], self.config.domain_rand.push_interval_s[1], (self.num_envs,), device=self.device)

            # 判断推机器人是否进行课程学习
            if 'push_robot_curriculum' in self.config.domain_rand and self.config.domain_rand.push_robot_curriculum:
                self.push_robot_vel_xy = self.config.domain_rand.max_push_vel_xy

    def _init_counters(self):
        self.common_step_counter = 0
        self.push_robot_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)
        self.push_robot_plot_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)
        self.command_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)
        self.reinit_epis_rand_counter = 0

    def _update_counters_each_step(self):
        self.common_step_counter  +=1
        self.push_robot_counter[:] += 1
        self.push_robot_plot_counter[:] += 1
        self.command_counter[:] += 1

    def _init_domain_rand_buffers(self):
        ######################################### DR related tensors #########################################
        if self.config.domain_rand.randomize_ctrl_delay:
            self.action_queue = torch.zeros(self.num_envs, self.config.domain_rand.ctrl_delay_step_range[1]+1, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.action_delay_idx = torch.randint(self.config.domain_rand.ctrl_delay_step_range[0], 
                                                self.config.domain_rand.ctrl_delay_step_range[1]+1, (self.num_envs,), device=self.device, requires_grad=False)

        # self._link_mass_scale = torch.ones(self.num_envs, len(self.config.robot.randomize_link_body_names), dtype=torch.float, device=self.device, requires_grad=False)
        self._kp_scale = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self._kd_scale = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self._rfi_lim_scale = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self._rao_scale = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_robot_vel_buf = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.record_push_robot_vel_buf = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

        self.feet_air_max_height = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        logger.info(colored(f"{self.config.rewards.set_reward} set reward on {self.config.rewards.set_reward_date}", "green"))
        
        self.reward_scales = self.config.rewards.reward_scales
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            logger.info(f"Scale: {key} = {self.reward_scales[key]}")
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt

        # 添加运动跟踪奖励函数
        if hasattr(self.config.rewards, 'tracking_reward_scales'):
            self.tracking_reward_scales = self.config.rewards.tracking_reward_scales
            for key in list(self.tracking_reward_scales.keys()):
                logger.info(f"Tracking Scale: {key} = {self.tracking_reward_scales[key]}")
                scale = self.tracking_reward_scales[key]
                if scale==0:
                    self.tracking_reward_scales.pop(key) 
                else:
                    self.tracking_reward_scales[key] *= self.dt
        else:
            self.tracking_reward_scales = {}

        self.use_reward_penalty_curriculum = self.config.rewards.reward_penalty_curriculum
        if self.use_reward_penalty_curriculum:
            self.reward_penalty_scale = self.config.rewards.reward_initial_penalty_scale

        logger.info(colored(f"Use Reward Penalty: {self.use_reward_penalty_curriculum}", "green"))
        if self.use_reward_penalty_curriculum:
            logger.info(f"Penalty Reward Names: {self.config.rewards.reward_penalty_reward_names}")
            logger.info(f"Penalty Reward Initial Scale: {self.config.rewards.reward_initial_penalty_scale}")
        
        self.use_reward_limits_dof_pos_curriculum = self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_curriculum
        self.use_reward_limits_dof_vel_curriculum = self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_curriculum
        self.use_reward_limits_torque_curriculum = self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_curriculum

        if self.use_reward_limits_dof_pos_curriculum:
            logger.info(f"Use Reward Limits DOF Curriculum: {self.use_reward_limits_dof_pos_curriculum}")
            logger.info(f"Reward Limits DOF Curriculum Initial Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_initial_limit}")
            logger.info(f"Reward Limits DOF Curriculum Max Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_max_limit}")
            logger.info(f"Reward Limits DOF Curriculum Min Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_min_limit}")
            self.soft_dof_pos_curriculum_value = self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_pos_initial_limit
        
        if self.use_reward_limits_dof_vel_curriculum:
            logger.info(f"Use Reward Limits DOF Vel Curriculum: {self.use_reward_limits_dof_vel_curriculum}")
            logger.info(f"Reward Limits DOF Vel Curriculum Initial Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_initial_limit}")
            logger.info(f"Reward Limits DOF Vel Curriculum Max Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_max_limit}")
            logger.info(f"Reward Limits DOF Vel Curriculum Min Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_min_limit}")
            self.soft_dof_vel_curriculum_value = self.config.rewards.reward_limit.reward_limits_curriculum.soft_dof_vel_initial_limit
        
        if self.use_reward_limits_torque_curriculum:
            logger.info(f"Use Reward Limits Torque Curriculum: {self.use_reward_limits_torque_curriculum}")
            logger.info(f"Reward Limits Torque Curriculum Initial Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_initial_limit}")
            logger.info(f"Reward Limits Torque Curriculum Max Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_max_limit}")
            logger.info(f"Reward Limits Torque Curriculum Min Limit: {self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_min_limit}")
            self.soft_torque_curriculum_value = self.config.rewards.reward_limit.reward_limits_curriculum.soft_torque_initial_limit

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))
        
        # 添加运动跟踪奖励函数
        for name, scale in self.tracking_reward_scales.items():
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                            for name in list(self.reward_scales.keys()) + list(self.tracking_reward_scales.keys())}

        if self.config.use_vec_reward:
            # reward_fn_state.reward_functions = [reward_fn_state.reward_functions[0]]
            num_rew_fn = len(self.reward_functions)+1
            self.rew_buf = torch.zeros(self.num_envs, num_rew_fn, dtype=torch.float, device=self.device, requires_grad=False)

    def set_is_evaluating(self):
        logger.info("Setting Env is evaluating")
        self.is_evaluating = True
    
    def step(self, actor_state):
        """ Apply actions, simulate, call self.post_physics_step()
        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        
        actions = actor_state["actions"]
        # actions *= 0.0
        self._pre_physics_step(actions)
        self._physics_step()
        self._post_physics_step()

        # if self.episode_length_buf[0] == 1:
        #     import ipdb; ipdb.set_trace()

        # left_force = self.simulator.gym.get_actor_force_sensor(self.simulator.envs[0],self.simulator.robot_handles[0], 0).get_forces()
        # right_force = self.simulator.gym.get_actor_force_sensor(self.simulator.envs[0],self.simulator.robot_handles[0], 1).get_forces()
        # print("left_force", left_force.force.x, left_force.force.y, left_force.force.z)
        # print("right_force", right_force.force.x, right_force.force.y, right_force.force.z)
        # print("left torque", left_force.torque.x, left_force.torque.y, left_force.torque.z)
        # print("right torque", right_force.torque.x, right_force.torque.y, right_force.torque.z)
        # print("force_sensor", self.simulator.force_sensor)
        # print("Foot force: ", torch.norm(self.simulator.contact_forces[:, self.feet_indices, :], dim=-1),
        #         '\t|', 'Stumble:', torch.any(torch.norm(self.simulator.contact_forces[:, self.feet_indices, :2], dim=2) >\
        #      5 *torch.abs(self.simulator.contact_forces[:, self.feet_indices, 2]), dim=1))
        # breakpoint()
        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras

    def _pre_physics_step(self, actions):
        clip_action_limit = self.config.robot.control.action_clip_value
        self.actions = torch.clip(actions, -clip_action_limit, clip_action_limit).to(self.device)

        # action noise
        # self.actions += torch.randn_like(self.actions) * 0.01
        # self.actions *= 1 + torch.randn_like(self.actions) * 0.01

        self.log_dict["action_clip_frac"] = (
                self.actions.abs() == clip_action_limit
            ).sum() / self.actions.numel()

        if self.config.domain_rand.randomize_ctrl_delay:
            self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
            self.action_queue[:, 0] = self.actions.clone()
            self.actions_after_delay = self.action_queue[torch.arange(self.num_envs), self.action_delay_idx].clone()
        else:
            self.actions_after_delay = self.actions.clone()


    def _physics_step(self):
        self.render()
        for _ in range(self.config.simulator.config.sim.control_decimation):
            self._apply_force_in_physics_step()
            self.simulator.simulate_at_each_physics_step()

    def _apply_force_in_physics_step(self):
        self.torques = self._compute_torques(self.actions_after_delay).view(self.torques.shape)
        self.simulator.apply_torques_at_dof(self.torques)

    def _post_physics_step(self):
        self._refresh_sim_tensors()
        self.episode_length_buf += 1
        # update counters
        self._update_counters_each_step()
        self.last_episode_length_buf = self.episode_length_buf.clone()

        self._pre_compute_observations_callback()
        
        self._update_tasks_callback()
        # compute observations, rewards, resets, ...
        self._check_termination()
        self._compute_reward()
        # check terminations
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_envs_idx(env_ids)

        # set envs
        refresh_env_ids = self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()
        if len(refresh_env_ids) > 0:
            self.simulator.set_actor_root_state_tensor(refresh_env_ids, self.simulator.all_root_states)
            self.simulator.set_dof_state_tensor(refresh_env_ids, self.simulator.dof_state)
            self.need_to_refresh_envs[refresh_env_ids] = False

        self._compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        
        self._post_compute_observations_callback()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.config.normalization.clip_observations
        for obs_key, obs_val in self.obs_buf_dict.items():
            self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_obs, clip_obs)

        for key in self.history_handler.history.keys():
            self.history_handler.add(key, self.hist_obs_dict[key])

        self.extras["to_log"] = self.log_dict
        if self.viewer:
            self._setup_simulator_control()
            self._setup_simulator_next_task()
            if self.debug_viz:
                self._draw_debug_vis()
    
    def _setup_simulator_next_task(self):
        pass

    def _setup_simulator_control(self):
        pass

    def _pre_compute_observations_callback(self):
        # prepare quantities
        self.base_quat[:] = self.simulator.base_quat[:]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.simulator.robot_root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.contacts = ( self.simulator.contact_forces[:, self.feet_indices, :].norm(dim=-1) > 1.).float()
        self.contacts_filt = torch.logical_or(self.contacts, self.last_contacts).float()
        
        # 更新运动时间
        self.motion_times += self.dt
        
        # 检查是否需要重新采样运动
        reset_envs = self.motion_times > self.motion_lengths
        if torch.any(reset_envs):
            reset_ids = reset_envs.nonzero(as_tuple=False).flatten()
            self._resample_motion_times(reset_ids)
        
        # 获取参考状态
        self._update_reference_states()
        
        # 计算差异
        self.dif_dof_pos = self.ref_dof_pos - self.simulator.dof_pos
        self.dif_dof_vel = self.ref_dof_vel - self.simulator.dof_vel
        self.dif_root_pos = self.ref_root_pos - self.simulator.robot_root_states[:, :3]
        
        # Noise Observation
        if self.use_noise_process:
            step_noise_process = self.noise_process.step()
            # self.base_quat_noise = self.simulator.base_quat[:]
            self.rpy_noise = get_euler_xyz_in_tensor(self.base_quat[:]) + \
                                                        step_noise_process[...,:3]*self.config.obs.noise_process.scale.rpy*(torch.pi/180.)
            self.base_quat_noise = quat_from_euler_xyz_better(self.rpy_noise)
            # assert torch.allclose(quat_from_euler_xyz_better(get_euler_xyz_in_tensor(self.base_quat_noise)), self.base_quat_noise), "base_quat_noise is not correct"
            
            
            self.base_lin_vel_noise = quat_rotate_inverse(self.base_quat_noise, self.simulator.robot_root_states[:, 7:10])
            self.base_ang_vel_noise = quat_rotate_inverse(  self.base_quat_noise, 
                                                            self.simulator.robot_root_states[:, 10:13] + 
                                                            step_noise_process[...,3:6]*self.config.obs.noise_process.scale.base_ang_vel)
            self.projected_gravity_noise = quat_rotate_inverse(self.base_quat_noise, self.gravity_vec)
            self.dof_pos_noise = self.simulator.dof_pos
            self.dof_vel_noise = self.simulator.dof_vel
            # print(self.rpy_noise,'\t|', self.base_quat_noise)
        else:
            self.base_quat_noise = self.base_quat
            self.base_lin_vel_noise = self.base_lin_vel
            self.base_ang_vel_noise = self.base_ang_vel
            self.projected_gravity_noise = self.projected_gravity
            self.dof_pos_noise = self.simulator.dof_pos
            self.dof_vel_noise = self.simulator.dof_vel

    def _update_reference_states(self):
        """更新参考状态"""
        # 使用AMPLoader获取参考状态
        ref_states = self.amp_loader.get_full_frame_at_time_batch(self.motion_ids, self.motion_times)
        
        # 提取参考状态
        self.ref_dof_pos = ref_states['dof_pos']
        self.ref_dof_vel = ref_states['dof_vel']
        self.ref_root_pos = ref_states['root_trans_offset']
        self.ref_root_rot = ref_states['root_rot']
        self.ref_root_vel = ref_states['base_lin_vel']
        self.ref_root_ang_vel = ref_states['base_ang_vel']
        
        # 将根位置添加到环境原点
        self.ref_root_pos[:, :2] += self.env_origins[:, :2]

    def _resample_motion_times(self, env_ids):
        """重新采样运动时间"""
        if len(env_ids) == 0:
            return
        
        # 重新采样运动ID和时间
        self.motion_ids[env_ids] = self.amp_loader.weighted_traj_idx_sample_batch(len(env_ids))
        self.motion_lengths[env_ids] = self.amp_loader.trajectory_lens[self.motion_ids[env_ids]]
        
        if self.is_evaluating and not self.config.enforce_randomize_motion_start_eval:
            self.motion_times[env_ids] = torch.zeros(len(env_ids), dtype=torch.float32, device=self.device)
        else:
            self.motion_times[env_ids] = self.amp_loader.traj_time_sample_batch(self.motion_ids[env_ids])

    def _update_tasks_callback(self):
        if self.config.domain_rand.push_robots:
            push_robot_env_ids = (self.push_robot_counter == (self.push_interval_s / self.dt).int()).nonzero(as_tuple=False).flatten()
            self.push_robot_counter[push_robot_env_ids] = 0
            self.push_robot_plot_counter[push_robot_env_ids] = 0
            self.push_interval_s[push_robot_env_ids] = torch.randint(self.config.domain_rand.push_interval_s[0], self.config.domain_rand.push_interval_s[1], (len(push_robot_env_ids),), device=self.device, requires_grad=False)
            self._push_robots(push_robot_env_ids)
            
        if 'reinit_epis_rand' in self.config.domain_rand and self.config.domain_rand.reinit_epis_rand > 0:
            if self.common_step_counter >= self.reinit_epis_rand_counter:
                print(f"Reinit domain rand at step {self.common_step_counter}")
                self._episodic_domain_randomization(torch.arange(self.num_envs, device=self.device))
                new_interval = -np.log(np.random.rand(1)) *self.config.domain_rand.reinit_epis_rand
                self.reinit_epis_rand_counter = self.common_step_counter + new_interval
        

    def _post_compute_observations_callback(self):
        self.last_actions[:] = self.actions[:]
        self.last_dof_pos[:] = self.simulator.dof_pos[:]
        self.last_dof_vel[:] = self.simulator.dof_vel[:]
        self.last_root_vel[:] = self.simulator.robot_root_states[:, 7:13]
        
        self.last_contacts[:] = self.contacts
        self.last_contacts_filt[:] = self.contacts_filt


    def _check_termination(self):
        """ Check if environments need to be reset
        """
        # self.reset_buf = 0
        # self.time_out_buf = 0
        # Note: DO NOT USE FOLLOWING TWO LINES STYLE
        self.reset_buf[:] = 0
        self.time_out_buf[:] = 0
        self.reset_buf_terminate_by = {
            
        }

        self._update_reset_buf()
        self._update_timeout_buf()

        self.reset_buf |= self.time_out_buf
        
        for key in self.reset_buf_terminate_by.keys():
            self.log_dict[f"terminate_by_{key}"] = self.reset_buf_terminate_by[key].float().mean()
        for key in self.log_dict.keys():
            if key.startswith("terminate_by_"):
                self.log_dict[key] = self.log_dict[key] / (self.reset_buf.float().mean() + 1e-15)
                
        # if torch.any(self.reset_buf):breakpoint()

    def _update_reset_buf(self):
        if self.config.termination.terminate_by_contact:
            self.reset_buf_terminate_by["contact"] = torch.any(torch.norm(self.simulator.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
            self.reset_buf |= self.reset_buf_terminate_by["contact"]

        if self.config.termination.terminate_by_gravity:
            # print(self.projected_gravity)
            self.reset_buf_terminate_by["gravity"] = torch.norm(self.projected_gravity[:, 0:2], dim=-1) > self.config.termination_scales.termination_gravity
            self.reset_buf |= self.reset_buf_terminate_by["gravity"]
            # self.reset_buf |= torch.any(torch.abs(self.projected_gravity[:, 0:1]) > self.config.termination_scales.termination_gravity_x, dim=1)
            # self.reset_buf |= torch.any(torch.abs(self.projected_gravity[:, 1:2]) > self.config.termination_scales.termination_gravity_y, dim=1)
        if self.config.termination.terminate_by_low_height:
            # import ipdb; ipdb.set_trace()
            self.reset_buf_terminate_by["low_height"] = torch.any(self.simulator.robot_root_states[:, 2:3] < self.config.termination_scales.termination_min_base_height, dim=1)
            self.reset_buf |= self.reset_buf_terminate_by["low_height"]

        if self.config.termination.terminate_when_close_to_dof_pos_limit:
            out_of_dof_pos_limits = -(self.simulator.dof_pos - self.simulator.dof_pos_limits_termination[:, 0]).clip(max=0.) # lower limit
            out_of_dof_pos_limits += (self.simulator.dof_pos - self.simulator.dof_pos_limits_termination[:, 1]).clip(min=0.)
            
            out_of_dof_pos_limits = torch.sum(out_of_dof_pos_limits, dim=1)
            # get random number between 0 and 1, if it is smaller than self.config.termination_probality.terminate_when_close_to_dof_pos_limit, apply the termination
            if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_dof_pos_limit:
                self.reset_buf_terminate_by["dof_pos_limit"] = out_of_dof_pos_limits > 0.
                self.reset_buf |= self.reset_buf_terminate_by["dof_pos_limit"]
            else:
                self.reset_buf_terminate_by["dof_pos_limit"] = torch.zeros_like(out_of_dof_pos_limits)
        
        if self.config.termination.terminate_when_close_to_dof_vel_limit:
            out_of_dof_vel_limits = torch.sum((torch.abs(self.simulator.dof_vel) - self.dof_vel_limits * self.config.termination_scales.termination_close_to_dof_vel_limit).clip(min=0., max=1.), dim=1)
            
            

            if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_dof_vel_limit:
                self.reset_buf_terminate_by["dof_vel_limit"] = out_of_dof_vel_limits > 0.
                self.reset_buf |= self.reset_buf_terminate_by["dof_vel_limit"]
            else:
                self.reset_buf_terminate_by["dof_vel_limit"] = torch.zeros_like(out_of_dof_vel_limits)
        
        if self.config.termination.terminate_when_close_to_torque_limit:
            out_of_torque_limits = torch.sum((torch.abs(self.torques) - self.torque_limits * self.config.termination_scales.termination_close_to_torque_limit).clip(min=0., max=1.), dim=1)
            
            if torch.rand(1) < self.config.termination_probality.terminate_when_close_to_torque_limit:
                self.reset_buf_terminate_by["torque_limit"] = out_of_torque_limits > 0.
                self.reset_buf |= self.reset_buf_terminate_by["torque_limit"]
            else:
                self.reset_buf_terminate_by["torque_limit"] = torch.zeros_like(out_of_torque_limits)
                
                
        # if self.reset_buf.any():
        #     breakpoint()

    def _update_timeout_buf(self):
        self.time_out_buf |= self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        
        # self.log_dict["terminate_by_time_out"] = self.time_out_buf.float().mean()
        self.reset_buf_terminate_by["time_out"] = self.time_out_buf

    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
            target_states (dict): Dictionary containing lists of target states for the robot
        """
        if len(env_ids) == 0:
            return
        self.need_to_refresh_envs[env_ids] = True
        self._reset_buffers_callback(env_ids, target_buf)
        self._reset_tasks_callback(env_ids)
        self._reset_robot_states_callback(env_ids)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = (self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.extras["time_outs"] = self.time_out_buf
        self.extras["episode"]["end_epis_length"] = self.last_episode_length_buf[env_ids]
        # self._refresh_sim_tensors()
        # breakpoint()

    def _reset_dofs_amp(self, env_ids, frames):
        """ Resets DOF position and velocities of selected environmments using AMP frames
        Args:
            env_ids (List[int]): Environemnt ids
            frames: AMP frames to initialize motion with
        """
        self.simulator.dof_pos[env_ids] = self.amp_loader.get_joint_pose_batch(frames).to(self.device)
        self.simulator.dof_vel[env_ids] = self.amp_loader.get_joint_vel_batch(frames).to(self.device)

    def _reset_root_states_amp(self, env_ids, frames):
        """ Resets ROOT states position and velocities of selected environmments using AMP frames
        Args:
            env_ids (List[int]): Environemnt ids
            frames: AMP frames to initialize motion with
        """
        # base position
        root_pos = self.amp_loader.get_root_trans_offset_batch(frames).to(self.device)
        root_pos[:, :2] = root_pos[:, :2] + self.env_origins[env_ids, :2]
        self.simulator.robot_root_states[env_ids, :3] = root_pos
        root_rot = self.amp_loader.get_root_rot_batch(frames).to(self.device)
        self.simulator.robot_root_states[env_ids, 3:7] = root_rot
        self.simulator.robot_root_states[env_ids, 7:10] = quat_rotate(root_rot, self.amp_loader.get_linear_vel_batch(frames).to(self.device))
        self.simulator.robot_root_states[env_ids, 10:13] = quat_rotate(root_rot, self.amp_loader.get_angular_vel_batch(frames).to(self.device))

    def _reset_robot_states_callback(self, env_ids):
        frames = self.amp_loader.get_full_frame_batch(len(env_ids))
        self._reset_dofs_amp(env_ids, frames)
        self._reset_root_states_amp(env_ids, frames)
        
        # 重置运动时间
        self._resample_motion_times(env_ids)