import os
import copy
import torch
from torch import nn
import numpy as np
import random
import pickle
import joblib
from typing import Any, List, Dict
from termcolor import colored
from loguru import logger

np2torch = lambda x: torch.tensor(x, dtype=torch.float32)
torch2np = lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x

def compute_obs_key_slices(config, obs_dim_dict, each_dict_obs_dims, auxiliary_obs_dims, future_mimic_dim_dict):
    """
    返回一个 dict: 
    {
        obs_key: {
            sub_key1: (start_idx, end_idx),
            sub_key2: (start_idx, end_idx),
            ...
        },
        ...
    }
    """
    obs_slices = dict()
    _obs_key_list = config.env.config.obs.obs_dict
    _aux_obs_key_list = config.env.config.obs.obs_auxiliary

    for obs_key, sub_keys in _obs_key_list.items():
        current_pos = 0
        obs_slices[obs_key] = dict()
        for key in sorted(sub_keys):
            raw_key = key[:-4] if key.endswith("_raw") else key
            if raw_key in each_dict_obs_dims:
                dim = each_dict_obs_dims[raw_key]
            elif raw_key == "future_mimic_buf":
                dim = future_mimic_dim_dict[raw_key]
            else:
                dim = auxiliary_obs_dims[raw_key]
            obs_slices[obs_key][key] = (current_pos, current_pos + dim)
            current_pos += dim

    return obs_slices


def determine_obs_dim(config) -> None:
    obs_dim_dict = dict()
    _obs_key_list = config.env.config.obs.obs_dict
    _aux_obs_key_list = config.env.config.obs.obs_auxiliary
    
    assert set(config.env.config.obs.noise_scales.keys()) == set(config.env.config.obs.obs_scales.keys())

    # convert obs_dims to list of dicts
    each_dict_obs_dims = {k: v for d in config.env.config.obs.obs_dims for k, v in d.items()}
    config.env.config.obs.obs_dims = each_dict_obs_dims
    logger.info(f"obs_dims: {each_dict_obs_dims}")

    future_mimic_dim_dict = None
    future_mimic_enabled = True if "tar_obs_mimic" in config.env.config.obs and config.env.config.obs.tar_obs_mimic.enabled else False
    if future_mimic_enabled:
        future_mimic_dim_dict = {}
        future_mimic_dim_dict['future_mimic_buf'] = 0
        for key in config.env.config.obs.future_mimic_buf:
            future_mimic_dim_dict['future_mimic_buf'] += each_dict_obs_dims[key]
        future_mimic_dim_dict['future_mimic_buf'] *= config.env.config.obs.tar_obs_mimic.tar_obs_mimic_counter
        
        logger.info(f"future_mimic_dim: {future_mimic_dim_dict}")

    auxiliary_obs_dims = {}
    for aux_obs_key, aux_config in _aux_obs_key_list.items():
        auxiliary_obs_dims[aux_obs_key] = 0
        history_len = config.env.config.obs.get(f"{aux_obs_key}_len")
        for _key, _num in aux_config.items():
            assert history_len == _num
            assert _key in config.env.config.obs.obs_dims.keys()
            auxiliary_obs_dims[aux_obs_key] += config.env.config.obs.obs_dims[_key] * _num
    logger.info(f"auxiliary_obs_dims: {auxiliary_obs_dims}")
    
    for obs_key, obs_config in _obs_key_list.items():
        obs_dim_dict[obs_key] = 0
        for key in obs_config:
            if key.endswith("_raw"): key = key[:-4]
            if key in config.env.config.obs.obs_dims.keys(): 
                obs_dim_dict[obs_key] += config.env.config.obs.obs_dims[key]
                logger.info(f"{obs_key}: {key} has dim: {config.env.config.obs.obs_dims[key]}")
            elif key == "future_mimic_buf":
                obs_dim_dict[obs_key] += future_mimic_dim_dict[key]
                logger.info(f"{obs_key}: {key} has dim: {future_mimic_dim_dict[key]}")
            else:
                obs_dim_dict[obs_key] += auxiliary_obs_dims[key]
                logger.info(f"{obs_key}: {key} has dim: {auxiliary_obs_dims[key]}")
                
                
    config.robot.algo_obs_dim_dict = obs_dim_dict
    logger.info(f"algo_obs_dim_dict: {config.robot.algo_obs_dim_dict}")
    return obs_dim_dict, each_dict_obs_dims, auxiliary_obs_dims, future_mimic_dim_dict

def pre_process_config(config) -> None:
    
    # compute observation_dim
    # config.robot.policy_obs_dim = -1
    # config.robot.critic_obs_dim = -1
    if not hasattr(config, '_preprocessed'):
        from omegaconf import OmegaConf
        OmegaConf.set_struct(config, False)
        config._preprocessed = True
    else:
        return 
    
    obs_dim_dict, each_dict_obs_dims, auxiliary_obs_dims, future_mimic_dim_dict = determine_obs_dim(config)
    
    obs_slices = compute_obs_key_slices(config, obs_dim_dict, each_dict_obs_dims, auxiliary_obs_dims, future_mimic_dim_dict)
    config.env.config.obs.post_compute_config["obs_slices"] = obs_slices
    print(f"obs_slices: {obs_slices}")
    # breakpoint()

    # for mh_ppo actor
    config.algo.config.module_dict.actor.history_length = config.obs.history_actor_len
    config.algo.config.module_dict.actor.history_single_length = 0
    for key in config.obs.obs_auxiliary.history_actor.keys():
        config.algo.config.module_dict.actor.history_single_length += each_dict_obs_dims[key]

    config.algo.config.module_dict.actor.motion_length = config.obs.tar_obs_mimic.tar_obs_mimic_counter
    motion_single_length = 0
    for key in config.obs.future_mimic_buf:
        motion_single_length += each_dict_obs_dims[key]
    config.algo.config.module_dict.actor.motion_single_length = motion_single_length
    print(config.algo.config.module_dict.actor)

    # for mh_ppo critic
    config.algo.config.module_dict.critic.history_length = config.obs.history_critic_len
    config.algo.config.module_dict.critic.history_single_length = 0
    for key in config.obs.obs_auxiliary.history_critic.keys():
        config.algo.config.module_dict.critic.history_single_length += each_dict_obs_dims[key]
    print(config.algo.config.module_dict.critic)
                
    if config.log_task_name=='motion_tracking':
        motion_file = config.robot.motion.motion_file
        if os.path.isfile(motion_file):
            with open(motion_file, 'rb') as f:
                motion_data = joblib.load(f)
            assert len(motion_data) == 1, 'current only support single motion tracking'
            the_motion_data = motion_data[next(iter(motion_data))]
            the_motion_data['fps'] = int(the_motion_data['fps'])
            assert type(the_motion_data['fps']) == int, 'motion fps should be an integer'
            config.obs.motion_len = len(the_motion_data['dof']) / the_motion_data['fps']
            config.obs.motion_file = motion_file
            logger.info(f"motion_len: {config.obs.motion_len}")
            logger.info(f"motion_file: {config.obs.motion_file}")
        else:
            config.obs.motion_len = -1
            config.obs.motion_file = None
    else:
        config.obs.motion_len = -1
        config.obs.motion_file = None
    # print the config
    logger.debug(f"PPO CONFIG")

def parse_observation(cls: Any, 
                      key_list: List, 
                      buf_dict: Dict, 
                      obs_scales: Dict, 
                      noise_scales: Dict,
                      current_noise_curriculum_value: Any) -> None:
    """ Parse observations for the legged_robot_base class
    """
    # breakpoint()
    for obs_key in key_list:
        if obs_key.endswith("_raw"):
            obs_key = obs_key[:-4]
            obs_noise = 0.
        else:
            obs_noise = noise_scales[obs_key] * current_noise_curriculum_value
        
        # print(f"obs_key: {obs_key}, obs_noise: {obs_noise}")
        
        actor_obs = getattr(cls, f"_get_obs_{obs_key}")().clone()
        obs_scale = obs_scales[obs_key]
        # Yuanhang: use rand_like (uniform 0-1) instead of randn_like (N~[0,1])
        # buf_dict[obs_key] = actor_obs * obs_scale + (torch.randn_like(actor_obs)* 2. - 1.) * obs_noise
        # print("noise_scales", noise_scales)
        # print("obs_noise", obs_noise)
        buf_dict[obs_key] = (actor_obs + (torch.rand_like(actor_obs)* 2. - 1.) * obs_noise) * obs_scale
