
from pathlib import Path
from humanoidverse.utils.motion_lib.skeleton import SkeletonTree
from loguru import logger
import os
from enum import Enum
import glob
import joblib

class MotionlibMode(Enum):
    file = 1
    directory = 2

class MotionLibBase():
    
    def __init__(self, motion_lib_cfg, num_envs, device):
        
        self.motion_config = motion_lib_cfg
        self._sim_fps = 1/self.motion_config.get("step_dt", 1/50)

        self.num_envs = num_envs
        self._device = device
        self.mesh_parsers = None

        skeleton_file = Path(self.motion_config.asset.assetRoot) / self.motion_config.asset.assetFileName
        self.skeleton_tree = SkeletonTree.from_mjcf(skeleton_file)

        logger.info(f"Loaded skeleton from {skeleton_file}")
        logger.info(f"Loading motion data from {self.motion_config.motion_file}...")

        self.load_data(self.motion_config.motion_file)
    
    def load_data(self, motion_file, min_length=-1, im_eval=False):
        if os.path.isfile(motion_file):
            self.mode = MotionlibMode.file
            self._motion_data_load = [motion_file]
        else:
            self.mode = MotionlibMode.directory
            self._motion_data_load = glob.glob(os.path.join(motion_file, "*.pkl"))
        


class MotionLibRobotJJH(MotionLibBase):
    def __init__(self, motion_lib_cfg, num_envs, device):
        super().__init__(motion_lib_cfg=motion_lib_cfg, num_envs=num_envs, device=device)
