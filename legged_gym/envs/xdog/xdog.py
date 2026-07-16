# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .xdog_rough_config import XdogRoughCfg

import math

class Xdog(LeggedRobot):
    cfg: XdogRoughCfg

    # def set_camera(self, position, lookat):
    #     """ Set camera position and direction
    #     """
    #     # cam_pos = gymapi.Vec3(position[0], position[1], position[2])
    #     # cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
    #     actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
    #     self.gym.refresh_actor_root_state_tensor(self.sim)
    #     self.root_states = gymtorch.wrap_tensor(actor_root_state)
    #
    #     cam_target = gymapi.Vec3(self.root_states[-1, 0], self.root_states[-1, 1], self.root_states[-1, 2])
    #     # cam_pos = cam_target + gymapi.Vec3(0.0, -1.5, 0.5)
    #     cam_pos = cam_target + gymapi.Vec3(5, 0, 0.5)
    #     self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
