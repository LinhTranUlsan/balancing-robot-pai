# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

from .observations import imu_pitch_angle, imu_pitch_rate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_upright_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tilt: sum of squares of the x/y components of projected_gravity_b (0 when upright)."""
    asset: Articulation = env.scene[asset_cfg.name]
    proj_grav = asset.data.projected_gravity_b
    return torch.sum(torch.square(proj_grav[:, :2]), dim=1)


def base_upright_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float = 0.2) -> torch.Tensor:
    """Bounded upright reward via an exponential kernel: 1.0 when upright, decaying to 0 as it tilts."""
    asset: Articulation = env.scene[asset_cfg.name]
    proj_grav = asset.data.projected_gravity_b
    tilt_sq = torch.sum(torch.square(proj_grav[:, :2]), dim=1)
    return torch.exp(-tilt_sq / std**2)


def ang_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the robot rotating about the Z axis (yaw), computed as the square of the yaw angular velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])


def lin_vel_x_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the robot moving along the X axis (body frame), computed as the square of the linear velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 0])


def pid_mimic_l2(
    env: ManagerBasedRLEnv, kp: float = 8.0, kd: float = 0.5, angle_setpoint: float = 0.0
) -> torch.Tensor:
    """Shaping reward pulling the action toward a reference PD target clamp(kp*(pitch-setpoint)+kd*rate).

    No integral term (a reward function has no state). If it worsens oscillation, flip the kp/kd sign.
    """
    pitch_angle = imu_pitch_angle(env).squeeze(-1)  # (N,)
    pitch_rate = imu_pitch_rate(env).squeeze(-1)  # (N,)
    pid_target = torch.clamp(kp * (pitch_angle - angle_setpoint) + kd * pitch_rate, -1.0, 1.0)

    policy_action = env.action_manager.action  # (N, num_actions), raw action [-1, 1] for both wheels
    pid_target = pid_target.unsqueeze(-1).expand_as(policy_action)
    return torch.mean(torch.square(policy_action - pid_target), dim=1)


def wheel_vel_diff_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the two wheels spinning differently: square of their angular-velocity difference.

    asset_cfg.joint_ids must point to the two wheel joints.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.square(wheel_vel[:, 0] - wheel_vel[:, 1])

