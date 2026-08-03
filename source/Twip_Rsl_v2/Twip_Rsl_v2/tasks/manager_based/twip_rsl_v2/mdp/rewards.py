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
    """
    Penalize the robot when tilted. Based on projected_gravity_b (the world's Z vector projected into the chassis frame).
    When perfectly upright this vector is [0, 0, 1]. We penalize the x (Roll) and y (Pitch) axes.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    proj_grav = asset.data.projected_gravity_b
    # Sum of squares of the x and y axes
    return torch.sum(torch.square(proj_grav[:, :2]), dim=1)


def base_upright_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float = 0.2) -> torch.Tensor:
    """
    Reward for standing upright via an exponential kernel: equals 1.0 when perfectly upright,
    decaying to 0 as it tilts. Complements base_upright_penalty (an unbounded penalty form).
    """
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
    """
    Penalize the deviation between the action the policy chooses and the action a reference PD controller
    (target = clamp(kp*(pitch_angle - angle_setpoint) + kd*pitch_rate, -1, 1)) would produce for the same current state.
    There is no I (integral) term because it would need its own state buffer, which does not fit a pure reward function.

    angle_setpoint: target pitch angle (rad) -- 0.0 means the robot is upright along the Z axis (not tilted
    to either side), the desired balanced state.

    This is an "imitation" (shaping) reward, not a replacement for upright/terminating -- it only pulls the
    policy toward near-linear PD behavior; the other rewards still decide whether it actually balances.

    Note: the sign of kp/kd has not been verified against the robot's true axis convention -- if applying it
    makes oscillation worse (instead of reducing it), try flipping the sign of kp and kd.
    """
    pitch_angle = imu_pitch_angle(env).squeeze(-1)  # (N,)
    pitch_rate = imu_pitch_rate(env).squeeze(-1)  # (N,)
    pid_target = torch.clamp(kp * (pitch_angle - angle_setpoint) + kd * pitch_rate, -1.0, 1.0)

    policy_action = env.action_manager.action  # (N, num_actions), raw action [-1, 1] for both wheels
    pid_target = pid_target.unsqueeze(-1).expand_as(policy_action)
    return torch.mean(torch.square(policy_action - pid_target), dim=1)


def wheel_vel_diff_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the two wheels spinning in opposite directions/at different speeds (square of the difference of their angular velocities).

    asset_cfg.joint_ids must point to the two wheel joints. Opposite spin -> large difference -> heavy penalty.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.square(wheel_vel[:, 0] - wheel_vel[:, 1])

