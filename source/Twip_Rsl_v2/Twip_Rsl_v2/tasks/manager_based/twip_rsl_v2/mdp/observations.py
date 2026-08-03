from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_G = 9.81


def imu_lin_acc(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Simulated accelerometer reading R^T*(a_body + g), normalized by g. Shape (N, 3).

    Body acceleration corrupts the tilt estimate here, just like a real accelerometer.
    """
    imu = env.scene["imu"]
    return imu.data.lin_acc_b / _G


def imu_ang_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gyroscope reading from the simulated IMU, in body frame. Shape (N, 3), rad/s."""
    imu = env.scene["imu"]
    return imu.data.ang_vel_b


def imu_pitch_angle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Body tilt angle (rad) about the wheel axis, from projected gravity (clean tilt). Shape (N, 1).

    Projected gravity (not the raw accelerometer) is used so motion doesn't corrupt the angle,
    matching the complementary-filtered pitch the firmware feeds in. Upright -> [0,0,-1] -> pitch 0.
    """
    robot = env.scene["robot"]
    pg = robot.data.projected_gravity_b          # (N, 3), = [0, 0, -1] when upright
    return torch.atan2(-pg[:, 1], -pg[:, 2]).unsqueeze(1)


def imu_pitch_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Body tilt rate (rad/s) about the wheel axis, from the base angular velocity. Shape (N, 1).

    Taken from the base so it stays consistent with imu_pitch_angle (rate = d(pitch)/dt), which
    lets the policy learn a coherent PD controller.
    """
    robot = env.scene["robot"]
    return robot.data.root_ang_vel_b[:, 0].unsqueeze(1)
