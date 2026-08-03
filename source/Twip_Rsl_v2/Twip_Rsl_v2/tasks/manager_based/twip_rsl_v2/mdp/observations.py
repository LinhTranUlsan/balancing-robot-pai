from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_G = 9.81


def imu_lin_acc(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Accelerometer reading from simulated IMU.

    Unlike projected_gravity (which is R^T * g — a perfect, noise-free tilt measurement),
    this returns R^T * (a_body + gravity_bias), i.e. what a real accelerometer sees.
    When the robot accelerates, body dynamics corrupt the tilt estimate — exactly as on hardware.

    Output shape: (N, 3), normalized by g so values are dimensionless (~[-1, 1] when near-level).
    """
    imu = env.scene["imu"]
    return imu.data.lin_acc_b / _G


def imu_ang_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Gyroscope reading from simulated IMU, expressed in IMU (body) frame.

    Shape: (N, 3), units rad/s.
    """
    imu = env.scene["imu"]
    return imu.data.ang_vel_b


def imu_pitch_angle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Body tilt angle (rad) about the wheel axis, from PROJECTED GRAVITY (clean true tilt).

    was atan2 of the RAW accelerometer (imu.lin_acc_b). During balancing the base
    accelerates, which corrupts that accelerometer tilt estimate, so the policy learned to IGNORE
    the angle and collapsed to a pure rate-damper (no proportional term) -> it cannot correct a
    static lean, so the real robot drifts and falls. The real firmware feeds a COMPLEMENTARY-FILTERED
    pitch (clean), so we train on projected_gravity_b (= true tilt) to match what the hardware sees.
    Residual sensor/filter noise is added by the ObsTerm's Gaussian noise (kept small on purpose).

    Upright: projected_gravity_b = [0, 0, -1] -> pitch = 0. Tilt about X_robot by phi -> pitch = phi.
    Shape: (N, 1).
    """
    robot = env.scene["robot"]
    pg = robot.data.projected_gravity_b          # (N, 3), = [0, 0, -1] when upright
    return torch.atan2(-pg[:, 1], -pg[:, 2]).unsqueeze(1)


def imu_pitch_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Body tilt rate (rad/s) about the wheel axis X_robot.

    read from the robot base angular velocity so it stays EXACTLY consistent with
    imu_pitch_angle above (rate = d(pitch)/dt, same axis and sign). Physically identical to the old
    gyro reading (imu ang_vel_b[1] == robot X after the +90 deg mount), just taken cleanly from the
    base. Consistency between angle and rate is what lets the policy learn a coherent PD controller.

    Shape: (N, 1).
    """
    robot = env.scene["robot"]
    return robot.data.root_ang_vel_b[:, 0].unsqueeze(1)
