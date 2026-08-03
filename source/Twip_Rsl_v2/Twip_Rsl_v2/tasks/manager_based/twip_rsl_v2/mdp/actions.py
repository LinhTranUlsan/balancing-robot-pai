# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.actions.actions_cfg import JointEffortActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointEffortAction
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# NOTE (superseded): this 2-D per-wheel action is kept for reference only. The env now uses
# SymmetricWheelEffortAction (1-D, defined at the bottom of this file) so the exported ONNX has a
# single output matching the on-board firmware. See ActionsCfg in twip_rsl_v2_env_cfg.py.
class JointEffortActionWithDeadzone(JointEffortAction):
    """Joint effort action that models the dead-zone of a real motor driver.

    A real driver needs a PWM pulse above a certain threshold (due to static friction / gearbox
    stiffness) before it produces torque. When |raw_action| is below the ``cfg.deadzone`` threshold
    (normalized [-1, 1] scale, same as the raw action), the output torque is forced to 0 instead of
    being scaled linearly as usual.
    """

    cfg: JointEffortActionWithDeadzoneCfg

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        dead_mask = self._raw_actions.abs() < self.cfg.deadzone
        self._processed_actions[dead_mask] = 0.0


@configclass
class JointEffortActionWithDeadzoneCfg(JointEffortActionCfg):
    """Configuration for :class:`JointEffortActionWithDeadzone`."""

    class_type: type = JointEffortActionWithDeadzone

    deadzone: float = 0.0
    """Dead-zone threshold, in terms of the normalized raw action [-1, 1] (same scale as the action before scaling)."""


# ======================================================================================
# >>> ADDED: single symmetric wheel effort (1-D action) for the RL -> firmware pipeline.
# ======================================================================================
class SymmetricWheelEffortAction(ActionTerm):
    """One scalar effort command applied IDENTICALLY to every wheel joint (action_dim == 1).

    Why: the physical robot's on-board firmware sends a SINGLE motor command to both wheels
    (POLICY_ACT_DIM == 1). A normal JointEffortAction over 2 joints makes the policy output 2
    values (one per wheel) and lets it steer/yaw -- which the firmware cannot reproduce and which
    only complicates balancing. Keeping action_dim == 1 makes the exported ONNX have exactly one
    output that maps directly onto the firmware's ``u`` in [-1, 1].

    Per step: raw action (N, 1) in [-1, 1] -> dead-zone -> * scale (Nm)
    -> clamp to [-scale, scale] -> broadcast the SAME torque to all wheel joints.
    """

    cfg: SymmetricWheelEffortActionCfg
    _asset: Articulation

    def __init__(self, cfg: SymmetricWheelEffortActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        # resolve the wheel joints this single command drives
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        # raw buffer is 1-D (one command); processed buffer is broadcast to every wheel joint
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        cmd = self._raw_actions.clone()
        # dead-zone on the normalized command (models driver static friction / gearbox stiffness)
        cmd[cmd.abs() < self.cfg.deadzone] = 0.0
        # scale to torque and clamp to the physical motor limit (keeps raw actor output ~[-1, 1])
        torque = torch.clamp(cmd * self.cfg.scale, -self.cfg.scale, self.cfg.scale)
        # same torque on every wheel -> pure fore/aft balancing, no differential/yaw (matches firmware)
        self._processed_actions = torque.repeat(1, self._num_joints)

    def apply_actions(self) -> None:
        self._asset.set_joint_effort_target(self._processed_actions, joint_ids=self._joint_ids)


@configclass
class SymmetricWheelEffortActionCfg(ActionTermCfg):
    """Configuration for :class:`SymmetricWheelEffortAction`."""

    class_type: type[ActionTerm] = SymmetricWheelEffortAction

    joint_names: list[str] = MISSING
    """Wheel joints that all receive the (identical) effort command."""

    scale: float = 1.0
    """Raw action in [-1, 1] is multiplied by this to get torque (Nm), then clamped to [-scale, scale]."""

    deadzone: float = 0.0
    """Below this |raw action| the torque is forced to 0 (models the real motor-driver dead-zone)."""
