# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a trained checkpoint and plot the pitch angle (deg) over time to inspect the oscillation
behavior (e.g. swinging 30 -> 10 -> 30 deg) numerically at a glance, instead of watching a video."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Chạy checkpoint RSL-RL và vẽ đồ thị góc pitch theo thời gian.")
parser.add_argument("--num_steps", type=int, default=1000, help="Số control step muốn ghi lại (mặc định ~10s @ 100Hz).")
parser.add_argument("--out", type=str, default="pitch_plot.png", help="Đường dẫn file ảnh PNG xuất ra.")
parser.add_argument("--num_envs", type=int, default=1, help="Số môi trường (nên để 1 để dễ theo dõi 1 robot).")
parser.add_argument("--task", type=str, default=None, help="Tên task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Twip_Rsl_v2.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    obs = env.get_observations()
    pitch_deg_log = []
    dones_log = []

    with torch.inference_mode():
        for _ in range(args_cli.num_steps):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)
            # obs order (concatenate_terms=True): [pitch_angle, pitch_rate, wheel_vel_0, wheel_vel_1]
            pitch_rad = obs["policy"][0, 0].item()
            pitch_deg_log.append(np.degrees(pitch_rad))
            dones_log.append(bool(dones[0].item()))

    dt = env.unwrapped.step_dt
    t = np.arange(len(pitch_deg_log)) * dt

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, pitch_deg_log, label="pitch angle (deg)")
    for i, d in enumerate(dones_log):
        if d:
            ax.axvline(t[i], color="red", linestyle="--", alpha=0.5)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch angle (deg)")
    ax.set_title("Pitch angle theo thời gian (nét đứt đỏ = episode reset/termination)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args_cli.out, dpi=150)
    print(f"[INFO]: Saved plot to: {os.path.abspath(args_cli.out)}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
