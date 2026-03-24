# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pace agent for Isaac Lab environments.")
parser.add_argument(
    "--num_envs", type=int, default=4096, help="Number of environments to simulate."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Pace-Unitree-G1-29dof-v0",
    help="Name of the task.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import pace_sim2real.tasks  # noqa: F401
from pace_sim2real.utils import project_root
from pace_sim2real import CMAESOptimizer


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # Create optimization
    bounds_params = env_cfg.sim2real.bounds_params.to(env.unwrapped.device)
    articulation = env.unwrapped.scene["robot"]
    joint_order = env_cfg.sim2real.joint_order
    sim_joint_ids = torch.tensor(
        [articulation.joint_names.index(name) for name in joint_order],
        device=env.unwrapped.device,
    )

    data_file = project_root() / "data" / env_cfg.sim2real.data_dir
    log_dir = project_root() / "logs" / "pace" / env_cfg.sim2real.robot_name

    data = torch.load(data_file)
    time_data = data["time"].to(env.unwrapped.device)
    target_dof_pos = data["des_dof_pos"].to(env.unwrapped.device)
    measured_dof_pos = data["dof_pos"].to(env.unwrapped.device)

    # Use joint_names from data file if available (subset of joints),
    # otherwise fall back to full joint_order from config
    data_joint_names = data.get("joint_names", joint_order)
    # Map data columns to simulator joint indices
    data_joint_ids = torch.tensor(
        [articulation.joint_names.index(name) for name in data_joint_names],
        device=env.unwrapped.device,
    )
    # Filter bounds_params to only the collected joints
    # bounds_params shape: (num_joints*4 + 1, 2) — armature, damping, friction, bias (each num_joints rows), then 1 delay row
    data_joint_order_indices = torch.tensor(
        [joint_order.index(name) for name in data_joint_names],
        device=env.unwrapped.device,
    )
    num_all_joints = len(joint_order)
    param_indices = []
    for group in range(4):  # armature, damping, friction, bias
        param_indices.append(group * num_all_joints + data_joint_order_indices)
    # append delay (last row)
    param_indices.append(
        torch.tensor([4 * num_all_joints], device=env.unwrapped.device)
    )
    param_indices = torch.cat(param_indices)
    bounds_params = bounds_params[param_indices, :]

    env.reset()

    # Regularization: armature만 ground truth 존재 (Unitree 제공)
    nominal_armature = articulation.data.default_joint_armature[0, data_joint_ids]
    num_data_joints = len(data_joint_names)
    # nominal_params를 sim_params와 같은 layout으로 구성 (armature + damping + friction + bias + delay)
    # armature만 실제 값, 나머지는 0 (regularization에서 사용하지 않음)
    nominal_params = torch.zeros(4 * num_data_joints + 1, device=env.unwrapped.device)
    nominal_params[:num_data_joints] = nominal_armature
    print(f"[INFO]: Nominal Armature: {nominal_armature}")
    print(f"[INFO]: Data joint IDs (sim): {data_joint_ids}")

    initial_dof_pos = (
        measured_dof_pos[0, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
    )

    time_steps = time_data.shape[0]
    sim_dt = env.unwrapped.sim.cfg.dt

    opt = CMAESOptimizer(
        bounds=bounds_params,
        population_size=env.unwrapped.num_envs,
        log_dir=log_dir,
        joint_order=data_joint_names,
        max_iteration=env_cfg.sim2real.cmaes.max_iteration,
        data=data,
        device=env.unwrapped.device,
        epsilon=env_cfg.sim2real.cmaes.epsilon,
        sigma=env_cfg.sim2real.cmaes.sigma,
        save_interval=env_cfg.sim2real.cmaes.save_interval,
        save_optimization_process=env_cfg.sim2real.cmaes.save_optimization_process,
        nominal_params=nominal_params,
        reg_weight=0.01,
    )

    env.reset()
    opt.update_simulator(articulation, data_joint_ids, initial_dof_pos)

    counter = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions — compare only collected joints
            opt.tell(
                env.unwrapped.scene.articulations["robot"].data.joint_pos[
                    :, data_joint_ids
                ],
                measured_dof_pos[counter, :]
                .unsqueeze(0)
                .repeat(env.unwrapped.num_envs, 1),
            )
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions[:, data_joint_ids] = (
                target_dof_pos[counter, :]
                .unsqueeze(0)
                .repeat(env.unwrapped.num_envs, 1)
            )
            # apply actions
            env.step(actions)
            counter += 1
            if counter % 400 == 0:
                print(
                    f"[INFO]: Step {counter * sim_dt:.1f} / {time_data[-1]:.1f} seconds ({counter / time_steps * 100:.1f} %)"
                )
            if counter >= time_steps:
                print("[INFO]: Reached the end of the trajectory, exiting.")
                counter = 0
                opt.evolve()
                if opt.finished():
                    break
                env.reset()
                opt.update_simulator(
                    env.unwrapped.scene["robot"], data_joint_ids, initial_dof_pos
                )
    # close optimizer
    opt.close()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
