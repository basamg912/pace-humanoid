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
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Pace-Unitree-G1-29dof-v0",
    help="Name of the task.",
)
parser.add_argument(
    "--min_frequency",
    type=float,
    default=0.1,
    help="Minimum frequency for the chirp signal in Hz.",
)
parser.add_argument(
    "--max_frequency",
    type=float,
    default=10.0,
    help="Maximum frequency for the chirp signal in Hz.",
)
parser.add_argument(
    "--duration",
    type=float,
    default=20.0,
    help="Duration of the chirp signal in seconds.",
)
parser.add_argument(
    "--test", type=bool, default=False, help="for Testing, True save logs"
)
parser.add_argument(
    "--joints",
    type=str,
    nargs="+",
    default=["left_leg", "right_leg", "left_arm", "right_arm"],
    choices=["left_leg", "right_leg", "waist", "left_arm", "right_arm"],
    help="Joint groups to excite and collect data for.",
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
from torch import pi

import pace_sim2real.tasks  # noqa: F401
from pace_sim2real.utils import project_root


def main():
    # parse configuration
    test = args_cli.test

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    # create environment
    #               entry-point             env_cfg_entry-point
    # gym.make = ManagerBasedRLEnv(cfg=PaceSim2realEnvcfg())
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    articulation = env.unwrapped.scene["robot"]

    joint_order = env_cfg.sim2real.joint_order
    joint_ids = torch.tensor(
        [articulation.joint_names.index(name) for name in joint_order],
        device=env.unwrapped.device,
    )
    print(f"\n[INFO]: joint_order: {joint_order} \n")
    print(f"[INFO]: joint_ids: {joint_ids}")

    # # config 에서 제공하는 damping 설정은 PD-GAIN 을 의미하는 것으로 확인
    # print(
    #     "[INFO]: Viscous Damping or PD gain?",
    #     articulation.data.default_joint_viscous_friction_coeff[:, joint_ids],
    # )
    armature = torch.tensor(
        [0.01] * len(joint_ids), device=env.unwrapped.device
    ).unsqueeze(0)
    damping = torch.tensor(
        [3.0] * len(joint_ids), device=env.unwrapped.device
    ).unsqueeze(0)
    friction = torch.tensor(
        [0.05] * len(joint_ids), device=env.unwrapped.device
    ).unsqueeze(0)

    # joint bias (initial pose 에 영향)
    bias = torch.tensor([0.02] * len(joint_ids), device=env.unwrapped.device).unsqueeze(
        0
    )
    time_lag = torch.tensor([[5]], dtype=torch.int, device=env.unwrapped.device)
    env.reset()

    articulation.write_joint_armature_to_sim(
        armature, joint_ids=joint_ids, env_ids=torch.arange(len(armature))
    )
    articulation.data.default_joint_armature[:, joint_ids] = armature
    articulation.write_joint_viscous_friction_coefficient_to_sim(
        damping, joint_ids=joint_ids, env_ids=torch.arange(len(damping))
    )
    articulation.data.default_joint_viscous_friction_coeff[:, joint_ids] = damping
    articulation.write_joint_friction_coefficient_to_sim(
        friction, joint_ids=joint_ids, env_ids=torch.tensor([0])
    )
    articulation.data.default_joint_friction_coeff[:, joint_ids] = friction
    drive_types = articulation.actuators.keys()
    for drive_type in drive_types:
        drive_indices = articulation.actuators[drive_type].joint_indices
        if isinstance(drive_indices, slice):
            all_idx = torch.arange(joint_ids.shape[0], device=joint_ids.device)
            drive_indices = all_idx[drive_indices]
        comparison_matrix = joint_ids.unsqueeze(1) == drive_indices.unsqueeze(0)
        drive_joint_idx = torch.argmax(comparison_matrix.int(), dim=0)
        articulation.actuators[drive_type].update_time_lags(time_lag)
        articulation.actuators[drive_type].update_encoder_bias(bias[:, drive_joint_idx])
        articulation.actuators[drive_type].reset(torch.arange(env.unwrapped.num_envs))

    data_dir = project_root() / "data" / env_cfg.sim2real.robot_name

    # Create a chirp signal for each action dimension

    duration = args_cli.duration  # seconds
    sample_rate = 1 / env.unwrapped.sim.get_physics_dt()  # Hz
    num_steps = int(duration * sample_rate)
    t = torch.linspace(0, duration, steps=num_steps, device=env.unwrapped.device)
    f0 = args_cli.min_frequency  # Hz
    f1 = args_cli.max_frequency  # Hz

    # Linear chirp: phase = 2*pi*(f0*t + (f1-f0)/(2*duration)*t^2)
    phase = 2 * pi * (f0 * t + ((f1 - f0) / (2 * duration)) * t**2)
    chirp_signal = torch.sin(phase)

    trajectory = torch.zeros((num_steps, len(joint_ids)), device=env.unwrapped.device)
    trajectory[:, :] = chirp_signal.unsqueeze(-1)
    # Per-joint chirp parameters for G1 29-DOF
    # Joint order:
    #  0: left_hip_pitch       1: left_hip_roll        2: left_hip_yaw
    #  3: left_knee             4: left_ankle_pitch     5: left_ankle_roll

    #  6: right_hip_pitch       7: right_hip_roll       8: right_hip_yaw
    #  9: right_knee            10: right_ankle_pitch   11: right_ankle_roll

    # 12: waist_yaw            13: waist_roll          14: waist_pitch

    # 15: left_shoulder_pitch  16: left_shoulder_roll  17: left_shoulder_yaw
    # 18: left_elbow           19: left_wrist_roll     20: left_wrist_pitch
    # 21: left_wrist_yaw

    # 22: right_shoulder_pitch 23: right_shoulder_roll 24: right_shoulder_yaw
    # 25: right_elbow          26: right_wrist_roll    27: right_wrist_pitch
    # 28: right_wrist_yaw

    # joint order 기준 작성
    trajectory_directions = torch.tensor(
        [
            # Left leg 1-6
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            # Right leg (roll mirrored) 7-12
            1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            # Waist 13-15
            0.0,
            0.0,
            0.0,
            # # 3/20 팔 제거
            # # Left arm 16-22
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            # # Right arm (roll mirrored) 23-28
            1.0,
            -1.0,
            -1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        device=env.unwrapped.device,
    )
    tb_leftleg = [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0]
    tb_rightleg = [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0]
    tb_waist = [0.0, 0.0, 0.0]
    tb_leftarm = [0.3, 0.25, 0.0, 0.97, 0.15, 0.0, 0.0]
    tb_rightarm = [0.3, 0.25, 0.0, 0.97, 0.15, 0.0, 0.0]
    trajectory_bias = torch.tensor(
        tb_leftleg + tb_rightleg + tb_waist
        # # 3/20 팔 제거
        + tb_leftarm + tb_rightarm,
        device=env.unwrapped.device,
    )
    enabled_groups = args_cli.joints
    # Per-group trajectory scales (non-zero = excited)
    scale_left_leg = (
        [0.35, 0.05, 0.3, 0.3, 0.15, -0.1]
        if "left_leg" in enabled_groups
        else [0.0] * 6
    )
    scale_right_leg = (
        [0.35, 0.05, 0.3, 0.3, 0.15, -0.1]
        if "right_leg" in enabled_groups
        else [0.0] * 6
    )
    scale_waist = [0.3, 0.2, 0.3] if "waist" in enabled_groups else [0.0] * 3
    scale_left_arm = (
        [0.5, 0.1, 0.1, 0.5, 0.15, 0.1, 0.1]
        if "left_arm" in enabled_groups
        else [0.0] * 7
    )
    scale_right_arm = (
        [0.5, 0.1, 0.1, 0.5, 0.15, 0.1, 0.1]
        if "right_arm" in enabled_groups
        else [0.0] * 7
    )
    trajectory_scale = torch.tensor(
        scale_left_leg + scale_right_leg + scale_waist
        # # 3/20 팔 제거
        + scale_left_arm + scale_right_arm,
        device=env.unwrapped.device,
    )
    # Mask for which joints are active (scale > 0) — used to filter saved data
    active_mask = trajectory_scale != 0
    active_indices = torch.where(active_mask)[0]
    active_joint_names = [joint_order[i] for i in active_indices.tolist()]
    print(f"[INFO]: Active joint groups: {enabled_groups}")
    print(f"[INFO]: Active joints: {active_joint_names}")
    #              초기 위치 용도 (joint limit 고려)        sin wave amplitude
    # trajectory = (trajectory + trajectory_bias) * trajectory_scale

    # # 관절의 초기 위치 (sin 파의 중심)을 0으로 설정 -> 하드웨어 안전을 위해 주석처리
    # trajectory_bias = torch.zeros_like(trajectory_bias)
    # trajectory_scale = torch.ones_like(trajectory_scale)

    # # bias (초기위치) 반영을 위해서 스케일을 먼저 곱하도록 수정
    # trajectory[:, :] = (
    #     (trajectory[:, :] + trajectory_bias.unsqueeze(0))
    #     * trajectory_directions.unsqueeze(0)
    #     * trajectory_scale.unsqueeze(0)
    # )
    trajectory[:, :] = (
        trajectory[:, :] * trajectory_scale.unsqueeze(0) + trajectory_bias.unsqueeze(0)
    ) * trajectory_directions.unsqueeze(0)

    init_pos = torch.zeros((1, articulation.num_joints), device=env.unwrapped.device)
    # joint ids 순서대로 init_pose 삽입 + joint bias
    init_pos[0, joint_ids] = trajectory[0, :] + bias[0]
    articulation.write_joint_position_to_sim(init_pos)
    articulation.write_joint_velocity_to_sim(
        torch.zeros((1, articulation.num_joints), device=env.unwrapped.device)
    )
    counter = 0
    # simulate environment
    dof_pos_buffer = torch.zeros(num_steps, len(joint_ids), device=env.unwrapped.device)
    dof_target_pos_buffer = torch.zeros(
        num_steps, len(joint_ids), device=env.unwrapped.device
    )
    time_data = t

    torch.save(articulation.root_physx_view.get_inertias(), data_dir / "inertia")
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute actions
            dof_pos_buffer[counter, :] = (
                env.unwrapped.scene.articulations["robot"].data.joint_pos[0, joint_ids]
                - bias[0]
            )
            actions = torch.zeros(
                (env.unwrapped.num_envs, articulation.num_joints),
                device=env.unwrapped.device,
            )
            actions[:, joint_ids] = trajectory[counter % num_steps, :].unsqueeze(0)
            # apply actions
            obs, _, _, _, _ = env.step(actions)
            dof_target_pos_buffer[counter, :] = env.unwrapped.scene.articulations[
                "robot"
            ]._data.joint_pos_target[0, joint_ids]
            counter += 1
            if counter % 400 == 0:
                print(f"[INFO]: Step {counter/sample_rate} seconds")
            if counter >= num_steps:
                break

    # close the simulator
    env.close()

    from time import sleep

    sleep(1)  # wait a bit for everything to settle
    if not test:
        (data_dir).mkdir(parents=True, exist_ok=True)
        # Save only active joints
        torch.save(
            {
                "time": time_data.cpu(),
                "dof_pos": dof_pos_buffer[:, active_indices].cpu(),
                "des_dof_pos": dof_target_pos_buffer[:, active_indices].cpu(),
                "joint_names": active_joint_names,
            },
            data_dir / "chirp_data.pt",
        )

        import matplotlib.pyplot as plt

        for idx, i in enumerate(active_indices.tolist()):
            plt.figure()
            plt.plot(
                t.cpu().numpy(),
                dof_pos_buffer[:, i].cpu().numpy(),
                label=f"{joint_order[i]} pos",
            )
            plt.plot(
                t.cpu().numpy(),
                dof_target_pos_buffer[:, i].cpu().numpy(),
                label=f"{joint_order[i]} target",
                linestyle="dashed",
            )
            plt.title(f"Joint {joint_order[i]} Trajectory")
            plt.xlabel("Time [s]")
            plt.ylabel("Joint position [rad]")
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
