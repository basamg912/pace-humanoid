from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# mdp/terminations.py 에 추가
def illegal_contact_filtered(env, sensor_cfg, threshold):
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    # (num_envs, history_length, num_bodies, 3)

    # body_ids 명시적으로 필터링
    body_ids = sensor_cfg.body_ids  # resolve된 인덱스
    filtered_forces = net_forces[:, :, body_ids, :]

    return torch.any(torch.norm(filtered_forces, dim=-1) > threshold, dim=(1, 2))


def lin_vel_tracking_xy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.25,
) -> torch.Tensor:
    """Reward for tracking desired linear velocity in xy plane."""
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_lin_vel_b[:, :2]
    cmd = env.command_manager.get_command("base_velocity")[:, :2]
    error = torch.sum(torch.square(lin_vel - cmd), dim=1)
    return torch.exp(-error / std**2)


def ang_vel_tracking_z(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.25,
) -> torch.Tensor:
    """Reward for tracking desired angular velocity around z axis."""
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel = asset.data.root_ang_vel_b[:, 2]
    cmd = env.command_manager.get_command("base_velocity")[:, 2]
    error = torch.square(ang_vel - cmd)
    return torch.exp(-error / std**2)


def lin_vel_penalty_z(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize vertical linear velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_penalty_xy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize angular velocity in xy plane (roll/pitch rate)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def joint_torque_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint torques."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1
    )


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(
        1
    )
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


def joint_acc_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint accelerations."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize change in actions between steps."""
    return torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
    )


def base_height_reward(
    env: ManagerBasedRLEnv,
    target_height: float = 0.76,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from target base height."""
    asset: Articulation = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2]
    return torch.square(base_height - target_height)


def feet_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Reward feet air time for encouraging gait."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold).clip(min=0.0), dim=1)
    return reward
