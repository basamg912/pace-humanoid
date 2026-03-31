import torch

params = torch.load(
    "/home/kist/work/workspace/pace-humanoid/logs/pace/Robot/26_03_25_13-42-39/mean_2420.pt"
)
joint_order = torch.load("logs/pace/Robot/26_03_25_13-42-39/config.pt")["joint_order"]

n = 12
damping = params[n : 2 * n]  # viscous friction (Nm·s/rad)
friction = params[2 * n : 3 * n]  # Coulomb friction


def g1_set_joint_params(articulation, num_envs, device):
    print(f"[INFO] {num_envs}")
    _damping = damping.to(device).unsqueeze(0).repeat(num_envs, 1)
    _friction = friction.to(device).unsqueeze(0).repeat(num_envs, 1)
    joint_ids = torch.tensor(
        [articulation.joint_names.index(name) for name in joint_order], device=device
    )
    articulation.write_joint_viscous_friction_coefficient_to_sim(
        _damping, joint_ids=joint_ids, env_ids=torch.arange(num_envs)
    )
    articulation.data.default_joint_viscous_friction_coeff[:, joint_ids] = _damping
    articulation.write_joint_friction_coefficient_to_sim(
        _friction, joint_ids=joint_ids, env_ids=torch.arange(num_envs)
    )
    articulation.data.default_joint_friction_coeff[:, joint_ids] = _friction
