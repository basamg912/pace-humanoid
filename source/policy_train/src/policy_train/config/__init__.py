import gymnasium as gym

from .g1_env_cfg import G1LowLevelEnvCfg, G1LowLevelEnvCfg_PLAY
from .g1_ppo_cfg import G1PPORunnerCfg

gym.register(
    id="Isaac-G1-LowLevel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1LowLevelEnvCfg,
        "rsl_rl_cfg_entry_point": G1PPORunnerCfg,
    },
)

gym.register(
    id="Isaac-G1-LowLevel-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1LowLevelEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": G1PPORunnerCfg,
    },
)
