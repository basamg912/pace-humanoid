import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CommandTermCfg as CommandTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.utils.noise import UniformNoiseCfg

from pace_sim2real.tasks.manager_based.pace.G1_CFG import (
    UNITREE_G1_29DOF_CFG,
    G1_12DOF_CFG,
)
from policy_train.mdp import *  # noqa: F401, F403
import policy_train.mdp as mdp

from pace_sim2real.utils import PaceDCMotorCfg, PaceDCMotor

##
# Scene
##


def read_params(path: str):
    import torch

    mean = torch.load(path)
    n = 12
    armature = mean[
        :n
    ].tolist()  # joint_order: [lhp, lhr, lhy, lk, lap, lar, rhp, rhr, rhy, rk, rap, rar]
    damping = mean[n : 2 * n].tolist()  # viscous friction (Nm·s/rad)
    friction = mean[2 * n : 3 * n].tolist()  # Coulomb friction
    bias = mean[3 * n : 4 * n].tolist()  # encoder bias (rad)
    delay = int(mean[4 * n].item())  # delay in sim steps

    return armature, damping, friction, bias, delay


# joint_order index reference (12 leg joints from CMA-ES fitting):
# 0:lhp  1:lhr  2:lhy  3:lk  4:lap  5:lar
# 6:rhp  7:rhr  8:rhy  9:rk 10:rap 11:rar
_armature, _damping, _friction, _bias, _delay = read_params(
    "/home/kist/work/workspace/pace-humanoid/logs/pace/Robot/26_03_25_13-42-39/mean_2420.pt"
)


@configclass
class G1SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    env_spacing: float = 2.5
    num_envs: int = 4096

    robot: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.8),
            joint_pos={
                ".*_hip_pitch_joint": -0.312,
                ".*_knee_joint": 0.669,
                ".*_ankle_pitch_joint": -0.363,
                ".*_shoulder_pitch_joint": 0.2,
                "left_shoulder_roll_joint": 0.2,
                "right_shoulder_roll_joint": -0.2,
                ".*_elbow_joint": 0.6,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            # ===== N7520-14.3 =====
            "N7520-14.3_hips": PaceDCMotorCfg(
                joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*"],
                effort_limit=88,
                velocity_limit=32.0,
                stiffness={".*_hip_.*": 100.0},
                damping={".*_hip_.*": 2.0},
                saturation_effort=120,
                armature=sum([_armature[i] for i in [0, 2, 6, 8]]) / 4,
                encoder_bias=sum([_bias[i] for i in [0, 2, 6, 8]]) / 4,
                max_delay=_delay,
            ),
            "N7520-14.3_waist": PaceDCMotorCfg(
                joint_names_expr=["waist_yaw_joint"],
                effort_limit=88,
                velocity_limit=32.0,
                stiffness={"waist_yaw_joint": 200.0},
                damping={"waist_yaw_joint": 5.0},
                saturation_effort=120,
                armature=0.01,
                encoder_bias=0.0,
                max_delay=_delay,
            ),
            # ===== N7520-22.5 =====
            "N7520-22.5_legs": PaceDCMotorCfg(
                joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],
                effort_limit=139,
                velocity_limit=20.0,
                stiffness={".*_hip_roll_.*": 100.0, ".*_knee_.*": 150.0},
                damping={".*_hip_roll_.*": 2.0, ".*_knee_.*": 4.0},
                saturation_effort=160,
                armature=sum([_armature[i] for i in [1, 3, 7, 9]]) / 4,
                encoder_bias=sum([_bias[i] for i in [1, 3, 7, 9]]) / 4,
                max_delay=_delay,
            ),
            # ===== N5020-16 =====
            "N5020-16_ankles": PaceDCMotorCfg(
                joint_names_expr=[".*_ankle_.*"],
                effort_limit=25,
                velocity_limit=37,
                stiffness=40.0,
                damping={".*_ankle_.*": 2.0},
                saturation_effort=30,
                armature=sum([_armature[i] for i in [4, 5, 10, 11]]) / 4,
                encoder_bias=sum([_bias[i] for i in [4, 5, 10, 11]]) / 4,
                max_delay=_delay,
            ),
            "N5020-16_arms": PaceDCMotorCfg(
                joint_names_expr=[
                    ".*_shoulder_.*",
                    ".*_elbow_.*",
                    ".*_wrist_roll.*",
                    "waist_roll_joint",
                    "waist_pitch_joint",
                ],
                effort_limit=25,
                velocity_limit=37,
                stiffness=40.0,
                damping={
                    ".*_shoulder_.*": 1.0,
                    ".*_elbow_.*": 1.0,
                    ".*_wrist_roll.*": 1.0,
                    "waist_.*_joint": 5.0,
                },
                saturation_effort=30,
                armature=0.01,
                encoder_bias=0.0,
                max_delay=_delay,
            ),
            # ===== W4010-25 =====
            "W4010-25_arms": PaceDCMotorCfg(
                joint_names_expr=[".*_wrist_pitch.*", ".*_wrist_yaw.*"],
                effort_limit=5,
                velocity_limit=22,
                stiffness=40.0,
                damping=1.0,
                saturation_effort=6,
                armature=0.01,
                encoder_bias=0.0,
                max_delay=0,
            ),
        },
    )
    # contact sensor 를 링크에 붙이고, 종료/보상 config term 에서 필터링

    contact_sensor = ContactSensorCfg(
        # 감시할 모든 링크를 포함해야 함: pelvis, torso, 그리고 양발(ankle)
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,  # feet_air_time 계산을 위해 필수
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# Commands
##


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.5,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.05, 0.05),
            heading=(-3.14, 3.14),
        ),
    )


##
# Actions
##


# 모든 joint 에 대해서 action 을 주입
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True,
    )


##
# Observations
##


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5)
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# Events (domain randomization)
##


@configclass
class EventsCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0, 0),
                "y": (0, 0),
                "z": (0, 0),
                "roll": (0, 0),
                "pitch": (0, 0),
                "yaw": (0, 0),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )


##
# Rewards
##


@configclass
class RewardsCfg:
    # Tracking
    lin_vel_xy = RewTerm(func=mdp.lin_vel_tracking_xy, weight=1.0)
    ang_vel_z = RewTerm(func=mdp.ang_vel_tracking_z, weight=1.0)
    # Base stability
    lin_vel_z = RewTerm(func=mdp.lin_vel_penalty_z, weight=-0.1)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_penalty_xy, weight=-0.05)
    # base_height = RewTerm(
    #     func=mdp.base_height_l2, weight=-10.0, params={"target_height": 0.78}
    # )
    is_alive = RewTerm(func=mdp.is_alive, weight=10.0)
    # Regularization
    # # 값의 스케일 자체가 크기 때문에, L2 노름 계산에서 weight를 작게 설정
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    terminated = RewTerm(
        func=mdp.is_terminated,
        weight=-300.0,
    )
    # penalize, robot's joint position - joint default position
    joint_dev_arm = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                name="robot",
                joint_names=[
                    ".*_shoulder_.*",
                    ".*_elbow_.*",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    joint_dev_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                name="robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )
    joint_dev_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"]
            )
        },
    )
    # undesired_contacts 는 Isaac Lab 기본 제공
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=["pelvis", ".*torso.*"]
            ),
            "threshold": 1.0,
        },
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    # Joint limits
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    # Gait
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.05,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
            ),
            "threshold": 0.2,
        },
    )


##
# Terminations
##


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.3}
    )


##
# Environment
##


@configclass
class G1LowLevelEnvCfg(ManagerBasedRLEnvCfg):
    scene: G1SceneCfg = G1SceneCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Simulation
        self.sim.dt = 0.005  # 200Hz physics
        self.decimation = 4  # 50Hz policy
        self.episode_length_s = 20.0
        # Viewer
        self.viewer.lookat = (0.0, 0.0, 0.76)
        self.viewer.eye = (3.0, 3.0, 2.0)
        # Rendering


@configclass
class G1LowLevelEnvCfg_PLAY(G1LowLevelEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        # Disable domain randomizations for play
        self.observations.policy.enable_corruption = False
