import os

import isaaclab.sim as sim_utils
from pace_sim2real.utils import PaceDCMotorCfg
from pace_sim2real import PaceSim2realEnvCfg, PaceSim2realSceneCfg, PaceCfg
from pace_sim2real.tasks.manager_based.pace.G1_CFG import UNITREE_G1_29DOF_CFG
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

import torch

UNITREE_MODEL_DIR = "path/to/unitree_model"  # Replace with the actual path to your unitree_model directory
UNITREE_ROS_DIR = "/home/kist/work/workspace/unitree_rl_lab/unitree_ros"  # Replace with the actual path to your unitree_ros package

# n7520-14.3
# 2 + 2 + 1 = 5
G1_PACE_ACTUATOR1_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*", "waist_yaw_joint"],
    effort_limit=88,
    velocity_limit=32.0,
    stiffness={
        ".*_hip_.*": 100.0,
        "waist_yaw_joint": 200.0,
    },
    damping={
        ".*_hip_.*": 2.0,
        "waist_yaw_joint": 5.0,
    },
    armature=0.01,
    saturation_effort=120,
    encoder_bias=[0.0] * 5,
    max_delay=10,
)

# n7520-22.5
# 2 + 2 = 4
G1_PACE_ACTUATOR2_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],
    effort_limit=139,
    velocity_limit=20.0,
    stiffness={
        ".*_hip_roll_.*": 100.0,
        ".*_knee_.*": 150.0,
    },
    damping={
        ".*_hip_roll_.*": 2.0,
        ".*_knee_.*": 4.0,
    },
    armature=0.01,
    saturation_effort=160,
    encoder_bias=[0.0] * 4,
    max_delay=10,
)
# n5020-16
# 6 + 2 + 2 + 4 + 2 = 16
G1_PACE_ACTUATOR3_CFG = PaceDCMotorCfg(
    joint_names_expr=[
        # # 3/20 팔 제거
        ".*_shoulder_.*",
        ".*_elbow_.*",
        ".*_wrist_roll.*",
        ".*_ankle_.*",
        "waist_roll_joint",
        "waist_pitch_joint",
    ],
    effort_limit=25,
    velocity_limit=37,
    stiffness=40.0,
    damping={
        # # 3/20 팔 제거
        ".*_shoulder_.*": 1.0,
        ".*_elbow_.*": 1.0,
        ".*_wrist_roll.*": 1.0,
        ".*_ankle_.*": 2.0,
        "waist_.*_joint": 5.0,
    },
    armature=0.01,
    saturation_effort=30,
    encoder_bias=[0.0] * 16,
    max_delay=10,
)
# w4010-25
# 4
# # 3.20 팔 제거
G1_PACE_ACTUATOR4_CFG = PaceDCMotorCfg(
    joint_names_expr=[".*_wrist_pitch.*", ".*_wrist_yaw.*"],
    effort_limit=5,
    velocity_limit=22,
    stiffness=40.0,
    damping=1.0,
    armature=0.01,
    saturation_effort=6,
    encoder_bias=[0.0] * 4,
    max_delay=10,
)


@configclass
class UnitreeUsdFileCfg(sim_utils.UsdFileCfg):
    activate_contact_sensors: bool = True
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )


@configclass
class UnitreeUrdfFileCfg(sim_utils.UrdfFileCfg):
    fix_base: bool = False
    activate_contact_sensors: bool = True
    replace_cylinders_with_capsules = True
    joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=0, damping=0
        )
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )

    def replace_asset(self, meshes_dir, urdf_path):
        """Replace the asset with a temporary copy to avoid modifying the original asset.

        When need to change the collisions, place the modified URDF file separately in this repository,
        and let `meshes_dir` be provided by `unitree_ros`.
        This function will auto construct a complete `robot_description` file structure in the `/tmp` directory.
        Note: The mesh references inside the URDF should be in the same directory level as the URDF itself.
        """
        tmp_meshes_dir = "/tmp/IsaacLab/unitree_rl_lab/meshes"
        if os.path.exists(tmp_meshes_dir):
            os.remove(tmp_meshes_dir)
        os.makedirs("/tmp/IsaacLab/unitree_rl_lab", exist_ok=True)
        os.symlink(meshes_dir, tmp_meshes_dir)

        self.asset_path = "/tmp/IsaacLab/unitree_rl_lab/robot.urdf"
        if os.path.exists(self.asset_path):
            os.remove(self.asset_path)
        os.symlink(urdf_path, self.asset_path)


@configclass
class G1PaceCfg(PaceCfg):
    robot_name = "Robot"
    data_dir = "Robot/chirp_data.pt"
    bounds_params: torch.Tensor = torch.zeros(
        # 29 x 4 + 1 = 117
        (117, 2)
    )
    joint_order: list[str] = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        # # 3/20 팔 제거
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

    def __post_init__(self):
        # set bounds for parameters
        self.bounds_params[:15, 0] = 1e-5
        self.bounds_params[:15, 1] = 1.0  # armature between 1e-5 - 1.0 [kgm2]
        self.bounds_params[15:30, 1] = 7.0  # dof_damping between 0.0 - 7.0 [Nm s/rad]
        self.bounds_params[30:45, 1] = 0.5  # friction between 0.0 - 0.5
        self.bounds_params[45:60, 0] = -0.1
        self.bounds_params[45:60, 1] = 0.1  # bias between -0.1 - 0.1 [rad]
        self.bounds_params[116, 1] = 10.0  # delay between 0.0 - 10.0 [sim steps]


@configclass
class G1PaceSceneCfg(PaceSim2realSceneCfg):
    robot = UNITREE_G1_29DOF_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        actuators={
            "N7520-14.3": G1_PACE_ACTUATOR1_CFG,
            "N7520-22.5": G1_PACE_ACTUATOR2_CFG,
            "N5020-16": G1_PACE_ACTUATOR3_CFG,
            # # 3/20 팔 제거 사용 안함
            "W4010-25": G1_PACE_ACTUATOR4_CFG,
        },
    )


@configclass
class G1PaceEnvCfg(PaceSim2realEnvCfg):
    scene: G1PaceSceneCfg = G1PaceSceneCfg()
    sim2real: PaceCfg = G1PaceCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # robot sim and control settings
        self.sim.dt = 0.0025  # 400Hz simulation
        self.decimation = 1  # 400Hz control
