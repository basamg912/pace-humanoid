import torch
import matplotlib.pyplot as plt

data = torch.load("data/Robot/chirp_data.pt")  # adjust path to your actual file

time = data["time"].numpy()
dof_pos = data["dof_pos"].numpy()
des_dof_pos = data["des_dof_pos"].numpy()

joint_names = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]

for i in range(dof_pos.shape[1]):
    plt.figure()
    plt.plot(time, dof_pos[:, i], label="actual")
    plt.plot(time, des_dof_pos[:, i], label="target", linestyle="dashed")
    plt.title(joint_names[i])
    plt.xlabel("Time [s]")
    plt.ylabel("Position [rad]")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
