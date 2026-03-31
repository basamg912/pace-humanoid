import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="link 물성치")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from pace_sim2real.tasks.manager_based.G1_CFG import UNITREE_G1_29DOF_CFG
