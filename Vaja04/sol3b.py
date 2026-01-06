import cv2
import matplotlib.pyplot as plt
import numpy as np
import workspace_utils
import matplotlib.patheffects as pe

import warnings
warnings.filterwarnings("ignore")

import math, time
np.set_printoptions(precision=2)
from pathlib import Path

import ikpy.chain
from ikpy.inverse_kinematics import inverse_kinematic_optimization
from ikpy.utils import geometry

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

from utils import *

JOINT_NAMES = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
    'gripper',
]

URDF_PATH = 'so101_new_calib.urdf'
my_chain = ikpy.chain.Chain.from_urdf_file(URDF_PATH)
my_chain.active_links_mask[0]=False

# Configure robot
port = "/dev/arm_f4"
# robot_config = SO101FollowerConfig(port=port, id='arm_f1')
calibration_dir='calibrations/'
robot_config = SO101FollowerConfig(port=port, id='arm_f4', calibration_dir=Path(calibration_dir))

robot = SO101Follower(robot_config)
robot.connect()
robot.bus.disable_torque()

# IMPORTANT for setting maximum velocity and acceleration
v = 500
a = 10
for j in JOINT_NAMES:
    robot.bus.write("Goal_Velocity", j, v)
    robot.bus.write("Acceleration", j, a)

# --- Load image ---
ime_slike = 'capture_f1.jpg'
im = cv2.imread(ime_slike)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# --- Compute homographies ---
corners = workspace_utils.get_workspace_corners(im, draw_markers=True)
H1, H2 = workspace_utils.calculate_homography_mapping(corners)

# --- Warp image to workspace (for clicking) ---
delovna_povrsina = cv2.warpPerspective(im, H1, (1000, 1000))

# --- Click points on the warped workspace ---
fig, ax_click = plt.subplots(figsize=(6,6))
ax_click.imshow(delovna_povrsina)
ax_click.set_title("Click points on warped workspace")
clicked_pts = plt.ginput(4, timeout=0)
plt.close(fig)

# --- Prepare inverse of H1 to map clicks back to original image ---
H1_inv = np.linalg.inv(H1)

# --- Plot ONLY the original image and mapped points ---
fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(im)
ax.set_title("Original image with robot coordinates")

for (uw, vw) in clicked_pts:

    # Step 1: warped → original image coordinates
    pt_w = np.array([uw, vw, 1.0])
    orig = H1_inv @ pt_w
    orig /= orig[2]

    u, v = orig[0], orig[1]

    # Step 2: original image → robot coordinates
    # pt = np.array([u, v, 1.0])
    mapped = H2 @ pt_w
    mapped /= mapped[2]

    Xr, Yr = mapped[0], mapped[1]

    pt = np.array([Xr, Yr, 0.05])

    ik = my_chain.inverse_kinematics(pt, optimizer='scalar')
    action = {JOINT_NAMES[i]+'.pos': np.rad2deg(v) for i, v in enumerate(ik[1:])}

    robot.send_action(action)

    time.sleep(1)


    # # Draw point on original image
    # ax.plot(u, v, 'yo', markersize=5)

    # # Label robot coordinates
    # ax.text(
    #     u + 10, v - 10,
    #     f"{Xr:.2f}, {Yr:.2f}",
    #     color='yellow',
    #     fontsize=10,
    #     path_effects=[
    #         pe.Stroke(linewidth=1, foreground='black'),
    #         pe.Normal()
    #     ]
    # )

# plt.show()
