import warnings
warnings.filterwarnings("ignore")

import math, time
import numpy as np
np.set_printoptions(precision=2)
from pathlib import Path

import ikpy.chain
from ikpy.inverse_kinematics import inverse_kinematic_optimization
from ikpy.utils import geometry

import numpy as np

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

CUBE_SIZE = 0.019
NUM_CUBES = 4

PICK_XY = np.array([0.25, 0])
PLACE_XY = np.array([0.2, 0.1])

APPROACH_Z = 0.14 # safe height above cubes
GRASP_OFFSET = 0.02 # višina na kateri bomo zgrabili kocko

def send_pose(chain, robot, pt, gripper):
	# target_orientation = np.eye(3)
	# target_orientation[2,2] = -1 # Z-os obrnemo dol z rotacijo okrog Y - pri tem se tudi X-os obrne
	# target_orientation[0,0] = -1
	target_orientation = geometry.rpy_matrix(0, np.deg2rad(180), 0)  # point down
	ik = chain.inverse_kinematics(pt, target_orientation, 'all', optimizer='scalar')
 
	# ik = chain.inverse_kinematics(pt, optimizer='scalar')

	ik[-1] = gripper  # 1=open, 0=closed, 0.5 je na pol odprt

	action = {
		JOINT_NAMES[i] + '.pos': np.rad2deg(v)
		for i, v in enumerate(ik[1:])
	}
	robot.send_action(action)

def move_linear(chain, robot, start, end, steps=20, gripper=1.0):
	# linearno interpoliraj od start do end v `steps` korakih
	for t in np.linspace(0, 1, steps):
		pt = (1 - t) * start + t * end
		send_pose(chain, robot, pt, gripper)

def main():
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

	for i in range(NUM_CUBES):
		pick_z = GRASP_OFFSET
		place_z = GRASP_OFFSET + i * CUBE_SIZE

		pick_above  = np.array([*PICK_XY, APPROACH_Z])
		pick_grasp  = np.array([*PICK_XY, pick_z])

		place_above = np.array([*PLACE_XY, APPROACH_Z])
		place_grasp = np.array([*PLACE_XY, place_z])

		# move above cube
		move_linear(my_chain, robot, pick_above, pick_above, gripper=0.5)

		# descend to pick
		move_linear(my_chain, robot, pick_above, pick_grasp, gripper=0.5)
		time.sleep(1)

		# close gripper
		send_pose(my_chain, robot, pick_grasp, gripper=0)
		time.sleep(1)

		# lift cube
		move_linear(my_chain, robot, pick_grasp, pick_above, gripper=0)

		# move above stack
		move_linear(my_chain, robot, pick_above, place_above, gripper=0)

		# descend to place on stack
		move_linear(my_chain, robot, place_above, place_grasp, gripper=0)
		time.sleep(1)

		# open gripper
		send_pose(my_chain, robot, place_grasp, gripper=0.5)
		time.sleep(1)

		# move above stack
		move_linear(my_chain, robot, place_grasp, place_above, gripper=0.5)

		# wait for next cube
		time.sleep(2)

if __name__=='__main__':
	main()
