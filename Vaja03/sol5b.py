
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


    offset = np.array([0, 0, 0])

    # rectangle 1
    #pts = generate_rectangle(N=60)
    #pts += offset




    # rectangle 2
    #width = 0.1
    #height = 0.05
    #target_position=np.array([0.2, 0.0, 0.05])

    #p1 = target_position
    #p2 = target_position + [0, width, 0]
    #p3 = target_position + [height, width, 0]
    #p4 = target_position + [height, 0, 0]

    #pts = [p1,p2,p3,p4]




    # circle
    pts = generate_circle(N=60, radius = 0.04)

    while True:
        for pt in pts:
            #target_orientation = np.eye(3)
            #target_orientation[2,2] = -1 # obrne Z-os
            #ik = my_chain.inverse_kinematics(pt, target_orientation, 'all', optimizer='scalar')

            ik = my_chain.inverse_kinematics(pt, optimizer='scalar')
            action = {JOINT_NAMES[i]+'.pos': np.rad2deg(v) for i, v in enumerate(ik[1:])}

            robot.send_action(action)
            #time.sleep(0.01)
            #time.sleep(2)

if __name__=='__main__':
    main()
