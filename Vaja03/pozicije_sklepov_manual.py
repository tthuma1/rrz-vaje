from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from pathlib import Path
import time
import numpy as np

arm_name = 'arm_f4'
port = f'/dev/{arm_name}'
calibration_dir='calibrations'
robot_config = SO101FollowerConfig(port=port, id=arm_name,
calibration_dir=Path(calibration_dir))
robot = SO101Follower(robot_config)

robot.connect() # povezava na robota
robot.bus.disable_torque() # ugasnemo motorje

action = {
    'shoulder_pan.pos': 90,
    # 'shoulder_pan.pos': 45,
    # 'shoulder_pan.pos': 180,

    # 'gripper.pos': 1.0,
    'gripper.pos': np.pi/2,
    # 'gripper.pos': 0.5,
    # 'gripper.pos': np.pi,
}

robot.send_action(action)

time.sleep(2)

robot.disconnect()
