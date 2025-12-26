from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from pathlib import Path

arm_name = 'arm_f4'
port = f'/dev/{arm_name}'
calibration_dir='calibrations'
robot_config = SO101FollowerConfig(port=port, id=arm_name,
calibration_dir=Path(calibration_dir))
robot = SO101Follower(robot_config)

robot.connect() # povezava na robota
robot.bus.disable_torque() # ugasnemo motorje

while True:
    current_obs = robot.get_observation()
    print(f'{current_obs=}')
robot.disconnect()
