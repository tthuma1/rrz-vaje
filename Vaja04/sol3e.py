import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import workspace_utils
import time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
np.set_printoptions(precision=2)

import ikpy.chain
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from ikpy.utils import geometry


# ---------------- ROBOT SETUP ---------------- #

JOINT_NAMES = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
    'gripper',
]

PICK_Z_ABOVE = 0.12
PICK_Z_GRASP = 0.04
PLACE_POS = np.array([0.1, 0.2, 0.2])


URDF_PATH = 'so101_new_calib.urdf'
my_chain = ikpy.chain.Chain.from_urdf_file(URDF_PATH)
my_chain.active_links_mask[0] = False

port = "/dev/arm_f4"
calibration_dir = "calibrations/"
robot_config = SO101FollowerConfig(
    port=port,
    id="arm_f4",
    calibration_dir=Path(calibration_dir)
)

robot = SO101Follower(robot_config)
robot.connect()
robot.bus.disable_torque()

for j in JOINT_NAMES:
    robot.bus.write("Goal_Velocity", j, 500)
    robot.bus.write("Acceleration", j, 10)

# ---------------- CAMERA SETUP ---------------- #

w, h = 1600, 1200
camera_id = 0

calib = np.load("wide_calibration_data.npz")
M = calib["camera_matrix"]
D = calib["dist_coeffs"]

cap = cv2.VideoCapture(camera_id)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

if not cap.isOpened():
    raise RuntimeError("Camera not opened")

# ---------------- HOMOGRAPHY INIT ---------------- #

ret, frame = cap.read()
if not ret:
    raise RuntimeError("Failed to grab initial frame")

undistorted = cv2.undistort(frame, M, D)
rotated = cv2.rotate(undistorted, cv2.ROTATE_180)
im = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)

corners = workspace_utils.get_workspace_corners(im, draw_markers=True)
H1, H2 = workspace_utils.calculate_homography_mapping(corners)

# workspace mask
points = np.load("tocke2.npy").reshape(4, 2)
x_min, x_max = int(points[:,0].min()), int(points[:,0].max())
y_min, y_max = int(points[:,1].min()), int(points[:,1].max())

# HSV blue limits
lower = np.array([90, 110, 30])
upper = np.array([140, 255, 255])
kernel = np.ones((11,11), np.uint8)

# ---------------- LIVE PLOT ---------------- #

plt.ion()
fig, ax = plt.subplots(figsize=(6,6))

last_move = time.time()


#cv2.namedWindow('neki', cv2.WINDOW_NORMAL)
#while True:
    #cv2.imshow('neki', im)

    #if cv2.waitKey(1) & 0xFF == ord("q"):
        #break


# ---------------- MAIN LOOP ---------------- #

def move_robot(target_position, gripper_open=True):
    target_orientation = geometry.rpy_matrix(
        0, np.deg2rad(180), 0
    )

    ik = my_chain.inverse_kinematics(
        target_position,
        target_orientation,
        'all',
        optimizer='scalar'
    )

    # gripper control
    ik[-1] = 90 if gripper_open else 0

    action = {
        JOINT_NAMES[i] + '.pos': np.rad2deg(v)
        for i, v in enumerate(ik[1:])
    }

    robot.send_action(action)
    time.sleep(0.6)


while True:

    ret, frame = cap.read()
    if not ret:
        break

    undistorted = cv2.undistort(frame, M, D)
    rotated = cv2.rotate(undistorted, cv2.ROTATE_180)
    im = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)

    warped = cv2.warpPerspective(im, H1, (1000, 1000))

    workspace_mask = np.zeros(warped.shape[:2], dtype=np.uint8)
    workspace_mask[y_min:y_max, x_min:x_max] = 255
    masked = cv2.bitwise_and(warped, warped, mask=workspace_mask)

    hsv = cv2.cvtColor(masked, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    closed, connectivity=4
)

# ignore background, sort by area (largest first)
objects = [
    (i, stats[i, cv2.CC_STAT_AREA])
    for i in range(1, num_labels)
    if stats[i, cv2.CC_STAT_AREA] > 300  # area threshold
]

objects.sort(key=lambda x: x[1], reverse=True)

for i, _ in objects:

    cx, cy = centroids[i]
    cx, cy = int(cx), int(cy)

    # visualize
    ax.plot(cx, cy, 'ro', markersize=6)

    # --- map to robot coords ---
    pt = np.array([cx, cy, 1.0])
    mapped = H2 @ pt
    mapped /= mapped[2]

    Xr, Yr = mapped[0], mapped[1]

    # ---------------- PICK ---------------- #

    # above object
    move_robot(
        np.array([Xr, Yr, PICK_Z_ABOVE]),
        gripper_open=True
    )

    # down to grasp
    move_robot(
        np.array([Xr, Yr, PICK_Z_GRASP]),
        gripper_open=True
    )

    # close gripper
    move_robot(
        np.array([Xr, Yr, PICK_Z_GRASP]),
        gripper_open=False
    )

    # lift
    move_robot(
        np.array([Xr, Yr, PICK_Z_ABOVE]),
        gripper_open=False
    )

    # ---------------- PLACE ---------------- #

    move_robot(
        PLACE_POS,
        gripper_open=False
    )

    # open gripper
    move_robot(
        PLACE_POS,
        gripper_open=True
    )

    time.sleep(0.5)

        #if time.time() - last_move > 0.1:
            #target = np.array([Xr, Yr, 0.08])
            #ik = my_chain.inverse_kinematics(target, optimizer="scalar")

            #action = {
                #JOINT_NAMES[i] + ".pos": np.rad2deg(v)
                #for i, v in enumerate(ik[1:])
            #}

            #robot.send_action(action)
            #last_move = time.time()

    #plt.pause(0.001)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------- CLEANUP ---------------- #

cap.release()
plt.ioff()
plt.close()
cv2.destroyAllWindows()
