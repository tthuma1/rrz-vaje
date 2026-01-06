import cv2
import matplotlib.pyplot as plt
import numpy as np
import workspace_utils
import time

np.set_printoptions(precision=2)
from pathlib import Path

import ikpy.chain
from ikpy.inverse_kinematics import inverse_kinematic_optimization
from ikpy.utils import geometry

# from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

from utils import *

# --------------- begin robot config
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
# --------------- end robot config

last_point = None

def save_homography(im):
    try:
        corners = workspace_utils.get_workspace_corners(im, draw_markers=True)
        H1, H2 = workspace_utils.calculate_homography_mapping(corners)
    except Exception:
        print("Can't read April tags.")
        return None, None

    np.savez('homography_3b.npz', H1=H1, H2=H2)
    return H1, H2

def load_homography():
    data = np.load('homography_3b.npz')
    return data['H1'], data['H2']

def send_robot_to_point(pt_im):
    pt = H2 @ pt_im
    pt /= pt[2]
    # print(pt)

    # dobili smo x in y; z koordinato bomo hardcodali na 0.08
    pt[2] = 0.08

    # ----- begin send to robot
    ik = my_chain.inverse_kinematics(pt, optimizer='scalar')
    action = {JOINT_NAMES[i]+'.pos': np.rad2deg(v) for i, v in enumerate(ik[1:])}

    robot.send_action(action)
    # ----- end send to robot


# H1, H2 = load_homography()
H1 = H2 = None

w, h = 1600, 1200
camera_id = 0

calib = np.load("wide_calibration_data.npz")
M = calib["camera_matrix"]
D = calib["dist_coeffs"]

cap = cv2.VideoCapture(camera_id)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

# HSV blue limits
lower = np.array([90, 110, 30])
upper = np.array([140, 255, 255])
kernel = np.ones((11,11), np.uint8)

while True:
    # Poskusimo pridobiti trenutno sliko s spletne kamere
    ret, frame = cap.read()

    # Če to ni mogoče (kamera izključena, itd.), končamo z izvajanjem funkcije
    if not ret:
        break

    undistorted = cv2.undistort(frame, M, D)
    im = cv2.rotate(undistorted, cv2.ROTATE_180)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # če rabiš testing "brez kamere", odkomentiraj to
    # ------ start test
    # im_path = 'capture_f1.jpg'
    # im = cv2.imread(im_path)
    # ------ end test

    if H1 is None or H2 is None:
        H1, H2 = save_homography(im)

    if H1 is not None:
        im = cv2.warpPerspective(im, H1, (1000, 1000))
    
    if H2 is not None:
        # grid mask - smiselno je iskati objekte samo znotraj grida, ker je sicer okoli dosti šuma (drugih objektov)
        points = np.load("tocke2.npy").reshape(4, 2)
        x_min, x_max = int(points[:,0].min()), int(points[:,0].max())
        y_min, y_max = int(points[:,1].min()), int(points[:,1].max())

        grid_mask = np.zeros(im.shape[:2], dtype=np.uint8)
        grid_mask[y_min:y_max, x_min:x_max] = 255
        im_grid = cv2.bitwise_and(im, im, mask=grid_mask)

        hsv = cv2.cvtColor(im_grid, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            closed, connectivity=4
        )

        if num_labels > 1:
            cx, cy = centroids[1]
            cx, cy = int(cx), int(cy)
            send_robot_to_point(np.array([cx, cy, 1.0]))

            robo_point = H2 @ np.array([cx, cy, 1.0])
            robo_point /= robo_point[2]

            cv2.circle(im, (cx, cy), 5, (255, 0, 0), -1)

            text = f"{robo_point[0]:.2f}, {robo_point[1]:.2f}"
            # Draw black outline (thicker)
            cv2.putText(
                im, text, (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 3, cv2.LINE_AA
            )

            # Draw white text on top
            cv2.putText(
                im, text, (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA
            )

    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imshow('frame', im)
    time.sleep(0.01)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


