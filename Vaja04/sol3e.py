import cv2
import matplotlib.pyplot as plt
import numpy as np
import workspace_utils
import time
import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(precision=2)
from pathlib import Path

import ikpy.chain
from ikpy.inverse_kinematics import inverse_kinematic_optimization
from ikpy.utils import geometry

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

from utils import *
import threading
import queue

pick_queue = queue.Queue()
robot_busy = threading.Event()

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
# robot.bus.disable_torque()

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

    np.savez('homography_3e.npz', H1=H1, H2=H2)
    return H1, H2

def load_homography():
    data = np.load('homography_3e.npz')
    return data['H1'], data['H2']

def move_robot(pt, gripper, orientation_mode='all', timeout=3):
    target_orientation = geometry.rpy_matrix(0, np.deg2rad(180), 0)  # point down
    ik = my_chain.inverse_kinematics(pt, target_orientation, orientation_mode, optimizer='scalar')
    ik[-1] = gripper
    action = {JOINT_NAMES[i]+'.pos': np.rad2deg(v) for i, v in enumerate(ik[1:])}

    robot.send_action(action)
    time.sleep(timeout)


PICK_Z = 0.03
APPROACH_Z = 0.12
def pick_and_move_block(pt_im):
    pt = H2 @ pt_im
    pt /= pt[2]
    # print(pt)

    # centroid, ki ga vidimo ni čisto točno tam, kjer je kocka - to popravi
    pick_x, pick_y = pt[0], pt[1]
    if pt[1] > 0: # ko gremo na levo stran malo overshoota - moramo zmanjšati x in y
        pick_x += -0.04
        pick_y += -0.01
    else: # ko gremo na desno stran malo overshoota - moramo povečati x in y
        pick_x += -0.05
        pick_y += 0.03

    # move robot above block
    above_pick_pt = np.array([pick_x, pick_y, APPROACH_Z])
    move_robot(above_pick_pt, 0.7)

    # pick cube
    pick_pt = np.array([pick_x, pick_y, PICK_Z])
    move_robot(pick_pt, 0.7)

    # close gripper
    move_robot(pick_pt, 0)

    # go above picked cube
    move_robot(above_pick_pt, 0, timeout=1)
    move_robot(above_pick_pt, 0, orientation_mode=None, timeout=1)

    # go above drop location
    # droppali bomo na istem x, samo y bomo obrnili - gremo iz ene strani na drugo
    above_drop_pt = np.array([pick_x, -pick_y, APPROACH_Z])
    move_robot(above_drop_pt, 0.0, orientation_mode=None)
    move_robot(above_drop_pt, 0.0, timeout=1)

    # place the block and open gripper
    place_pt = np.array([pick_x, -pick_y, PICK_Z])
    move_robot(place_pt, 0.0)
    move_robot(place_pt, 0.7)

    # go above drop location
    move_robot(above_drop_pt, 0.7, timeout=1)
    move_robot(above_drop_pt, 0.7, orientation_mode=None, timeout=1)

def robot_worker():
    while True:
        pt_im = pick_queue.get()   # blocks until task available
        robot_busy.set()
        try:
            pick_and_move_block(pt_im)
        except Exception as e:
            print("Robot error:", e)
        last_point = None
        robot_busy.clear()
        pick_queue.task_done()

def process_blocks(mask, y_condition):
    global last_point
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    for i in range(1, num_labels):
        cx, cy = map(int, centroids[i])
        robo_point = H2 @ np.array([cx, cy, 1.0])
        robo_point /= robo_point[2]

        if robot_busy.is_set():
            continue

        # y_condition decides if block is already on target side
        if not y_condition(robo_point[1]):
            continue

        pick_queue.put(np.array([cx, cy, 1.0]))
        last_point = (cx, cy)


threading.Thread(target=robot_worker, daemon=True).start()

H1, H2 = load_homography()
# H1 = H2 = None

w, h = 1600, 1200
camera_id = 1

calib = np.load("wide_calibration_data.npz")
M = calib["camera_matrix"]
D = calib["dist_coeffs"]

cap = cv2.VideoCapture(camera_id)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

# HSV blue limits
blue_lower = np.array([80, 110, 60])

blue_upper = np.array([150, 255, 255])

# HSV red limits (two ranges)
red_lower1 = np.array([0, 220, 110])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 220, 110])
red_upper2 = np.array([180, 255, 255])

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
        # Blue mask
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

        # Red mask (two ranges)
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        red_opened  = cv2.morphologyEx(red_mask,  cv2.MORPH_OPEN, kernel)
        blue_opened = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)

        blue_closed = cv2.morphologyEx(blue_opened, cv2.MORPH_CLOSE, kernel)
        red_closed  = cv2.morphologyEx(red_opened,  cv2.MORPH_CLOSE, kernel)


        # Blue: left -> right (ignore already-right blocks)
        process_blocks(
            blue_closed,
            y_condition=lambda cy: cy > 0
        )

        # Red: right -> left (ignore already-left blocks)
        process_blocks(
            red_closed,
            y_condition=lambda cy: cy < 0
        )

    if last_point is not None:
        cx, cy = last_point
        cv2.circle(im, (cx, cy), 5, (255, 0, 0), -1)

        robo_point = H2 @ np.array([cx, cy, 1.0])
        robo_point /= robo_point[2]

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

