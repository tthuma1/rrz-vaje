import cv2
import numpy as np
import matplotlib.pyplot as plt

# določa kateri id oznake pripada kateremu vogalu
corner_mapping = {
    0: 1,
    1: 2,
    2: 3,
    3: 0,
    4: -1
}

# slovar, ki preslika id oznake v njene koordinate
corner_coordinates = {
    0: None,
    1: None,
    2: None,
    3: None,
}

# velikost preslikane delovne površine v pikslih
workspace_width = 1000
workspace_height = 1000

workspace = np.array([
    [0, workspace_height],
    [0,0],
    [workspace_width, 0],
    [workspace_width, workspace_height],
])

# mapiranje iz koordinat markerjev v koordinate robota, začnemo z levim spodnjim markerjem
robot_ws = np.array([
    [0,0.25],
    [0.488, 0.25],
    [0.488, -0.25],
    [0,-0.25],
])

def get_workspace_corners(im):

    # poišče oznake april in vrne njihove zunanje vogale
    # ustrezen vogal vsake oznake se določi preko preslikave v spremenljivki corner_mapping

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, arucoParams)

    (corners, ids, rejected) = detector.detectMarkers(im)

    if corners:
        for i, id in enumerate(ids.tolist()):
            id = id[0]
            for corner_idx, pt in enumerate(corners[i][0]):
                if corner_idx==4: # id 4 je oznaka na robotski roki in se ignorira
                    continue
                if corner_mapping[id]==corner_idx:
                    pt = (int(pt[0]), int(pt[1]))
                    corner_coordinates[id]=pt

    return np.array(list(corner_coordinates.values()))

def calculate_homography_mapping(corners):

    # preslika zaznane vogale oznak april v delovno površino robota (H1)
    # nato preslika delovno površino robota v koordinatni sistem robota (H2)

    H1, _ = cv2.findHomography(corners.astype(np.float32), workspace.astype(np.float32))
    H2, _ = cv2.findHomography(workspace.astype(np.float32), robot_ws.astype(np.float32))

    return H1, H2