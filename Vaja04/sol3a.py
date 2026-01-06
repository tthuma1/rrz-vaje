import cv2
import matplotlib.pyplot as plt
import numpy as np
import workspace_utils
import matplotlib.patheffects as pe

def save_homography(im):
    corners = workspace_utils.get_workspace_corners(im, draw_markers=True)
    H1, H2 = workspace_utils.calculate_homography_mapping(corners)

    np.savez('homography.npz', H1=H1, H2=H2)

def load_homography():
    data = np.load('homography.npz')
    return data['H1'], data['H2']

im_path = 'capture_f1.jpg'
im = cv2.imread(im_path)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# save_homography(im)
H1, H2 = load_homography()

delovna_povrsina = cv2.warpPerspective(im, H1, (1000, 1000))

st_tock = 4
plt.clf()
plt.imshow(delovna_povrsina)
plt.title(f"Označi {st_tock} točke")
clicked_pts = np.array(plt.ginput(st_tock, timeout=0))
plt.close()

H1_inv = np.linalg.inv(H1)

plt.clf()
plt.imshow(im)
plt.title("Originalna slika, koordinate KS robota")

pts = np.vstack([clicked_pts[:,0], clicked_pts[:,1], np.ones(st_tock)])

# s `H1_inv` gremo iz warped delovne površine => originalno sliko
orig_pts = H1_inv @ pts
orig_pts /= orig_pts[2]

# s `H2` gremo iz warped delovne površine v KS robota
robot_pts = H2 @ pts
robot_pts /= robot_pts[2]

for i in range(orig_pts.shape[1]): # i gre po vsaki točki
    orig_x = orig_pts[0][i]
    orig_y = orig_pts[1][i]

    robot_x = robot_pts[0][i]
    robot_y = robot_pts[1][i]

    # točke rišemo na originalni sliki
    plt.plot(orig_x, orig_y, 'o', markersize=5)

    # izpiši točke v KS robota
    plt.text(
        orig_x + 10, orig_y - 10,
        f"{robot_x:.2f}, {robot_y:.2f}",
        color='white',
        fontsize=10,
        path_effects=[
            pe.Stroke(linewidth=1, foreground='black'),
            pe.Normal()
        ]
    )

plt.show()
