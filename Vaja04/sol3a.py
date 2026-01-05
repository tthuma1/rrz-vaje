import cv2
import matplotlib.pyplot as plt
import numpy as np
import workspace_utils
import matplotlib.patheffects as pe

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

    # Draw point on original image
    ax.plot(u, v, 'yo', markersize=5)

    # Label robot coordinates
    ax.text(
        u + 10, v - 10,
        f"{Xr:.2f}, {Yr:.2f}",
        color='yellow',
        fontsize=10,
        path_effects=[
            pe.Stroke(linewidth=1, foreground='black'),
            pe.Normal()
        ]
    )

plt.show()
