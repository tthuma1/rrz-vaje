import cv2
import matplotlib.pyplot as plt
import numpy as np
import workspace_utils
import workspace_utils
import matplotlib.patheffects as pe

### a), b)

def solve(ime_slike, lowers, uppers):
    im = cv2.imread(ime_slike)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    corners = workspace_utils.get_workspace_corners(im, draw_markers=True)
    H1, H2 = workspace_utils.calculate_homography_mapping(corners)

    plt.subplot(2, 3, 1)
    plt.imshow(im)
    plt.title("Originalna slika")
    plt.scatter(corners[:,0], corners[:,1], c='b', s=10)

    delovna_povrsina = cv2.warpPerspective(im, H1, (1000, 1000))

    plt.subplot(2, 3, 2)
    plt.imshow(delovna_povrsina)
    plt.title("Delovna povrsina")
    # src_pts = np.float32(plt.ginput(4)).reshape(-1,1,2)
    # np.save('tocke2', src_pts)

    points = np.load("tocke2.npy")
    points = points.reshape(4,2)
    x_min = int(np.min(points[:, 0]))
    x_max = int(np.max(points[:, 0]))
    y_min = int(np.min(points[:, 1]))
    y_max = int(np.max(points[:, 1]))

    delovna_povrsina_mask = np.zeros(delovna_povrsina.shape[:2], dtype=np.uint8)
    delovna_povrsina_mask[y_min:y_max, x_min:x_max] = 255

    plt.subplot(2, 3, 3)
    plt.imshow(delovna_povrsina_mask, cmap='gray')
    plt.title("Maska delovne površine")

    masked_povrsina = cv2.bitwise_and(delovna_povrsina, delovna_povrsina, mask=delovna_povrsina_mask)
    plt.subplot(2,3,4)
    plt.imshow(masked_povrsina, cmap="gray")
    plt.title("Delovna površina")

    hsv_grid = cv2.cvtColor(masked_povrsina, cv2.COLOR_RGB2HSV)

    # HSV meje
    objects_mask = np.zeros_like(hsv_grid[..., 0]) # (1000, 1000) namesto (1000, 1000, 3)
    for lower, upper in zip(lowers, uppers):
        objects_mask_tmp = hsv_grid
        objects_mask_tmp = cv2.inRange(objects_mask_tmp, lower, upper)
        objects_mask = cv2.bitwise_or(objects_mask, objects_mask_tmp)


    kernel = np.ones((11,11), np.uint8)
    opened = cv2.morphologyEx(objects_mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    plt.subplot(2, 3, 5)
    plt.imshow(closed, cmap='gray')
    plt.title("Maska objektov")

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=4)

    plt.subplot(2, 3, 6)
    plt.imshow(labels)
    plt.title("Regije na sliki")

    ax = plt.subplot(2, 3, 2)

    for i, (cx, cy) in enumerate(centroids):
        if i == 0:
            continue

        cx, cy = int(cx), int(cy)

        # --- Map centroid to robot coordinates using H2 ---
        pt = np.array([cx, cy, 1.0])
        mapped = H2 @ pt
        mapped /= mapped[2]

        Xr = mapped[0]   # meters
        Yr = mapped[1]   # meters

        # --- Red dot ---
        ax.plot(cx, cy, 'ro', markersize=1)

        # --- Label using plt.text ---
        ax.text(
            cx + 10, cy - 10,
            f"{Xr:.2f}, {Yr:.2f}",
            color='white',
            fontsize=8,
            path_effects=[
                pe.Stroke(linewidth=1, foreground='black'),
                pe.Normal()
            ]
        )


    plt.tight_layout()
    plt.show()

solve('capture_f1.jpg',
    [
        np.array([90, 110, 30])
    ],
    [
        np.array([140, 255, 255])
    ])

solve('moja_slika3.jpg',
    [
        # blue
        np.array([90, 90, 30]),
        # red
        np.array([0, 100, 20]),
        np.array([160, 100, 20]),
        # black
        np.array([0, 0, 1]),
    ],
    [
        # blue
        np.array([140, 255, 255]),
        # red
        np.array([10, 255, 255]),
        np.array([180, 255, 255]),
        # black
        np.array([180, 255, 100]),
    ])


