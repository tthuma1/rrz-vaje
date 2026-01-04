import cv2
import matplotlib.pyplot as plt
import numpy as np
import workspace_utils
import workspace_utils

ime_slike = 'capture_f1.jpg'
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
lower = np.array([90, 110, 30])
upper = np.array([140, 255, 255])

objects_mask = cv2.inRange(hsv_grid, lower, upper)

kernel = np.ones((11,11), np.uint8)
closed = cv2.morphologyEx(objects_mask, cv2.MORPH_CLOSE, kernel)

plt.subplot(2, 3, 5)
plt.imshow(closed, cmap='gray')
plt.title("Maska objektov")

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=4)

plt.subplot(2, 3, 6)
plt.imshow(labels)
plt.title("Regije na sliki")

plt.subplot(2, 3, 2)
img = delovna_povrsina.copy()

for i, (cx, cy) in enumerate(centroids):
    cx, cy = int(cx), int(cy)

    # Skip label 0 (background)
    if i == 0:
        continue

    # Draw small red dot
    cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)

    # Draw text label
    cv2.putText(
        img,
        f"{i}: ({cx}, {cy})",
        (cx + 5, cy - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.tight_layout()
plt.show()