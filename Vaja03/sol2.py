import numpy as np
from utils import *
import matplotlib.pyplot as plt

def dh_transform(a, alpha, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d ],
        [0.0, 0.0, 0.0, 1.0 ]
    ])

def stanford_manipulator(q):
    S1 = dh_transform(0, -np.pi/2, 5, q[0])
    S2 = dh_transform(0, np.pi/2, 5, q[1])
    S3 = dh_transform(0, 0, q[2] + 2, 0)

    S12 = S1 @ S2
    S123 = S12 @ S3

    return S1, S12, S123

def antropomorphic_manipulator(q):
    S1 = dh_transform(0, np.pi/2, 3, q[0])
    S2 = dh_transform(3, 0, 0, q[1])
    S3 = dh_transform(3, 0, 0, q[2])

    S12 = S1 @ S2
    S123 = S12 @ S3

    return S1, S12, S123

q = [0,0,0]
# q = [-np.pi/2,0,-4]

fig = plt.figure()
ax = plt.axes(projection='3d')
S1, S12, S123 = stanford_manipulator(q)
#show_coordinate_system(ax, stanford_manipulator(q2)[2])

show_coordinate_system(ax, np.eye(4))
show_coordinate_system(ax, S1)
show_coordinate_system(ax, S12)
show_coordinate_system(ax, S123)

p0 = [0,0,0,1] # izhodišče robota
p1 = S1 @ p0
p2 = S12 @ p0
p3 = S123 @ p0

# p1 = S1[:, 3]
# p2 = S12[:, 3]
# p3 = S123[:, 3]

print("Lokacija konice stanford:", p3)

xs = [p0[0], p1[0], p2[0], p3[0]]
ys = [p0[1], p1[1], p2[1], p3[1]]
zs = [p0[2], p1[2], p2[2], p3[2]]

ax.plot(xs, ys, zs)
ax.scatter(xs, ys, zs)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_xlim([-1, 7])
ax.set_ylim([-1, 7])
ax.set_zlim([-1, 7])

plt.title(f"Stanford manipulator, {q=}")
plt.show()


### antropomorphic

q = [0,0,0]
# q = [np.pi/2,0,0]

fig = plt.figure()
ax = plt.axes(projection='3d')
S1, S12, S123 = antropomorphic_manipulator(q)
#show_coordinate_system(ax, antropomorphic_manipulator(q2))

show_coordinate_system(ax, np.eye(4))
show_coordinate_system(ax, S1)
show_coordinate_system(ax, S12)
show_coordinate_system(ax, S123)

p0 = [0,0,0,1]
p1 = S1 @ p0
p2 = S12 @ p0
p3 = S123 @ p0

print("Lokacija konice antropomorphic:", p3)

xs = [p0[0], p1[0], p2[0], p3[0]]
ys = [p0[1], p1[1], p2[1], p3[1]]
zs = [p0[2], p1[2], p2[2], p3[2]]

ax.plot(xs, ys, zs)
ax.scatter(xs, ys, zs)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_xlim([-1, 7])
ax.set_ylim([-1, 7])
ax.set_zlim([-1, 7])

plt.title(f"Antropomorfni manipulator, {q=}")
plt.show()


