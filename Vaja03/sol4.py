import math
import numpy as np
np.set_printoptions(precision=2)

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import warnings
warnings.filterwarnings("ignore")

import ikpy.chain
import ikpy.utils.plot as plot_utils
from ikpy.utils import geometry
from utils import *

def sol4a():
    Netwon_Raphson_demo()

def sol4d():
	global stopped

	# Link lengths
	L1 = 0.5  # base height
	L2 = 0.4
	L3 = 0.3

	# DH parameters for the 3-DOF non-planar arm (all revolute)
	# Table:
	# i | ai     | alpha_i | di    | theta_i
	# 1 | 0      | +pi/2   | L1    | q1
	# 2 | L2     | 0       | 0     | q2
	# 3 | L3     | 0       | 0     | q3
	dh_params = [
		{"a": 0.0, "alpha": np.pi/2, "d": L1, "theta_offset": 0.0, "joint_type": "r"},
		{"a": L2, "alpha": 0.0,      "d": 0.0, "theta_offset": 0.0, "joint_type": "r"},
		{"a": L3, "alpha": 0.0,      "d": 0.0, "theta_offset": 0.0, "joint_type": "r"},
	]

	stopped = False

	# stop with 'q' button
	def on_press(event):
		global stopped
		if event.key == 'q':
			stopped = True

	# Choose a target position inside the reachable workspace
	target = np.array([0.3, 0.1, 0.6])

	# Initial guess for joint angles (radians)
	q0 = np.random.random(3)

	# set up figure
	fig = plt.figure(figsize=(8,6))
	ax = fig.add_subplot(111, projection='3d')
	fig.canvas.mpl_connect('key_press_event', on_press)

	# generate points
	n_pts = 20
	width = 0.3
	pts = generate_square(width=width, N=n_pts, target_position=target)
	max_iter = 20

	# infinite loop
	while not stopped:

		# iterate through points
		for target in pts:
			if stopped:
				break

			# solve for position
			q_sol, success, iters, error = ik_ccd(target, q0, dh_params, max_iter=max_iter)
			# get joint positions for the solution
			_, positions = end_effector_pos(q_sol, dh_params)

			# plot robot positions
			ax.clear()
			ax.set_xlim([-1, 1])
			ax.set_ylim([-1, 1])
			ax.set_zlim([0, 2])
			plt.plot(0,0, 0, 'k*')
			plt.plot(target[0],target[1], target[2], 'b*')
			plt.plot(pts[:,0],pts[:,1], pts[:,2], 'm.')
			plt.title(f'error: {error:.3f} in {iters} iterations')

			for i, T in enumerate(positions):
				x,y,z = T[:3, -1]

				if i>0:
					T_ = positions[i-1]
					x_,y_,z_ = T_[:3, -1]
					plt.plot([x,x_],[y,y_],[z,z_], 'r-')
				else:
					plt.plot([0,x],[0,y],[0,z], 'r-')

				plt.plot(x,y, z, 'r.')
			plt.draw(); plt.pause(0.01)

def sol4d_plot():
	# Link lengths
	L1 = 1  # base height
	L2 = 1
	L3 = 1
	L4 = 1

	dh_params = [
		{"a": L1, "alpha": 0.0, "d": 0.0, "theta_offset": 0.0, "joint_type": "r"},
		{"a": L2, "alpha": 0.0, "d": 0.0, "theta_offset": 0.0, "joint_type": "r"},
		# {"a": L3, "alpha": 0.0, "d": 0.0, "theta_offset": 0.0, "joint_type": "r"},
		# {"a": L4, "alpha": 0.0, "d": 0.0, "theta_offset": 0.0, "joint_type": "r"},
	]

	# q0 = [np.pi/2, np.pi/2, np.pi/2, np.pi/2]
	q0 = np.array([0,0])

	target = np.array([1, 0.5, 0])
	max_iter = 20

	q_sol, success, iters, error = ik_ccd(target, q0, dh_params, tol=1e-5, max_iter=max_iter, do_plot=True)

if __name__ == "__main__":
	# sol4a()
	# sol4d()
	sol4d_plot()

