import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

from utils import *

# Plot setup
def plot_robot(ax, positions):
	ax.clear()
	ax.set_xlim([-1, 1])
	ax.set_ylim([-1, 1])
	ax.set_zlim([0, 2])
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlabel("Z")

	# Plot joints
	positions = np.array(positions)
	ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='red', s=50)

	# Plot connections
	for i in range(len(positions) - 1):
		ax.plot(
			[positions[i, 0], positions[i+1, 0]],
			[positions[i, 1], positions[i+1, 1]],
			[positions[i, 2], positions[i+1, 2]],
			'blue'
		)

	ax.plot([0, positions[0,0]], [0, positions[0,1]], [0, positions[0,2]])

def sol3_antro():
	L1 = 0.5  # base height
	L2 = 0.4
	L3 = 0.3

	# anthropomorphic
	dh_params = [
		{"a": 0.0, "alpha": np.pi/2, "d": L1, "theta_offset": 0.0, "joint_type": "r"},
		{"a": L2, "alpha": 0.0,      "d": 0.0, "theta_offset": 0.0, "joint_type": "r"},
		{"a": L3, "alpha": 0.0,      "d": 0.0, "theta_offset": 0.0, "joint_type": "r"},
	]

	q0 = np.zeros(len(dh_params))
	_, positions = end_effector_pos(q0, dh_params)

	# Create a plot
	fig = plt.figure(figsize=(8, 8))
	ax = fig.add_subplot(111, projection='3d')

	positions = [p[:3,-1] for p in positions]

	plot_robot(ax, positions)

	# Adjust the layout to make space for sliders
	plt.subplots_adjust(left=0.2, bottom=0.25, top=0.9)

	# Create sliders for joint values
	slider_axes = []
	sliders = []
	num_joints = len(dh_params)
	for i in range(num_joints):
		# Arrange sliders below the plot with proper spacing
		ax_slider = plt.axes([0.2, 0.05 + (num_joints - 1 - i) * 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
		slider_label = f'Joint {i+1} {"(Linear)" if dh_params[i]["joint_type"]!="r" else "(Rotational)"}'
		# slider_range = (0, 1) if dh_params[i]["joint_type"]!="r" else (-180, 180)
		slider_range = (0, 1) if dh_params[i]["joint_type"]!="r" else (-np.pi, np.pi)
		slider = Slider(ax_slider, slider_label, slider_range[0], slider_range[1], valinit=q0[i])
		sliders.append(slider)

	# Update function for sliders
	def update(val):
		global joint_values
		joint_values = [slider.val for slider in sliders]

		_, positions = end_effector_pos(joint_values, dh_params)
		positions = [p[:3,-1] for p in positions]

		print("Končna pozicija:", positions[-1])

		plot_robot(ax, positions)
		fig.canvas.draw_idle()

	# Attach update function to sliders
	for slider in sliders:
		slider.on_changed(update)

	plt.show()

def sol3_stanford():
	L1 = 0.5  # base height
	L2 = 0.4
	L3 = 0.3

	# stanford RRP robot
	dh_params = [
		{"a": 0.0, "alpha": -np.pi/2, "d": L1, "theta_offset": 0.0, "joint_type": "r"},
		{"a": 0.0, "alpha": np.pi/2,  "d": L2, "theta_offset": 0.0, "joint_type": "r"},
		{"a": 0.0, "alpha": 0.0,      "d_offset": L3, "theta_offset": 0.0, "joint_type": "p"},
	]

	q0 = np.zeros(len(dh_params))
	_, positions = end_effector_pos(q0, dh_params)

	# Create a plot
	fig = plt.figure(figsize=(8, 8))
	ax = fig.add_subplot(111, projection='3d')

	positions = [p[:3,-1] for p in positions]

	plot_robot(ax, positions)

	# Adjust the layout to make space for sliders
	plt.subplots_adjust(left=0.2, bottom=0.25, top=0.9)

	# Create sliders for joint values
	slider_axes = []
	sliders = []
	num_joints = len(dh_params)
	for i in range(num_joints):
		# Arrange sliders below the plot with proper spacing
		ax_slider = plt.axes([0.2, 0.05 + (num_joints - 1 - i) * 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
		slider_label = f'Joint {i+1} {"(Linear)" if dh_params[i]["joint_type"]!="r" else "(Rotational)"}'
		# slider_range = (0, 1) if dh_params[i]["joint_type"]!="r" else (-180, 180)
		slider_range = (0, 1) if dh_params[i]["joint_type"]!="r" else (-np.pi, np.pi)
		slider = Slider(ax_slider, slider_label, slider_range[0], slider_range[1], valinit=q0[i])
		sliders.append(slider)

	# Update function for sliders
	def update(val):
		global joint_values
		joint_values = [slider.val for slider in sliders]

		_, positions = end_effector_pos(joint_values, dh_params)
		positions = [p[:3,-1] for p in positions]

		print("Končna pozicija:", positions[-1])

		plot_robot(ax, positions)
		fig.canvas.draw_idle()

	# Attach update function to sliders
	for slider in sliders:
		slider.on_changed(update)

	plt.show()


if __name__=='__main__':
	sol3_antro()
	sol3_stanford()
