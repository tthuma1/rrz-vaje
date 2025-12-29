import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import warnings
warnings.filterwarnings("ignore")

import ikpy.chain
import ikpy.utils.plot as plot_utils
from ikpy.utils import geometry

def sliders():

	radius = 0.5

	URDF_PATH = "so101_new_calib.urdf"

	my_chain = ikpy.chain.Chain.from_urdf_file(URDF_PATH)

	# Initial parameter values
	initial_coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
	num_sliders = len(initial_coeffs)

	link_names = my_chain.links[1:-1]
	link_names = [x.name for x in link_names]

	initial_coeffs = np.hstack((0.0, initial_coeffs, 0.0))

	p = np.array([0.1, -0.2, 0.05])

	y = my_chain.forward_kinematics(initial_coeffs)

	fig, ax = plot_utils.init_3d_figure()

	ax.cla()
	ax.set_xlim(-radius, radius)
	ax.set_ylim(-radius, radius)
	ax.set_zlim(-radius, radius)

	my_chain.plot(initial_coeffs, ax)

	# Plot point p and vector from end effector to p
	ee_pos = y[:3, 3]
	vector = p - ee_pos
	norm = np.linalg.norm(vector)
	print(f"Oddaljenost od cilja: {norm}")
	ax.scatter(p[0], p[1], p[2], color='red', s=50)
	ax.plot([ee_pos[0], p[0]], [ee_pos[1], p[1]], [ee_pos[2], p[2]])

	# Create main figure and axes
	plt.subplots_adjust(bottom=0.3, top=0.93) # Make space at the bottom for sliders

	# Plot the initial line
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_title("S0-101")

	# Slider layout parameters
	slider_height = 0.03
	slider_spacing = 0.005
	slider_left = 0.15
	slider_width = 0.75
	slider_bottom_start = 0.05 # bottom position of the lowest slider

	slider_axes = []
	sliders = []

	# Create 8 sliders stacked vertically
	# num_sliders = 5
	for i in range(num_sliders):
		# Position from bottom up
		bottom = slider_bottom_start + (num_sliders-i) * (slider_height + slider_spacing)
		ax_slider = plt.axes([slider_left, bottom, slider_width, slider_height])
		slider_axes.append(ax_slider)

		# slider names
		label = f"{link_names[i]}"

		slider = Slider(
			ax=ax_slider,
			label=label,
			valmin=-90.0,
			valmax=90.0,
			valinit=initial_coeffs[i],
		)
		sliders.append(slider)

	def update(val):
		"""
		Update function for ALL sliders.
		This is called whenever any slider's value changes.
		"""
		coeffs = [np.deg2rad(s.val) for s in sliders]
		coeffs = np.hstack((0.0, coeffs, 0.0))
		ax.clear()

		# Update the plot
		my_chain.plot(coeffs, ax)
		ax.set_xlim(-radius, radius)
		ax.set_ylim(-radius, radius)
		ax.set_zlim(-radius, radius)

		# Plot point p and vector from end effector to p
		ee_pos = my_chain.forward_kinematics(coeffs)[:3, 3]
		vector = p - ee_pos
		norm = np.linalg.norm(vector)
		print(f"Oddaljenost od cilja: {norm}")
		ax.scatter(p[0], p[1], p[2], color='red', s=50)
		ax.plot([ee_pos[0], p[0]], [ee_pos[1], p[1]], [ee_pos[2], p[2]])

		# Redraw the figure canvas
		fig.canvas.draw_idle()

	# Connect all sliders to the same update function
	for s in sliders:
		s.on_changed(update)

	plt.show()

if __name__ == "__main__":
	sliders()

# Poskusite definirati algoritem, ki opisuje vaš pristop k optimizaciji pozicije robotske roke (v psevdokodi).
#    Najprej premakneš shoulder_pan, da robot gleda v pravo smer. Potem pokrčiš shouolder_lift in wrist_flex tako,
#    da zadaneš ciljno točko. 