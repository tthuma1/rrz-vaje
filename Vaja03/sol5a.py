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

def IK_demo():
    global stopped

    stopped = False

    # stop with 'q' button
    def on_press(event):
        global stopped
        if event.key == "q":
            stopped = True

    URDF_PATH = "so101_new_calib.urdf"

    my_chain = ikpy.chain.Chain.from_urdf_file(URDF_PATH)

    target_orientation = geometry.rpy_matrix(0, np.deg2rad(180), 0)  # point down
    # target_orientation = geometry.rpy_matrix(0, np.deg2rad(90),0) # point forward

    N_pts = 30
    radius = 0.5

    # N = 10
    fig, ax = plot_utils.init_3d_figure()
    ax = fig.add_subplot(111, projection="3d")
    fig.canvas.mpl_connect("key_press_event", on_press)

    #offset = np.array([0.1, 0, 0.5])
    #offset = np.array([0, 0, -0.5])
    offset = np.array([0, 0, 0])

    #pts = generate_figure_8(N=N_pts).T
    #pts = generate_square(0.3, 20, np.array([0.3, 0.1, 0.6]))
    #pts = generate_rectangle(N=20)
    pts = generate_circle(N=100, radius = 0.1)

    #pts *= 0.3
    #pts *= 0.8

    pts += offset

    while not stopped:

        for target_position in pts:
            if stopped:
                break
            ax.cla()

            target_orientation = np.eye(3)
            target_orientation[2,2]=-1
            ik = my_chain.inverse_kinematics(target_position, target_orientation, 'all', optimizer='scalar')

            #ik = my_chain.inverse_kinematics(target_position, optimizer="scalar")  # ignores orientation
            # ik = my_chain.inverse_kinematics(target_position, target_orientation, "all", optimizer='scalar') # includes orientation

            ax.set_xlim(-radius, radius)
            ax.set_ylim(-radius, radius)
            ax.set_zlim(-radius, radius)

            my_chain.plot(ik, ax, target=target_position)

            plt.plot(pts[:, 0], pts[:, 1], pts[:, 2], marker=".", alpha=0.5)

            plt.draw()
            plt.pause(0.01)

if __name__ == "__main__":
    IK_demo()
