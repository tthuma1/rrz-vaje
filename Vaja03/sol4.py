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

def sol4():
    Netwon_Raphson_demo()

if __name__ == "__main__":
	sol4()


#I have a robot defined with these Denavit–Hartenberg parameters in python:
## Link lengths
	#L1 = 0.5  # base height
	#L2 = 0.4
	#L3 = 0.3

	## DH parameters for the 3-DOF non-planar arm (all revolute)
	## Table:
	## i | ai     | alpha_i | di    | theta_i
	## 1 | 0      | +pi/2   | L1    | q1
	## 2 | L2     | 0       | 0     | q2
	## 3 | L3     | 0       | 0     | q3
	#dh_params = [
		#{"a": 0.0, "alpha": np.pi/2, "d": L1, "theta_offset": 0.0, "joint_type": "r"},
		#{"a": L2, "alpha": 0.0,      "d": 0.0, "theta_offset": 0.0, "joint_type": "r"},
		#{"a": L3, "alpha": 0.0,      "d": 0.0, "theta_offset": 0.0, "joint_type": "r"},
	#]
#I have a list of points generated like this: pts = generate_square(width=width, N=n_pts, target_position=target) I need to write a function for cyclic coordinate descent algorithm that uses these points and DH parameters for inverse kinematics. Follow this pseduo code:
#for i=1,…,k
#do
   #qi←qi−1
   #for all joints j=1,...,n
#do
       #Compute q∗j=argminqj∥f(qi)∥
       #Set qi,j←q∗j
#return qk




#Below is a **clean, working CCD inverse kinematics implementation** tailored to **your exact DH table and point list**.
#It assumes:

#* All joints are **revolute**
#* You already have `generate_square(...)`
#* You want to track each target point sequentially
#* Forward kinematics is computed from your DH parameters

#I’ll keep it modular so you can plug it directly into your project.

#---

### 1. Forward kinematics from DH parameters

#First, a helper to compute a single DH transform and the full chain.

#```python
#import numpy as np

#def dh_transform(a, alpha, d, theta):
    #"""Standard DH homogeneous transform"""
    #ca, sa = np.cos(alpha), np.sin(alpha)
    #ct, st = np.cos(theta), np.sin(theta)

    #return np.array([
        #[ct, -st * ca,  st * sa, a * ct],
        #[st,  ct * ca, -ct * sa, a * st],
        #[0,        sa,       ca,      d],
        #[0,         0,        0,      1]
    #])
#```

#```python
#def forward_kinematics(q, dh_params):
    #"""
    #Returns:
    #- list of joint positions (including base at index 0)
    #- end-effector position
    #"""
    #T = np.eye(4)
    #joint_positions = [T[:3, 3].copy()]

    #for qi, dh in zip(q, dh_params):
        #T = T @ dh_transform(
            #dh["a"],
            #dh["alpha"],
            #dh["d"],
            #qi + dh["theta_offset"]
        #)
        #joint_positions.append(T[:3, 3].copy())

    #return joint_positions, joint_positions[-1]
#```

#---

### 2. Single-joint CCD update (core math)

#This computes the **optimal angle update** for one joint using geometric projection.

#```python
#def ccd_update(q, joint_idx, target, dh_params):
    #joint_positions, ee_pos = forward_kinematics(q, dh_params)

    #p_j = joint_positions[joint_idx]
    #p_e = ee_pos

    #v_e = p_e - p_j
    #v_t = target - p_j

    ## Project onto plane orthogonal to joint axis (Z-axis of joint frame)
    #z_axis = np.array([0, 0, 1])

    #v_e_proj = v_e - np.dot(v_e, z_axis) * z_axis
    #v_t_proj = v_t - np.dot(v_t, z_axis) * z_axis

    #if np.linalg.norm(v_e_proj) < 1e-6 or np.linalg.norm(v_t_proj) < 1e-6:
        #return q[joint_idx]

    #v_e_proj /= np.linalg.norm(v_e_proj)
    #v_t_proj /= np.linalg.norm(v_t_proj)

    #cos_angle = np.clip(np.dot(v_e_proj, v_t_proj), -1.0, 1.0)
    #angle = np.arccos(cos_angle)

    ## Sign using cross product
    #sign = np.sign(np.dot(z_axis, np.cross(v_e_proj, v_t_proj)))
    #return q[joint_idx] + sign * angle
#```

#---

### 3. Full CCD algorithm (matches your pseudocode)

#This directly follows:

#```
#for i=1,…,k
  #qi ← qi−1
  #for j=1,…,n
    #qj* = argmin ||f(q)||
    #qi,j ← qj*
#```

#```python
#def ccd_ik(
    #q_init,
    #target,
    #dh_params,
    #max_iters=100,
    #tol=1e-3
#):
    #q = q_init.copy()
    #n_joints = len(q)

    #for _ in range(max_iters):
        #for j in range(n_joints):
            #q[j] = ccd_update(q, j, target, dh_params)

        #_, ee_pos = forward_kinematics(q, dh_params)
        #if np.linalg.norm(ee_pos - target) < tol:
            #break

    #return q
#```

#---

### 4. Using it with your generated square points

#```python
#q = np.zeros(3)  # initial guess

#for target in pts:
    #q = ccd_ik(q, target, dh_params)
    #print("Target:", target, "Solution:", q)
#```

#---

### 5. Why this works well for your robot

#* Your DH structure is **RRR with vertical offset**
#* CCD handles **non-planar chains naturally**
#* No Jacobian inversion → numerically stable
#* Perfect for **trajectory tracking** (like your square)

#---

### 6. Optional improvements (if you want)

#* Joint limits
#* Damping small updates
#* Multiple CCD sweeps per point
#* Warm-start from previous solution (you already do this)

#If you want, I can:

#* Add **joint limits**
#* Add **visualization**
#* Convert this to **Jacobian IK**
#* Optimize it for real-time execution



