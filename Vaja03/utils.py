import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_coordinate_system(ax, M, scale=0.2):

	x_1 = np.array([M[0, 3], M[0, 3] + M[0, 0]*scale])
	y_1 = np.array([M[1, 3], M[1, 3] + M[1, 0]*scale])
	z_1 = np.array([M[2, 3], M[2, 3] + M[2, 0]*scale])

	x_2 = np.array([M[0, 3], M[0, 3] + M[0, 1]*scale])
	y_2 = np.array([M[1, 3], M[1, 3] + M[1, 1]*scale])
	z_2 = np.array([M[2, 3], M[2, 3] + M[2, 1]*scale])

	x_3 = np.array([M[0, 3], M[0, 3] + M[0, 2]*scale])
	y_3 = np.array([M[1, 3], M[1, 3] + M[1, 2]*scale])
	z_3 = np.array([M[2, 3], M[2, 3] + M[2, 2]*scale])

	ax.plot3D(x_1, y_1, z_1, 'red', linewidth=4)
	ax.plot3D(x_2, y_2, z_2, 'green', linewidth=4)
	ax.plot3D(x_3, y_3, z_3, 'blue', linewidth=4)

def dh_transform(a, alpha, d, theta):
	"""
	Standard DH transform matrix A_i from parameters (a, alpha, d, theta).
	"""
	ca, sa = np.cos(alpha), np.sin(alpha)
	ct, st = np.cos(theta), np.sin(theta)

	return np.array([
		[ct, -st * ca,  st * sa, a * ct],
		[st,  ct * ca, -ct * sa, a * st],
		[0.0,    sa,       ca,      d   ],
		[0.0,   0.0,      0.0,     1.0  ]
	])

def fk_dh(q, dh_params):
	"""
	Forward kinematics for a serial robot defined by standard DH parameters
	with both revolute ("R") and prismatic ("P") joints.

	q          : joint variables (array-like, size n)
	dh_params  : list of dicts, one per joint, e.g.
		{
		  "joint_type": "R" or "P",
		  "a": ...,
		  "alpha": ...,
		  # for R:
		  "d": ...,
		  "theta_offset": ...,
		  # for P:
		  "d_offset": ...,
		  "theta_offset": ...
		}
	Returns final end effector pose and list of joint matrices.
	"""
	assert len(q) == len(dh_params)
	T = np.eye(4)

	positions = []

	for i, qi in enumerate(q):
		p = dh_params[i]
		a = p["a"]
		alpha = p["alpha"]
		jt = p["joint_type"].upper()

		if jt == "R":
			# Revolute: theta variable, d constant
			d = p["d"]
			theta = p["theta_offset"] + qi
		elif jt == "P":
			# Prismatic: d variable, theta constant
			d = p["d_offset"] + qi
			theta = p["theta_offset"]
		else:
			raise ValueError(f"Unknown joint_type {jt}, expected 'R' or 'P'")

		A = dh_transform(a, alpha, d, theta)
		T = T @ A
		positions.append(T)

	return T, positions

def end_effector_pos(q, dh_params):
	"""
	Extracts the end-effector position (x, y, z) from FK.
	Also returns the joint matrices from fk_dh
	"""
	T, positions = fk_dh(q, dh_params)
	return T[0:3, 3], positions

def generate_figure_8(N=100):
	# https://en.wikipedia.org/wiki/Lemniscate_of_Bernoulli

	N = 100
	x = np.linspace(-1, 1, N)

	c = 0.5

	fn = lambda t, a: ((a*np.sin(t)/(1+(np.cos(t)**2))), a*((np.sin(t)*np.cos(t))/(1+np.cos(t)**2)))
	t = np.linspace(0,2*np.pi, N)

	points = []

	for c in np.linspace(0, 1, 10):
		a = c*np.sqrt(2)
		res = np.array(fn(t, a)).T

	z = np.zeros((res.shape[0]))

	return np.vstack((z, res[:,0], res[:,1]))

def generate_square(width = 0.1, N=100, target_position=np.array([0.0, 0.0, 0.0])):

	ls = np.linspace(0, width, N//4)

	points = []

	p1 = target_position
	p2 = target_position+[0,0,width]
	p3 = target_position+[0,width,width]
	p4 = target_position+[0,width,0]

	for x in np.linspace(0, width, N):
		points.append(p1+[0,0,x])
	for x in np.linspace(0, width, N):
		points.append(p2+[0,x,0])
	for x in np.linspace(0, width, N):
		points.append(p3+[0,0,-x])
	for x in np.linspace(0, width, N):
		points.append(p4+[0,-x,0])

	return np.array(points)

def generate_rectangle(width = 0.1, height = 0.05, N=100, target_position=np.array([0.2, 0.0, 0.05])):
	points = []

	p1 = target_position
	p2 = target_position + [0, width, 0]
	p3 = target_position + [height, width, 0]
	p4 = target_position + [height, 0, 0]

	for x in np.linspace(0, width, N):
		points.append(p1 + [0, x, 0])
	for x in np.linspace(0, height, N):
		points.append(p2 + [x, 0, 0])
	for x in np.linspace(0, width, N):
		points.append(p3 + [0, -x, 0])
	for x in np.linspace(0, height, N):
		points.append(p4 + [-x, 0, 0])

	return np.array(points)

def generate_circle(radius=0.05, N=100, target_position=np.array([0.25, 0.0, 0.05])):
	points = []

	target_position = np.asarray(target_position, dtype=float)

	angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

	for theta in angles:
		points.append(
			target_position + np.array([
				radius * np.cos(theta),
				radius * np.sin(theta),
				0.0
			])
		)

	return np.array(points)

def ccd_step(q, target, dh_params):
	n = len(q)

	for j in range(n):
		# Compute transform up to joint j
		T = fk_dh(q, dh_params)[j]

		# T = np.eye(4)
		# for i in range(j):
		# 	p = dh_params[i]
		# 	theta = q[i] if p["joint_type"] == "r" else p.get("theta_offset", 0.0)
		# 	d     = q[i] if p["joint_type"] == "p" else p.get("d", 0.0)
		# 	T = T @ dh_transform(p["a"], p["alpha"], d, theta)

		joint_pos = T[:3, 3]
		z_axis = T[:3, 2]

		ee_pos = fk_dh(q, dh_params)

		if dh_params[j]["joint_type"] == "r":
			v1 = ee_pos - joint_pos
			v2 = target - joint_pos

			v1 /= np.linalg.norm(v1)
			v2 /= np.linalg.norm(v2)

			cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
			angle = np.arccos(cos_angle)

			if np.dot(np.cross(v1, v2), z_axis) < 0:
				angle = -angle

			q[j] += angle

		else:  # prismatic
			direction = z_axis
			delta = np.dot(target - ee_pos, direction)
			q[j] += delta

	return q

def ik_ccd(
	target,
	q0,
	dh_params,
	max_iters=50,
	tol=1e-4
):
	q = np.array(q0, dtype=float)

	for i in range(max_iters):
		ee = end_effector_pos(q, dh_params)
		err_norm = np.linalg.norm(target - ee)
		if np.linalg.norm(target - ee) < tol:
			return q, True, i, err_norm
			
		q = ccd_step(q, target, dh_params)

	return q, False, max_iters, err_norm

def ccd_step_jacobian(q, target, dh_params, step_scale):
	n = len(q)
	# CCD loop
	for j in range(n):
		# Current EE position and error
		p, _ = end_effector_pos(q, dh_params)
		e = target - p

		# Jacobian column for joint j
		J = numeric_jacobian(q, dh_params)
		Jj = J[:, j]

		denom = np.dot(Jj, Jj)
		if denom < 1e-10:
			continue

		dqj = step_scale * np.dot(Jj, e) / denom
		q[j] += dqj
	
	return q

def ik_ccd_jacobian(
	pts,
	q0,
	dh_params,
	tol=1e-4,
	max_outer_iters=50,
	step_scale=1.0
):
	"""
	Cyclic Coordinate Descent IK using numeric Jacobian columns.

	pts               : list/array of target positions (Nx3)
	q0                : initial joint configuration
	dh_params         : DH parameter list
	tol               : position error tolerance
	max_outer_iters   : CCD iterations per target
	step_scale        : damping on joint updates

	returns:
		qs : list of joint configurations (one per target)
	"""

	q = np.array(q0, dtype=float)
	qs = []

	n = len(q)

	for target in pts:
		for _ in range(max_outer_iters):
			p, _ = end_effector_pos(q, dh_params)
			e = target - p

			if np.linalg.norm(e) < tol:
				break

			q = ccd_step_jacobian(q, dh_params, step_scale)

		qs.append(q.copy())

	return qs

def numeric_jacobian(q, dh_params, h=1e-6):
	"""
	Numerically estimate the Jacobian J = d p / d q for position only.
	q        : current joint configuration (n,)
	returns  : (3 x n) Jacobian
	"""
	q = np.array(q, dtype=float)
	n = len(q)
	p0, _ = end_effector_pos(q, dh_params)
	J = np.zeros((3, n))

	for j in range(n):
		dq = q.copy()
		dq[j] += h
		p_plus, _ = end_effector_pos(dq, dh_params)
		J[:, j] = (p_plus - p0) / h

	return J

def ik_newton_dh(target_pos, q0, dh_params, tol=1e-6, max_iter=100, lr=0.5):
	"""
	Newton–Raphson IK for position in 3D using DH-based FK and numeric Jacobian.

	target_pos : desired end-effector position [x_d, y_d, z_d]
	q0         : initial guess for joint angles (array-like)
	dh_params  : DH parameter list (see fk_dh)
	tol        : convergence tolerance on position error norm
	max_iter   : maximum iterations
	"""
	q = np.array(q0, dtype=float)

	for k in range(max_iter):
		# Forward kinematics and error
		p, positions = end_effector_pos(q, dh_params)

		e = target_pos - p
		err_norm = np.linalg.norm(e)

		if err_norm < tol:
			return q, True, k, err_norm

		# Jacobian and Newton step
		J = numeric_jacobian(q, dh_params)

		# Solve J dq = e  (use pinv for robustness)
		try:
			dq = np.linalg.solve(J, e)
		except np.linalg.LinAlgError:
			dq = np.linalg.pinv(J) @ e

		q = q + lr*dq

	# If we get here, no convergence
	return q, False, max_iter, err_norm

def Netwon_Raphson_demo():
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
			q_sol, success, iters, error = ik_newton_dh(target, q0, dh_params, max_iter=max_iter)
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

if __name__=='__main__':
	Netwon_Raphson_demo()
