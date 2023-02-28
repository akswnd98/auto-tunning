from vpython import *

import numpy as np
from utils import *
from scipy.integrate import *
import matplotlib.pyplot as plt

ETA_SIGNAL_RATIO = 1000.
P_SIGNAL_RATIO = 1000.
phi_ref_signal = 5. * np.pi / 180. * ETA_SIGNAL_RATIO
theta_ref_signal = 5. * np.pi / 180. * ETA_SIGNAL_RATIO
psi_dot_ref_signal = 0.
z_dot_ref_signal = 1 * P_SIGNAL_RATIO
P_phi = 10.
D_phi = 1.
P_theta = 10.
D_theta = 1.
P_psi_dot = 1.
P_z_dot = 1.
m = 0.64225
l = 0.152
r = 0.2
I = np.array([
  [0.00413138072, 0, 0],
  [0, 0.00413138072, 0],
  [0, 0, 0.00822223903]
], dtype=np.float32)

state_diff = get_state_diff(
  ETA_SIGNAL_RATIO,
  P_SIGNAL_RATIO,
  phi_ref_signal,
  theta_ref_signal,
  psi_dot_ref_signal,
  z_dot_ref_signal,
  P_phi,
  D_phi,
  P_theta,
  D_theta,
  P_psi_dot,
  P_z_dot,
  m,
  l,
  r,
  I
)

initial_state = np.concatenate([
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 0, 0]
], dtype=np.float32)

t_eval = np.arange(0, 5, 0.01, dtype=np.float32)

sol = solve_ivp(state_diff, [0, 5], initial_state, t_eval=t_eval)

eta_dot = sol.y[0: 3]
eta = sol.y[3: 6]
p_dot = sol.y[6: 9]
p = sol.y[9: 12]

def make_grid (x_min, x_max, z_min, z_max, num):
  for x in np.linspace(x_min, x_max, num):
    curve(pos=[vec(x, 0, z_min), vec(x, 0, z_max)], color=color.white, radius=0.01)
  for z in np.linspace(z_min, z_max, num):
    curve(pos=[vec(z_min, 0, z), vec(z_max, 0, z)], color=color.white, radius=0.01)

make_grid(-5, 5, -5, 5, 10)

drone = box(
  pos=vec(0, 0, 0),
  size=vec(0.304, 0.01, 0.304),
  up=vec(0, 1, 0),
  color=color.green,
  make_trail=True,
  trail_type='points',
  trail_radius=0.005,
  interval=0.4,
)

t = 0
dt = 0.01

for i in range(t_eval.shape[0]):
  rate(1 / dt)
  drone.pos = vec(p[0][i], p[2][i], p[1][i])
  R = get_R(eta[:, i])
  up = R @ np.array([0, 0, 1], dtype=np.float32)
  drone.up = vec(up[0], up[2], up[1])
