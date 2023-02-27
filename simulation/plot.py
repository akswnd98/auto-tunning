import numpy as np
from utils import *
from scipy.integrate import *
import matplotlib.pyplot as plt

ETA_SIGNAL_RATIO = 1000.
P_SIGNAL_RATIO = 1000.
phi_ref_signal = 10. * np.pi / 180. * ETA_SIGNAL_RATIO
theta_ref_signal = 20. * np.pi / 180. * ETA_SIGNAL_RATIO
psi_dot_ref_signal = 0.
z_dot_ref_signal = 0.
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

sol = solve_ivp(state_diff, [0, 5], initial_state, t_eval=np.arange(0, 5, 0.01, dtype=np.float32))

eta_dot = sol.y[0: 3]
eta = sol.y[3: 6]
p_dot = sol.y[6: 9]
p = sol.y[9: 12]

plt.plot(sol.t, eta[0], label='phi')
plt.plot(sol.t, eta[1], label='theta')
plt.plot(sol.t, eta_dot[2], label='psi_dot')
plt.plot(sol.t, p_dot[2], label='z_dot')
plt.plot(sol.t, p[0], label='x')
plt.plot(sol.t, p[1], label='y')
plt.plot(sol.t, p[2], label='z')

plt.legend()

plt.show()
