import numpy as np

g = 9.80665

# 미분 방정식

def get_state_diff (
    eta_signal_ratio,
    p_signal_ratio,
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
  ):
  T = np.array([
    [-l / np.sqrt(2), -l / np.sqrt(2), l / np.sqrt(2), l / np.sqrt(2)],
    [-l / np.sqrt(2), l / np.sqrt(2), l / np.sqrt(2), -l / np.sqrt(2)],
    [r, -r, r, -r],
    [1 / m, 1 / m, 1 / m, 1 / m]
  ], dtype=np.float32)
  T_inv = np.transpose(T)
  I_inv = np.linalg.inv(I)
  def state_diff (t, y):
    eta_dot = y[0: 3]
    eta = y[3: 6]
    p_dot = y[6: 9]
    p = y[9: 12]
    R = get_R(eta)
    R_inv = np.transpose(R)
    C = get_C(eta)
    C_inv = get_C_inv(eta)
    C_dot = get_C_dot(eta, eta_dot)
    eta_dot_dot_ref = np.array([
      get_phi_dot_dot_ref(
        get_phi_dot_signal_error(
          get_phi_dot_signal_ref(
            get_phi_signal_error(
              phi_ref_signal,
              eta[0] * eta_signal_ratio
            ),
            P_phi
          ),
          eta_dot[0] * eta_signal_ratio
        ),
        D_phi
      ),  
      get_theta_dot_dot_ref(
        get_theta_dot_signal_error(
          get_theta_dot_signal_ref(
            get_theta_signal_error(
              theta_ref_signal,
              eta[1] * eta_signal_ratio
            ),
            P_theta
          ),
          eta_dot[1] * eta_signal_ratio
        ),
        D_theta
      ),
      get_psi_dot_dot_ref(
        get_psi_dot_signal_error(
          psi_dot_ref_signal,
          eta_dot[2] * eta_signal_ratio
        ),
        P_psi_dot
      )
    ])
    z_dot_dot_ref = get_z_dot_dot_ref(
      get_z_dot_signal_error(
        z_dot_ref_signal,
        p_dot[2] * p_signal_ratio
      ),
      P_z_dot
    )
    f = get_force_vector(
      eta_dot_dot_ref,
      z_dot_dot_ref,
      T_inv,
      I,
      C,
      R_inv
    )
    f = np.clip(f, 0, 0.4 * g)
    p_dot_dot = get_p_dot_dot(R, m, f)
    eta_dot_dot = get_eta_dot_dot(I, I_inv, C, C_inv, C_dot, T[0: 3], eta_dot, f)
    return np.concatenate([
      eta_dot_dot,
      eta_dot,
      p_dot_dot,
      p_dot
    ], dtype=np.float32)
  return state_diff

def get_R (eta):
  phi, theta, psi = eta
  return np.array([
    [
      np.cos(psi) * np.cos(theta),
      np.sin(phi) * np.sin(theta) * np.cos(psi) - np.sin(psi) * np.cos(phi),
      np.sin(phi) * np.sin(psi) + np.sin(theta) * np.cos(phi) * np.cos(psi)
    ], [
      np.sin(psi) * np.cos(theta),
      np.sin(phi) * np.sin(psi) * np.sin(theta) + np.cos(phi) * np.cos(psi),
      -np.sin(phi) * np.cos(psi) + np.sin(psi) * np.sin(theta) * np.cos(phi)
    ], [
      -np.sin(theta),
      np.sin(phi) * np.cos(theta),
      np.cos(phi) * np.cos(theta)
    ]
  ], dtype=np.float32)

def get_C (eta):
  phi, theta, psi = eta
  return np.array([
    [1, 0, -np.sin(theta)],
    [0, np.cos(phi), np.sin(phi) * np.cos(theta)],
    [0, -np.sin(phi), np.cos(phi) * np.cos(theta)]
  ], dtype=np.float32)

def get_C_inv (eta):
  phi, theta, psi = eta
  return np.array([
    [1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
    [0, np.cos(phi), -np.sin(phi)],
    [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]
  ], dtype=np.float32)

def get_C_dot (eta, eta_dot):
  phi, theta, psi = eta
  phi_dot, theta_dot, psi_dot = eta_dot
  return np.array([
    [0, 0, -theta_dot * np.cos(theta)],
    [
      0,
      -phi_dot * np.sin(phi),
      phi_dot * np.cos(phi) * np.cos(theta) - theta_dot * np.sin(phi) * np.sin(theta)
    ],
    [
      0,
      -phi_dot * np.cos(phi),
      -phi_dot * np.sin(phi) * np.cos(theta) - theta_dot * np.sin(theta) * np.cos(phi)
    ]
  ], dtype=np.float32)

# 역 동역학

def get_force_vector (eta_dot_dot_ref, z_dot_dot_ref, T_inv, I, C, R_inv):
  p_dot_dot_ref = np.array([0, 0, z_dot_dot_ref], dtype=np.float32)

  return T_inv @ np.concatenate([I @ C @ eta_dot_dot_ref, (R_inv @ (p_dot_dot_ref + np.array([0, 0, g])))[2: ]], dtype=np.float32)

# 정 동역학

def get_eta_dot_dot (I, I_inv, C, C_inv, C_dot, T_eta, eta_dot, f):
  return C_inv @ I_inv @ (
    -I @ C_dot @ eta_dot - np.cross(C @ eta_dot, I @ C @ eta_dot)
    + T_eta @ f
  )

def get_p_dot_dot (R, m, f):
  return -np.array([0, 0, g]) + R / m @ np.array([0, 0, np.sum(f)])

# phi control

def get_phi_signal_error (phi_ref_signal, phi_signal):
  return phi_ref_signal - phi_signal

def get_phi_dot_signal_ref (phi_signal_error, P):
  return phi_signal_error * P

def get_phi_dot_signal_error (phi_dot_signal_ref, phi_dot_signal):
  return phi_dot_signal_ref - phi_dot_signal

def get_phi_dot_dot_ref (phi_dot_signal_error, D):
  return phi_dot_signal_error * D

# theta control

def get_theta_signal_error (theta_ref_signal, theta_signal):
  return theta_ref_signal - theta_signal

def get_theta_dot_signal_ref (theta_signal_error, P):
  return theta_signal_error * P

def get_theta_dot_signal_error (theta_dot_signal_ref, theta_dot_signal):
  return theta_dot_signal_ref - theta_dot_signal

def get_theta_dot_dot_ref (theta_dot_signal_error, D):
  return theta_dot_signal_error * D

# psi dot control

def get_psi_dot_signal_error (psi_dot_ref_signal, psi_dot_signal):
  return psi_dot_ref_signal - psi_dot_signal

def get_psi_dot_dot_ref (psi_dot_signal_error, P):
  return psi_dot_signal_error * P

# z dot control

def get_z_dot_signal_error (z_dot_ref_signal, z_dot_signal):
  return z_dot_ref_signal - z_dot_signal

def get_z_dot_dot_ref (z_dot_signal_error, P):
  return z_dot_signal_error * P


