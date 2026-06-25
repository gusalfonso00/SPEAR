"""
SPEAR complementary filter for attitude tracking and gravity removal.

Assumption set - read before using:

  YAW IS NOT OBSERVABLE.
    This is a 6-axis filter (accel + gyro). There is no magnetometer. Gravity
    cannot distinguish rotations around the gravity axis (yaw), so yaw drifts at
    the gyro bias rate. This is a fundamental limitation of 6-DOF IMU attitude
    estimation - not a bug. For a javelin throw (~0.5s of flight), yaw drift is
    small enough to live with. For longer integration it dominates.

  World frame: Z up, X/Y arbitrary.
    X and Y are defined by the IMU's initial orientation at startup. There is no
    concept of "north" without a magnetometer.

  Body frame: as defined by the IMU PCB.
    Matches the X/Y/Z silkscreen on the LSM6DSO32. The chip datasheet defines
    which axis is which.

  Quaternion convention: scalar-first [w, x, y, z].
    q rotates body frame TO world frame.
    Applying:  v_world = R @ v_body,  where R = quat_to_rotation_matrix(q).
    Equivalently (pure quaternion sandwich):
               v_world = q * [0, v_body] * q_conj.

  Complementary filter alpha:
    alpha = 0.98 means each sample contributes 98% gyro integration, 2% accel
    tilt correction. At 100 Hz, over 50 samples (~0.5s) the accel correction has
    meaningful pull. Over 10s it has fully corrected any initial tilt error.
    Yaw is never corrected by accel (gravity has no yaw component).

  Accel gate:
    During a throw, |accel| >> 1g because linear acceleration dominates. Treating
    that as a gravity vector would corrupt the attitude estimate. The gate
    (default 0.7g-1.3g) suppresses the tilt correction during those samples.
    Gyro integrates freely when the gate is open, so attitude drifts faster then.
    That's the unavoidable tradeoff - the only alternative is a full Kalman filter
    with a dynamic model.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Quaternion primitives
# ---------------------------------------------------------------------------

def quat_multiply(q1, q2):
    """Hamilton product of two quaternions [w, x, y, z].

    Not commutative: q1 * q2 != q2 * q1.
    Represents composing rotation q2 first, then q1.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,   # real part
        w1*x2 + x1*w2 + y1*z2 - z1*y2,   # i part
        w1*y2 - x1*z2 + y1*w2 + z1*x2,   # j part
        w1*z2 + x1*y2 - y1*x2 + z1*w2,   # k part
    ])


def quat_normalize(q):
    """Return unit quaternion. Normalizing after every integration step
    prevents numerical drift from accumulating (quaternion leaves the unit sphere)."""
    n = np.linalg.norm(q)
    if n < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])   # fallback to identity
    return q / n


def quat_conjugate(q):
    """Conjugate [w, x, y, z] -> [w, -x, -y, -z].

    For a unit quaternion, the conjugate is the inverse rotation (body<->world swap).
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_rotation_matrix(q):
    """3x3 Direction Cosine Matrix (DCM) from unit quaternion [w, x, y, z].

    Rotates body frame to world frame: v_world = R @ v_body.
    R^T rotates the other way: v_body = R.T @ v_world.
    """
    w, x, y, z = q
    return np.array([
        [1.0 - 2.0*(y*y + z*z),   2.0*(x*y - w*z),         2.0*(x*z + w*y)        ],
        [2.0*(x*y + w*z),         1.0 - 2.0*(x*x + z*z),   2.0*(y*z - w*x)        ],
        [2.0*(x*z - w*y),         2.0*(y*z + w*x),         1.0 - 2.0*(x*x + y*y)  ],
    ])


def quat_from_accel(accel):
    """Initial attitude quaternion from a single gravity measurement.

    Assumes the sensor is stationary. Finds q such that:
        R @ normalize(accel) = [0, 0, 1]
    i.e., the body-to-world quaternion that maps the measured gravity direction
    (in body frame) to world-up.

    Method: half-angle trick. For two unit vectors a and b, the shortest-arc
    quaternion rotating a to b is:
        [1 + dot(a,b), cross(a,b)] (unnormalized)

    Singularity: if accel points along -Z (IMU directly upside down), cross is
    zero and the formula breaks. We pick an arbitrary 180-degree rotation around X.
    """
    a = accel / np.linalg.norm(accel)      # unit measured gravity in body frame
    b = np.array([0.0, 0.0, 1.0])         # world up (where gravity should point)

    dot = float(np.dot(a, b))

    if dot > 0.9999:                       # already aligned with world up
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -0.9999:                      # pointing straight down: 180 deg around X
        return np.array([0.0, 1.0, 0.0, 0.0])

    cross = np.cross(a, b)                 # rotation axis (unnormalized)
    w = 1.0 + dot                          # proportional to 2*cos^2(theta/2)
    q = np.array([w, cross[0], cross[1], cross[2]])
    return q / np.linalg.norm(q)


def quat_to_euler(q):
    """ZYX Euler angles [roll, pitch, yaw] in degrees from unit quaternion [w, x, y, z].

    ZYX (aerospace) convention: the body reaches its orientation by rotating
    first by yaw around Z, then pitch around Y, then roll around X.

    Use for plotting and reporting only. Euler angles have gimbal lock at
    pitch = +/-90 deg and should never be used internally in the filter.
    """
    w, x, y, z = q

    # Roll: rotation around X (arctan2 has full quadrant coverage, no singularity)
    roll  = np.degrees(np.arctan2(2.0*(w*x + y*z), 1.0 - 2.0*(x*x + y*y)))

    # Pitch: rotation around Y (arcsin is singular at +/-90 deg; clamped)
    sin_p = np.clip(2.0*(w*y - z*x), -1.0, 1.0)
    pitch = np.degrees(np.arcsin(sin_p))

    # Yaw: rotation around Z (arctan2, same as roll)
    yaw   = np.degrees(np.arctan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z)))

    return np.array([roll, pitch, yaw])


# ---------------------------------------------------------------------------
# Gravity removal (standalone so any attitude source can drive it)
# ---------------------------------------------------------------------------

def gravity_remove(accel, attitude, local_g):
    """Subtract gravity from accelerometer readings given attitude history.

    Standalone function - can be called with attitude from any source (this
    complementary filter, Madgwick, etc.) as long as the quaternion convention
    matches (scalar-first, body-to-world).

    Parameters
    ----------
    accel    : Nx3 float array, m/s^2, in body frame, bias-corrected
    attitude : Nx4 float array, quaternions [w,x,y,z], body-to-world
    local_g  : float, local gravitational acceleration in m/s^2

    Returns
    -------
    linear_accel_world : Nx3, m/s^2, gravity-removed acceleration in world frame
    linear_accel_body  : Nx3, m/s^2, gravity-removed acceleration in body frame
    """
    N = len(accel)
    linear_accel_body  = np.zeros((N, 3))
    linear_accel_world = np.zeros((N, 3))

    # In world frame, gravity specific force points up: [0, 0, +local_g]
    # (The accelerometer measures specific force = -gravity + linear_accel.
    #  When stationary: reads [0, 0, +local_g] in world frame. Sign is upward.)
    g_world = np.array([0.0, 0.0, local_g])

    for i in range(N):
        R = quat_to_rotation_matrix(attitude[i])
        g_body = R.T @ g_world                     # world gravity rotated into body frame
        linear_accel_body[i]  = accel[i] - g_body  # subtract expected gravity component
        linear_accel_world[i] = R @ linear_accel_body[i]   # rotate residual to world frame

    return linear_accel_world, linear_accel_body


# ---------------------------------------------------------------------------
# Complementary filter
# ---------------------------------------------------------------------------

def complementary_filter(accel, gyro, dt, local_g=9.7966,
                         alpha=0.98, accel_gate=(0.7, 1.3)):
    """Complementary filter: attitude tracking from 6-axis IMU.

    At each step:
      1. Predict: integrate gyro angular velocity to propagate quaternion forward.
      2. Correct (if gated): find the body-frame rotation that aligns the
         predicted gravity direction with the measured accel direction. Apply
         (1-alpha) of that correction via a small-angle quaternion update.
      3. Gate check: if |accel| is outside [gate_lo, gate_hi], the sensor is
         under linear acceleration (not a clean gravity reference). Skip the
         correction step and integrate gyro only.

    Parameters
    ----------
    accel      : Nx3 float array, m/s^2, body frame, bias-corrected
    gyro       : Nx3 float array, dps, body frame, bias-corrected
    dt         : float, sample period in seconds
    local_g    : float, local gravitational acceleration in m/s^2
    alpha      : float, gyro trust weight per sample [0, 1]
    accel_gate : tuple (lo, hi), gate bounds as multiples of local_g

    Returns
    -------
    attitude           : Nx4 array of quaternions [w,x,y,z], body-to-world
    linear_accel_world : Nx3, m/s^2, gravity-removed accel in world frame
    linear_accel_body  : Nx3, m/s^2, gravity-removed accel in body frame
    accel_gate_active  : N bool array, True = accel correction applied this step
    """
    N = len(accel)

    # Convert gyro dps -> rad/s for integration
    gyro_rad = gyro * (np.pi / 180.0)

    attitude          = np.zeros((N, 4))
    accel_gate_active = np.ones(N, dtype=bool)   # True by default; set False when gated

    gate_lo = accel_gate[0] * local_g
    gate_hi = accel_gate[1] * local_g

    # --- Initialize attitude from first 100 samples (assumes stationary) ---
    init_n     = min(100, N)
    accel_init = accel[:init_n].mean(axis=0)    # average accel during startup window
    init_mag   = np.linalg.norm(accel_init)
    if not (gate_lo <= init_mag <= gate_hi):
        print(f"WARNING: Init accel magnitude {init_mag:.3f} m/s2 is outside gate "
              f"[{gate_lo:.2f}, {gate_hi:.2f}]. Was sensor stationary at start?")

    q          = quat_from_accel(accel_init)    # initial attitude from gravity direction
    attitude[0] = q

    # --- Main filter loop ---
    for i in range(1, N):

        # Step 1: Gyro prediction
        # Represent angular velocity as a pure quaternion [0, wx, wy, wz]
        omega      = gyro_rad[i]
        omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])

        # Quaternion kinematic equation: q_dot = 0.5 * q * omega_quat
        # Integrating gives the new attitude after rotating by omega*dt
        q_dot  = 0.5 * quat_multiply(q, omega_quat)
        q_pred = quat_normalize(q + q_dot * dt)   # Euler step + renormalize

        # Step 2: Accel correction (only when |accel| is plausibly just gravity)
        accel_mag = np.linalg.norm(accel[i])

        if gate_lo <= accel_mag <= gate_hi:
            # Predicted gravity direction in body frame:
            # R^T rotates world-up [0,0,1] into body frame using the predicted attitude
            R           = quat_to_rotation_matrix(q_pred)
            g_body_pred = R.T @ np.array([0.0, 0.0, 1.0])   # unit vector

            # Measured gravity direction in body frame (normalized accel)
            g_body_meas = accel[i] / accel_mag               # unit vector

            # ENU world frame, Z up. At rest, specific force in body frame =
            # R_body_from_world * [0,0,+1] = g_body_pred.
            # cross(measured_up_body, predicted_up_body) yields the rotation axis
            # that nudges predicted toward measured when applied as a body-frame rate.
            # Sign matters: cross(predicted, measured) pushes attitude to the antipode
            # (180 deg flipped) - a stable but wrong equilibrium where the correction
            # term goes to zero and the gyro integrates uncorrected from there.
            # For small angles: |cross(a, b)| ~ sin(theta) ~ theta (rotation magnitude).
            e = np.cross(g_body_meas, g_body_pred)

            # Scale by (1-alpha): apply only a small fraction of the correction.
            # Small-angle correction quaternion: [1, e/2 * (1-alpha)] normalized.
            # This is equivalent to rotating the attitude by (1-alpha)*theta
            # around the error axis.
            corr = 0.5 * (1.0 - alpha) * e
            dq   = quat_normalize(np.array([1.0, corr[0], corr[1], corr[2]]))

            # Apply the body-frame correction to the predicted attitude
            q = quat_normalize(quat_multiply(q_pred, dq))

        else:
            # Outside gate: linear acceleration is dominating, don't trust accel
            q = q_pred
            accel_gate_active[i] = False

        attitude[i] = q

    # --- Gravity removal using final attitude history ---
    linear_accel_world, linear_accel_body = gravity_remove(accel, attitude, local_g)

    return attitude, linear_accel_world, linear_accel_body, accel_gate_active
