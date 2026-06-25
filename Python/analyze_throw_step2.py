"""
SPEAR analysis - Step 2: attitude tracking and gravity removal.

Runs the complementary filter from spear_filter.py on the same calibrated
data that Step 1 uses, then integrates gravity-removed world-frame acceleration
to velocity. Does NOT replace analyze_throw.py (Step 1) - run both to compare.

First test: run on the stationary log. Expected results:
  - Peak speed near zero (< 0.1 m/s). Residual comes from gravity-removal error
    from attitude estimation error, not from raw accel bias (that's already removed).
  - Accel gate never triggered (|accel| stays near 1g throughout).
  - Roll/pitch stable near the mounting angles.
  - Yaw drifts slowly at the residual gyro bias rate after calibration correction.

If the stationary test passes, the same script on an actual throw log will show
the throw dynamics with gravity removed.
"""

# %%
import os
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

import spear_analysis as sa
import spear_filter as sf

LOCAL_G = 9.7966   # Boulder, CO - must match characterize_bias.py

# --- Load log and apply calibration ---
df = sa.load_log()

cal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imu_calibration.json')
if os.path.exists(cal_path):
    calibration = sa.load_calibration(cal_path)
    print(f"Calibration source: imu_calibration.json  "
          f"(characterized {calibration['characterized_at'][:10]}, "
          f"gravity {calibration['gravity_axis']})")
    df = sa.apply_bias(df, calibration=calibration)
else:
    print("Calibration source: per-log stationary window (no imu_calibration.json found)")
    bias = sa.calibrate_bias(df, t_start=0.0, t_end=3.0)
    df = sa.apply_bias(df, bias)

# Build numpy arrays from bias-corrected columns
t     = df['t_s'].values
accel = df[['ax_corr', 'ay_corr', 'az_corr']].values    # m/s^2, body frame
gyro  = df[['gx_corr', 'gy_corr', 'gz_corr']].values    # rad/s, body frame
dt    = float(np.median(np.diff(t)))                      # nominal sample period (s)

# Filter expects gyro in dps (converts to rad/s internally to match comment docs)
gyro_dps = gyro * (180.0 / np.pi)

# --- Run complementary filter ---
print(f"\nRunning complementary filter  "
      f"(alpha=0.98, gate=0.7-1.3g, local_g={LOCAL_G} m/s^2, dt={dt*1000:.2f} ms)...")

attitude, lin_accel_world, lin_accel_body, gate_active = sf.complementary_filter(
    accel, gyro_dps, dt, local_g=LOCAL_G, alpha=0.98, accel_gate=(0.7, 1.3)
)

# --- Integrate world-frame linear accel -> world-frame velocity ---
vx    = integrate.cumulative_trapezoid(lin_accel_world[:, 0], t, initial=0.0)
vy    = integrate.cumulative_trapezoid(lin_accel_world[:, 1], t, initial=0.0)
vz    = integrate.cumulative_trapezoid(lin_accel_world[:, 2], t, initial=0.0)
speed = np.sqrt(vx**2 + vy**2 + vz**2)

# --- Euler angles (for plotting only - not used in filter) ---
euler_deg = np.array([sf.quat_to_euler(q) for q in attitude])   # Nx3, [roll, pitch, yaw] deg

# --- Derived quantities for plotting ---
accel_mag     = np.linalg.norm(accel, axis=1)
lin_accel_mag = np.linalg.norm(lin_accel_world, axis=1)
gate_lo_abs   = 0.7 * LOCAL_G
gate_hi_abs   = 1.3 * LOCAL_G
gate_suppressed = int(np.sum(~gate_active))

# --- Print summary ---
yaw_range = euler_deg[:, 2].max() - euler_deg[:, 2].min()
duration  = t[-1] - t[0]

print(f"\n{'='*58}")
print(f"Step 2 Summary")
print(f"{'='*58}")
print(f"  Duration:              {duration:.1f} s  ({len(t)} samples, {1/dt:.1f} Hz)")
print(f"  Accel gate suppressed: {gate_suppressed} samples  "
      f"({100.0*gate_suppressed/len(t):.1f}% of record)")
print(f"  Peak speed:            {speed.max():.4f} m/s")
print(f"  Final velocity:        "
      f"vx={vx[-1]:+.4f}  vy={vy[-1]:+.4f}  vz={vz[-1]:+.4f} m/s")
print(f"  Final speed:           {speed[-1]:.4f} m/s")
print(f"  Peak linear accel:     {lin_accel_mag.max():.4f} m/s^2  "
      f"({lin_accel_mag.max()/LOCAL_G*1000:.1f} mg)")
print(f"  Roll  range:           "
      f"{euler_deg[:,0].min():.2f} to {euler_deg[:,0].max():.2f} deg")
print(f"  Pitch range:           "
      f"{euler_deg[:,1].min():.2f} to {euler_deg[:,1].max():.2f} deg")
print(f"  Yaw   drift:           "
      f"{yaw_range:.2f} deg over {duration:.0f} s  "
      f"({yaw_range/duration*60:.2f} deg/min)")
print(f"{'='*58}")

# --- 4-panel figure ---
fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
fig.suptitle('SPEAR Step 2: Complementary Filter + Gravity Removal', fontsize=13)

# Blended transform: data-x, axes-y (0=bottom, 1=top of each panel).
# Used to shade full panel height wherever gate was suppressed.
def shade_gate_suppressed(ax, t, gate_active):
    """Shade background red wherever accel gate was suppressed (gyro-only mode)."""
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.fill_between(t, 0, 1, where=~gate_active,
                    transform=trans, alpha=0.25, color='red',
                    label='gate suppressed (gyro-only)', zorder=0)

# Panel 1: raw accel magnitude with gate boundaries
ax = axes[0]
ax.plot(t, accel_mag, color='steelblue', linewidth=0.8, label='|accel| (m/s^2)')
ax.axhline(LOCAL_G,   color='green',  linewidth=1.0, linestyle='--', label=f'local g ({LOCAL_G})')
ax.axhline(gate_lo_abs, color='orange', linewidth=0.8, linestyle=':', label='gate lo (0.7g)')
ax.axhline(gate_hi_abs, color='orange', linewidth=0.8, linestyle=':', label='gate hi (1.3g)')
ax.fill_between(t, gate_lo_abs, gate_hi_abs, alpha=0.10, color='green')
shade_gate_suppressed(ax, t, gate_active)
ax.set_ylabel('|accel| (m/s^2)')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: Euler angles (roll, pitch, yaw)
ax = axes[1]
ax.plot(t, euler_deg[:, 0], label='roll',  linewidth=0.8)
ax.plot(t, euler_deg[:, 1], label='pitch', linewidth=0.8)
ax.plot(t, euler_deg[:, 2], label='yaw',   linewidth=0.8, linestyle='--')
shade_gate_suppressed(ax, t, gate_active)
ax.set_ylabel('Euler angles (deg)')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)

# Panel 3: linear accel in world frame (gravity removed)
ax = axes[2]
for col, label in zip([0, 1, 2], ['lin_ax (world)', 'lin_ay (world)', 'lin_az (world)']):
    ax.plot(t, lin_accel_world[:, col], label=label, linewidth=0.8)
shade_gate_suppressed(ax, t, gate_active)
ax.set_ylabel('Linear accel (m/s^2)')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)

# Panel 4: world-frame velocity + speed magnitude
ax = axes[3]
ax.plot(t, vx, label='vx', linewidth=0.8)
ax.plot(t, vy, label='vy', linewidth=0.8)
ax.plot(t, vz, label='vz', linewidth=0.8)
ax.plot(t, speed, label='speed |v|', linewidth=1.2, color='black', linestyle='--')
shade_gate_suppressed(ax, t, gate_active)
ax.set_ylabel('Velocity (m/s)')
ax.set_xlabel('Time (s)')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.show()
# %%
