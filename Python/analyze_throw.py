"""
SPEAR analysis - Step 1: raw integration to velocity.

Loads the most recent IMU log, applies bias correction, integrates
acceleration to velocity, and plots the result.

Calibration source priority:
  1. imu_calibration.json (from characterize_bias.py) - preferred
  2. Per-log stationary window (first 3s) - fallback when no JSON exists

Step 1 deliberately skips orientation tracking and gravity removal -
expect the velocity to drift visibly. That drift motivates Step 2.
"""

# %%
import os
import spear_analysis as sa

# 1. Load most recent log
df = sa.load_log()

# 2. Apply bias - prefer imu_calibration.json if present
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
    print("Bias values:")
    for k, v in bias.items():
        print(f"  {k}: {v:+.4f}")
    df = sa.apply_bias(df, bias)

# 3. Integrate corrected acceleration to velocity
df = sa.integrate_velocity(df)

# 4. Plot
sa.plot_acc_and_vel(df, title="SPEAR Step 1: raw velocity integration")
# %%
