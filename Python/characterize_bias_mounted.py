"""
Mounted (in-situ) IMU bias characterization via sphere fit.

Use this INSTEAD of characterize_bias.py when the sensor is already mounted
on the javelin and cannot be placed flat. No orientation assumption is made:
for any stationary pose, the bias-corrected accel magnitude must equal local
gravity, so with enough different poses the bias is the vector that puts
every pose on a sphere of radius local_g.

HOW TO CAPTURE THE CALIBRATION LOG (one continuous recording):

  1. Start the logger:  python log_imu_udp.py
  2. Prop the javelin still in one orientation. Hands off. Hold ~10 s.
  3. Move it to a NEW orientation, prop it, hands off, ~10 s again.
  4. Repeat for 8-10 orientations total, then stop the logger (Ctrl+C).

  Orientation spread matters more than count. DO NOT only roll the javelin
  about its long axis: that sweeps gravity around a circle in one sensor
  plane and leaves the long-axis bias unobservable (the script warns if it
  detects this). Good pose set:
    - horizontal on the ground, rolled 0 / 90 / 180 / 270 deg
    - tip up, leaning steeply against a wall
    - tip down, leaning steeply
    - two ~45 degree leans with different rolls
  Each pose just needs to be motionless; against a wall, on grass, in a
  shoe, whatever works. The script finds the still periods automatically.

  Calibrate at field temperature if possible (bias drifts with temp).

WHAT IT WRITES:

  imu_calibration.json in the same format characterize_bias.py produces,
  so analyze_throw.py / analyze_throw_step2.py / analyze_field_throw.py
  pick it up with no changes. The previous file's values are printed next
  to the new ones before overwriting.

Usage:
    python characterize_bias_mounted.py [logfile.csv]
        [--local-g M_S2] [--gyro-tol DPS] [--accel-std-tol M_S2]
        [--min-pose-dur S] [--output FILE]
"""

import argparse
import glob
import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import spear_analysis as sa

G = 9.80665              # standard gravity: unit conversion (mg) only
LOCAL_GRAVITY_MPS2 = 9.7966   # Boulder CO default, override with --local-g


def to_mg(mps2):
    return mps2 * 1000.0 / G


def find_still_poses(df, gyro_tol_dps, accel_std_tol, min_dur_s):
    """Find stationary pose windows in RAW data.

    This is a chicken-and-egg problem: the field pipeline's quiet detector
    checks | |accel| - g |, but that test needs the bias we are trying to
    find (a 0.5 m/s^2 bias shifts |accel| enough to fail it). So stillness
    is detected WITHOUT any magnitude assumption:
      - every gyro axis below gyro_tol_dps (raw gyro bias is well under
        1 dps, so a raw-value threshold is safe), AND
      - rolling std of |accel| below accel_std_tol (stability, not level).

    Returns a list of (i_start, i_end) inclusive index pairs.
    """
    t = df['t_s'].values
    gyro_dps = np.degrees(df[['gx', 'gy', 'gz']].values)
    amag = np.linalg.norm(df[['ax', 'ay', 'az']].values, axis=1)

    # Rolling std of |accel| over ~0.5 s, trailing window
    dt_med = float(np.median(np.diff(t)))
    nwin = max(5, int(round(0.5 / dt_med)))
    amag_std = pd.Series(amag).rolling(nwin, min_periods=nwin).std().values

    still = ((np.abs(gyro_dps) < gyro_tol_dps).all(axis=1) &
             (amag_std < accel_std_tol))          # NaN start compares False

    # Consecutive runs, kept if long enough (time-based, drop-proof)
    edges = np.flatnonzero(np.diff(
        np.concatenate(([False], still, [False])).astype(int)))
    runs = list(zip(edges[0::2], edges[1::2] - 1))
    return [(i0, i1) for i0, i1 in runs if t[i1] - t[i0] >= min_dur_s]


def sphere_fit_bias(pose_means, local_g):
    """Fit accel bias b so that || mean_i - b || = local_g for every pose.

    Two stages:
      1. Algebraic (linear) fit for a starting point: expanding
         ||m - b||^2 = g^2 gives  2 m . b - (||b||^2 - g^2) = ||m||^2,
         which is linear in (b, c) with c = ||b||^2 - g^2.
      2. Gauss-Newton refinement of the true geometric residual
         ||m - b|| - g via scipy least_squares.

    Returns (bias vector, per-pose residuals in m/s^2).
    """
    M = np.asarray(pose_means)

    # Stage 1: linear least squares
    A = np.hstack([2.0 * M, -np.ones((len(M), 1))])
    y = (M ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    b0 = sol[:3]

    # Stage 2: geometric refinement
    def residual(b):
        return np.linalg.norm(M - b, axis=1) - local_g

    fit = least_squares(residual, b0)
    return fit.x, residual(fit.x)


def main():
    parser = argparse.ArgumentParser(
        description='Sphere-fit IMU bias from a multi-pose log with the '
                    'sensor mounted. Writes imu_calibration.json.')
    parser.add_argument('logfile', nargs='?', default=None,
                        help='CSV in Data Logs (default: most recent)')
    parser.add_argument('--local-g', type=float, default=LOCAL_GRAVITY_MPS2,
                        metavar='M_S2',
                        help=f'Local gravity for the sphere radius '
                             f'(default {LOCAL_GRAVITY_MPS2}, Boulder CO)')
    parser.add_argument('--gyro-tol', type=float, default=5.0, metavar='DPS',
                        help='Per-axis gyro stillness threshold (default 5). '
                             'Poses are propped, not handheld, so keep tight.')
    parser.add_argument('--accel-std-tol', type=float, default=0.1,
                        metavar='M_S2',
                        help='Rolling |accel| std threshold for stillness '
                             '(default 0.1)')
    parser.add_argument('--min-pose-dur', type=float, default=5.0,
                        metavar='S',
                        help='Minimum still duration to count as a pose '
                             '(default 5)')
    parser.add_argument('--output', default=None,
                        help='Output JSON path (default: imu_calibration.json '
                             'next to this script)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Load log (most recent when unspecified, same rule as the pipeline) ---
    if args.logfile is None:
        candidates = glob.glob(os.path.join(script_dir, 'Data Logs',
                                            'imu_log_*.csv'))
        if not candidates:
            raise FileNotFoundError('No CSV files found in Data Logs')
        log_path = max(candidates, key=os.path.getmtime)
    else:
        log_path = args.logfile
    df = sa.load_log(log_path)
    source_log = os.path.basename(log_path if os.path.isabs(log_path)
                                  else os.path.join('Data Logs', log_path))
    t = df['t_s'].values

    # --- Find the still poses ---
    poses = find_still_poses(df, args.gyro_tol, args.accel_std_tol,
                             args.min_pose_dur)

    print()
    print('=' * 68)
    print('Mounted IMU Bias Characterization (sphere fit)')
    print(f'  Log:    {source_log}')
    print(f'  Poses found: {len(poses)}  (still >= {args.min_pose_dur:.0f} s, '
          f'gyro < {args.gyro_tol:.0f} dps, |accel| std < '
          f'{args.accel_std_tol} m/s^2)')
    print('=' * 68)

    if len(poses) < 4:
        raise ValueError(
            f'Only {len(poses)} still poses found; the sphere fit needs at '
            f'least 4 (8-10 recommended). Check the capture: each pose must '
            f'be hands-off and motionless for {args.min_pose_dur:.0f}+ s. '
            f'If poses were shorter, re-run with --min-pose-dur, or re-log.')

    accel_raw = df[['ax', 'ay', 'az']].values
    gyro_raw = df[['gx', 'gy', 'gz']].values
    pose_means = np.array([accel_raw[i0:i1 + 1].mean(axis=0)
                           for i0, i1 in poses])

    # --- Orientation diversity check ---
    # If every pose direction lies near one plane (e.g. the javelin was only
    # ROLLED about its long axis), the bias component perpendicular to that
    # plane is unconstrained and the fit will silently absorb garbage there.
    # The third singular value of the centered direction set measures
    # out-of-plane spread.
    dirs = pose_means / np.linalg.norm(pose_means, axis=1, keepdims=True)
    svals = np.linalg.svd(dirs - dirs.mean(axis=0), compute_uv=False)
    coplanar = svals[2] / svals[0] < 0.15 if svals[0] > 0 else True

    # --- Fit ---
    bias, resid = sphere_fit_bias(pose_means, args.local_g)

    # --- Per-pose table: lets a bad pose (bumped, still moving) stand out ---
    print(f"{'pose':<5} {'time (s)':>15} {'dur':>6} {'raw |a|':>9} "
          f"{'residual':>10}")
    print(f"{'-'*5} {'-'*15} {'-'*6} {'-'*9} {'-'*10}")
    for k, ((i0, i1), r) in enumerate(zip(poses, resid)):
        mag = np.linalg.norm(pose_means[k])
        flag = '  <- check this pose' if abs(r) > 0.05 else ''
        print(f"{k+1:<5} {t[i0]:>7.1f}-{t[i1]:<7.1f} {t[i1]-t[i0]:>5.1f}s "
              f"{mag:>8.3f} {r:>+9.4f}{flag}")
    print()
    print(f"Fit RMS residual: {np.sqrt(np.mean(resid**2)):.4f} m/s^2 "
          f"({to_mg(np.sqrt(np.mean(resid**2))):.1f} mg)")
    if np.sqrt(np.mean(resid**2)) > 0.05:
        print("WARNING: residuals are large for a static calibration. A pose "
              "was probably not fully still; re-log or drop the flagged pose "
              "by tightening --gyro-tol / --accel-std-tol.")
    if coplanar:
        print("WARNING: pose directions are nearly coplanar. The bias "
              "component perpendicular to that plane is poorly constrained. "
              "Add tip-up and tip-down poses and re-log. (This happens when "
              "the javelin was only rolled about its long axis.)")

    # --- Gyro bias: orientation-independent, pooled over all still poses ---
    durs = np.array([t[i1] - t[i0] for i0, i1 in poses])
    gyro_means = np.array([gyro_raw[i0:i1 + 1].mean(axis=0)
                           for i0, i1 in poses])
    gyro_bias_rads = np.average(gyro_means, axis=0, weights=durs)
    gyro_bias_dps = np.degrees(gyro_bias_rads)

    # --- Noise floors from the longest pose ---
    k_long = int(np.argmax(durs))
    i0, i1 = poses[k_long]
    accel_std = accel_raw[i0:i1 + 1].std(axis=0)
    gyro_std_dps = np.degrees(gyro_raw[i0:i1 + 1].std(axis=0))

    bias_mg = to_mg(bias)
    print()
    print(f"Accel bias:  x={bias[0]:+.4f}  y={bias[1]:+.4f}  "
          f"z={bias[2]:+.4f} m/s^2")
    print(f"             x={bias_mg[0]:+.1f}  y={bias_mg[1]:+.1f}  "
          f"z={bias_mg[2]:+.1f} mg")
    print(f"Gyro bias:   x={gyro_bias_dps[0]:+.4f}  "
          f"y={gyro_bias_dps[1]:+.4f}  z={gyro_bias_dps[2]:+.4f} dps")

    # --- Compare against the existing calibration before overwriting ---
    out_path = args.output or os.path.join(script_dir, 'imu_calibration.json')
    if os.path.exists(out_path):
        old = sa.load_calibration(out_path)
        ob = old['accel_bias_mps2']
        og = old['gyro_bias_dps']
        print()
        print(f"Previous calibration ({old.get('characterized_at', '?')[:10]}, "
              f"{old.get('gravity_axis', '?')}):")
        print(f"  accel old -> new:  "
              f"x {ob['x']:+.4f} -> {bias[0]:+.4f}   "
              f"y {ob['y']:+.4f} -> {bias[1]:+.4f}   "
              f"z {ob['z']:+.4f} -> {bias[2]:+.4f} m/s^2")
        print(f"  gyro  old -> new:  "
              f"x {og['x']:+.4f} -> {gyro_bias_dps[0]:+.4f}   "
              f"y {og['y']:+.4f} -> {gyro_bias_dps[1]:+.4f}   "
              f"z {og['z']:+.4f} -> {gyro_bias_dps[2]:+.4f} dps")

    # --- Write, same schema as characterize_bias.py so nothing downstream
    #     changes. gravity_axis is a descriptive string here because the
    #     sphere fit makes no axis assumption. ---
    result = {
        "source_log":              source_log,
        "characterized_at":        datetime.now(timezone.utc).isoformat(),
        "gravity_axis":            "none (mounted sphere fit)",
        "method":                  "sphere_fit_mounted",
        "local_gravity_mps2_used": args.local_g,
        "n_poses":                 len(poses),
        "fit_rms_residual_mps2":   round(float(np.sqrt(np.mean(resid**2))), 6),
        "accel_bias_mps2": {
            "x": round(float(bias[0]), 7),
            "y": round(float(bias[1]), 7),
            "z": round(float(bias[2]), 7),
        },
        "accel_noise_std_mps2": {
            "x": round(float(accel_std[0]), 7),
            "y": round(float(accel_std[1]), 7),
            "z": round(float(accel_std[2]), 7),
        },
        "gyro_bias_dps": {
            "x": round(float(gyro_bias_dps[0]), 7),
            "y": round(float(gyro_bias_dps[1]), 7),
            "z": round(float(gyro_bias_dps[2]), 7),
        },
        "gyro_noise_std_dps": {
            "x": round(float(gyro_std_dps[0]), 7),
            "y": round(float(gyro_std_dps[1]), 7),
            "z": round(float(gyro_std_dps[2]), 7),
        },
        "notes": (f"Sphere fit over {len(poses)} still poses with the sensor "
                  f"mounted; no gravity-axis assumption. All three accel bias "
                  f"components are true sensor offsets (mounting tilt does "
                  f"not enter). Bias only; scale factor not fitted."),
    }
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print()
    print(f"Calibration written: {os.path.relpath(out_path)}")


if __name__ == '__main__':
    main()
