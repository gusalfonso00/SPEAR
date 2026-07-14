"""
Characterize IMU static bias and noise from a fully stationary capture.
Writes imu_calibration.json for use in subsequent analyses.

Usage:
    python characterize_bias.py [--gravity-axis {+x,-x,+y,-y,+z,-z}] [--local-g M_S2]

--gravity-axis specifies which axis carries the 1g gravity signal in the sensor
output. The stored bias is (measured mean - expected gravity), so the JSON holds
true sensor offset rather than raw mean. Re-run whenever the IMU is re-mounted.

--local-g overrides the local gravitational acceleration used in the gravity
subtraction step. Default is Boulder, CO (9.7966 m/s²). Standard gravity
(9.80665) is ~10 mg too high at elevation and should not be used for subtraction.
"""

import argparse
import json
import math
import os
import glob
from datetime import datetime, timezone

import spear_analysis as sa

G = 9.80665  # standard gravity (m/s²) - unit conversion only, do not use for subtraction

LOCAL_GRAVITY_MPS2 = 9.7966  # Boulder, CO: 40.01N, ~1655m elevation
                              # Theoretical from IGF80 + free-air correction.
                              # Standard gravity (9.80665) is ~10 mg too high here.
                              # Override via --local-g if running elsewhere.

# Expected acceleration direction per axis (signs only; magnitude set at runtime by local_g).
# Values here use G as a placeholder - they are NOT used directly in subtraction.
GRAVITY_MAP = {
    '+x': ( G,  0,  0),
    '-x': (-G,  0,  0),
    '+y': ( 0,  G,  0),
    '-y': ( 0, -G,  0),
    '+z': ( 0,  0,  G),
    '-z': ( 0,  0, -G),
}


def to_mg(mps2):
    return mps2 * 1000.0 / G


def to_dps(rads):
    return rads * 180.0 / math.pi


def main():
    parser = argparse.ArgumentParser(
        description='Characterize IMU static bias and noise. Writes imu_calibration.json.')
    parser.add_argument(
        '--gravity-axis', default='-z',
        choices=list(GRAVITY_MAP.keys()),
        help='Axis where gravity appears in sensor output. '
             'Example: +z means Z reads ~+9.81 m/s² when flat. Default: -z')
    parser.add_argument(
        '--local-g', type=float, default=LOCAL_GRAVITY_MPS2, metavar='M_S2',
        help=f'Local gravitational acceleration in m/s² used for gravity subtraction '
             f'(default: {LOCAL_GRAVITY_MPS2} - Boulder CO). Override for your location.')
    args = parser.parse_args()

    local_g     = args.local_g
    grav_letter = args.gravity_axis[1]      # 'x', 'y', or 'z'
    grav_sign   = args.gravity_axis[0]      # '+' or '-'
    grav_sign_f = +1.0 if grav_sign == '+' else -1.0
    # Build expected gravity vector using local g - magnitude is location-dependent
    _gv = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    _gv[grav_letter] = grav_sign_f * local_g
    grav = (_gv['x'], _gv['y'], _gv['z'])
    grav_mg_exp = grav_sign_f * to_mg(local_g)  # expected in mg (standard-g unit convention)

    # --- Find and load most recent log ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir   = os.path.join(script_dir, "Data Logs")
    csv_files  = (glob.glob(os.path.join(data_dir, "imu_log_*.csv")) +
                  glob.glob(os.path.join(data_dir, "session_*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    latest     = max(csv_files, key=os.path.getmtime)
    source_log = os.path.basename(latest)

    df = sa.load_log(latest)

    # --- Trim: drop first 2s (settling/handling) and last 1s (stop artifact) ---
    t       = df['t_s']
    t_start = t.iloc[0] + 2.0
    t_end   = t.iloc[-1] - 1.0

    if t_end <= t_start:
        raise ValueError(
            f"Log too short after trimming ({t.iloc[-1]:.1f}s total). Need at least 3s.")

    mask   = (t >= t_start) & (t <= t_end)
    window = df.loc[mask]
    n      = len(window)
    duration = t_end - t_start

    # --- Per-axis statistics ---
    accel_cols = ['ax', 'ay', 'az']
    gyro_cols  = ['gx', 'gy', 'gz']

    means_a = {c: window[c].mean() for c in accel_cols}
    stds_a  = {c: window[c].std()  for c in accel_cols}
    means_g = {c: window[c].mean() for c in gyro_cols}
    stds_g  = {c: window[c].std()  for c in gyro_cols}

    # Bias = measured mean - expected gravity contribution per axis
    accel_bias = {
        'x': means_a['ax'] - grav[0],
        'y': means_a['ay'] - grav[1],
        'z': means_a['az'] - grav[2],
    }
    gyro_bias_dps = {axis: to_dps(means_g[f'g{axis}']) for axis in ('x', 'y', 'z')}

    # --- Summary table ---
    print()
    print('=' * 68)
    print('IMU Bias/Noise Characterization')
    print(f'  Log:        {source_log}')
    print(f'  Window:     {t_start:.2f}s - {t_end:.2f}s  ({duration:.1f}s, {n} samples)')
    print(f'  Gravity:    {args.gravity_axis}  ({grav_mg_exp:+.1f} mg on {grav_letter.upper()}, local g = {local_g} m/s²)')
    print('=' * 68)
    print(f"{'Channel':<9} {'Mean':>13} {'Std':>12}   Sanity")
    print(f"{'-'*9} {'-'*13} {'-'*12}   {'-'*30}")

    for col, letter in zip(accel_cols, ['x', 'y', 'z']):
        mean_mg = to_mg(means_a[col])
        std_mg  = to_mg(stds_a[col])
        if letter == grav_letter:
            delta  = abs(mean_mg - grav_mg_exp)
            sanity = (f"OK  (gravity axis, delta {delta:+.0f} mg)"
                      if delta < 100 else
                      f"WARN (expected {grav_mg_exp:+.0f} mg, got {mean_mg:+.0f})")
        else:
            sanity = ("OK"
                      if abs(mean_mg) < 200 else
                      f"WARN (large: {mean_mg:+.0f} mg - tilt?)")
        print(f"a{letter:<8} {mean_mg:>+12.1f} mg {std_mg:>10.2f} mg   {sanity}")

    print()
    for col, letter in zip(gyro_cols, ['x', 'y', 'z']):
        mean_d = to_dps(means_g[col])
        std_d  = to_dps(stds_g[col])
        if abs(mean_d) < 2.0:
            sanity = "OK"
        elif abs(mean_d) < 5.0:
            sanity = "WARN (marginal)"
        else:
            sanity = "FAIL (check gyro)"
        print(f"g{letter:<8} {mean_d:>+12.4f} dps {std_d:>9.4f} dps   {sanity}")

    print('=' * 68)

    # --- Write JSON ---
    non_grav = {'x', 'y', 'z'} - {grav_letter}
    non_grav_str = '/'.join(sorted(non_grav)).upper()
    notes = (
        f"{grav_letter.upper()}-axis bias has gravity ({args.gravity_axis}, {local_g} m/s²) subtracted; "
        f"stored value is true sensor offset from expected. "
        f"{non_grav_str} bias may be confounded with mounting tilt - "
        f"re-characterize with a proper fixture for full 3-axis calibration."
    )

    result = {
        "source_log":              source_log,
        "characterized_at":        datetime.now(timezone.utc).isoformat(),
        "gravity_axis":            args.gravity_axis,
        "local_gravity_mps2_used": local_g,
        "duration_used_s":     round(duration, 3),
        "n_samples_used":      n,
        "accel_bias_mps2": {
            "x": round(accel_bias['x'], 7),
            "y": round(accel_bias['y'], 7),
            "z": round(accel_bias['z'], 7),
        },
        "accel_noise_std_mps2": {
            "x": round(stds_a['ax'], 7),
            "y": round(stds_a['ay'], 7),
            "z": round(stds_a['az'], 7),
        },
        "gyro_bias_dps": {
            "x": round(gyro_bias_dps['x'], 7),
            "y": round(gyro_bias_dps['y'], 7),
            "z": round(gyro_bias_dps['z'], 7),
        },
        "gyro_noise_std_dps": {
            "x": round(to_dps(stds_g['gx']), 7),
            "y": round(to_dps(stds_g['gy']), 7),
            "z": round(to_dps(stds_g['gz']), 7),
        },
        "notes": notes,
    }

    out_path = os.path.join(script_dir, "imu_calibration.json")
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print()
    print(f"Calibration written: {os.path.relpath(out_path)}")
    bias_mg = {k: to_mg(v) for k, v in accel_bias.items()}
    print(f"  accel bias (gravity removed): "
          f"x={bias_mg['x']:+.1f} mg  y={bias_mg['y']:+.1f} mg  z={bias_mg['z']:+.1f} mg")
    print(f"  gyro bias:                   "
          f"x={gyro_bias_dps['x']:+.4f} dps  "
          f"y={gyro_bias_dps['y']:+.4f} dps  "
          f"z={gyro_bias_dps['z']:+.4f} dps")

    # Gravity subtraction sanity check: old (standard g) vs new (local g)
    grav_mean    = means_a[f'a{grav_letter}']
    bias_std_g   = grav_mean - grav_sign_f * G           # what bias would be with 9.80665
    bias_local_g = accel_bias[grav_letter]               # what bias is with local g
    print(f"\n  {grav_letter.upper()}-axis bias - standard g vs local g:")
    print(f"    standard g (9.80665 m/s²):  {bias_std_g:+.4f} m/s²  ({to_mg(bias_std_g):+.1f} mg)")
    print(f"    local g    ({local_g} m/s²):  {bias_local_g:+.4f} m/s²  ({to_mg(bias_local_g):+.1f} mg)")
    print(f"    difference:                  {bias_local_g - bias_std_g:+.4f} m/s²  ({to_mg(bias_local_g - bias_std_g):+.2f} mg)")


if __name__ == '__main__':
    main()
