"""
SPEAR throw window viewer: clip a throw log to just the throw.

The 60 s flash ring makes diagnostic plots mostly dead time. This script
finds the throw (same peak-based logic as the field pipeline), clips to a
few seconds before onset through the end of the log, and plots only that. It also marks a RELAXED quiet window: the stillest
1 s stretch of the pre-throw hold, chosen by minimum rolling variance
instead of the strict absolute tolerances (which handheld tremor can
fail). This is a viewer only - it changes nothing in the analysis
pipeline and writes no analysis products, just a PNG.

Usage:
    python plot_throw_window.py [logfile] [--pre S]

    logfile   throw_013, session_*.csv, etc. (default: most recent log)
    --pre     seconds of context before onset (default 5.0)
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

import spear_analysis as sa
from analyze_field_throw import all_logs, resolve_log


def main():
    parser = argparse.ArgumentParser(
        description='Plot just the throw portion of a log.')
    parser.add_argument('logfile', nargs='?', default=None,
                        help='Log to view (default: most recent)')
    parser.add_argument('--pre', type=float, default=5.0, metavar='S',
                        help='Seconds of context before onset (default 5)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'Data Logs')
    if args.logfile is None:
        logs = all_logs(data_dir)
        if not logs:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        log_path = logs[0]
    else:
        log_path = resolve_log(args.logfile, data_dir)
    log_name = os.path.basename(log_path)

    # Same calibration path as the field pipeline
    df = sa.load_log(log_path)
    cal_path = os.path.join(script_dir, 'imu_calibration.json')
    if os.path.exists(cal_path):
        df = sa.apply_bias(df, calibration=sa.load_calibration(cal_path))
    else:
        df = sa.apply_bias(df, bias=sa.calibrate_bias(df, 0.0, 3.0))

    params = sa.SpearParams()
    phases = sa.detect_phases(df, params)
    t = df['t_s'].values

    if phases.onset_idx is None:
        raise SystemExit(f"No throw onset in {log_name}; nothing to clip. "
                         "Use analyze_field_throw.py --diagnostic instead.")

    onset = phases.onset_idx
    i_start = int(np.searchsorted(t, t[onset] - args.pre))

    # Relaxed quiet: stillest 1 s of the pre-onset context. This is what
    # the eye calls "the quiet part before the throw"; the strict detector
    # may have rejected it (tremor above the absolute tolerances).
    still = sa.find_stillest_window(phases.rolling_var, t, i_start, onset)

    # Report what the strict detector would have said about that stretch
    print(f"{log_name}: onset t={t[onset]:.2f} s, "
          f"showing {t[i_start]:.2f} s to end of log ({t[-1]:.2f} s)")
    if still is not None:
        j0, j1 = still
        gyro_dps = np.degrees(df[['gx_corr', 'gy_corr', 'gz_corr']].values)
        seg_acc = np.abs(phases.accel_mag[j0:j1 + 1] - params.local_g)
        seg_gyr = np.abs(gyro_dps[j0:j1 + 1])
        print(f"  relaxed quiet window: {t[j0]:.2f} - {t[j1]:.2f} s")
        print(f"    | |a|-g |  max {seg_acc.max():.3f} m/s^2  "
              f"(strict tol {params.quiet_accel_tol})")
        print(f"    |gyro|     max {seg_gyr.max():.1f} dps    "
              f"(strict tol {params.quiet_gyro_tol})")
        strict = (seg_acc.max() <= params.quiet_accel_tol and
                  seg_gyr.max() < params.quiet_gyro_tol)
        print(f"    strict quiet detector would "
              f"{'ACCEPT' if strict else 'REJECT'} this stretch")

    # --- Plot: clipped |accel| and log-variance, phase overlays ---
    sl = slice(i_start, len(t))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.suptitle(f"Throw window: {log_name}  "
                 f"(onset - {args.pre:.0f} s to freeze)", fontsize=12)

    ax1.plot(t[sl], phases.accel_mag[sl], color='steelblue', linewidth=0.9)
    ax1.set_ylabel('|accel| (m/s^2)')
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(t[sl], phases.rolling_var[sl], color='steelblue',
                 linewidth=0.9)
    ax2.set_ylabel('rolling var of |accel| (log)')
    ax2.set_xlabel('Time (s)')
    ax2.grid(True, alpha=0.3, which='both')

    for ax in (ax1, ax2):
        if still is not None:
            ax.axvspan(t[still[0]], t[still[1]], alpha=0.20, color='green',
                       label='relaxed quiet (stillest 1 s)')
        ax.axvline(t[onset], color='orange', linewidth=1.2, label='onset')
        if phases.flight_start is not None:
            ax.axvline(t[phases.flight_start], color='blue', linewidth=1.2,
                       linestyle='--', label='FLIGHT entry')
        if phases.impact_idx is not None:
            ax.axvline(t[phases.impact_idx], color='red', linewidth=1.2,
                       label='IMPACT')

    ax1.legend(loc='upper right', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    plt.tight_layout()

    base = os.path.splitext(log_name)[0]
    png_path = os.path.join(os.path.dirname(log_path),
                            f"{base}_window.png")
    fig.savefig(png_path, dpi=150)
    print(f"Saved: {png_path}")
    plt.show()


if __name__ == '__main__':
    main()
