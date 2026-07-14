"""
SPEAR field throw analysis.

Runs the full throw pipeline on one log: phase detection (QUIET -> ACTIVE ->
FLIGHT -> IMPACT), arbitrary-orientation attitude initialization from the
first quiet window, complementary filter with per-sample dt, windowed
velocity integration (v=0 at the last quiet window before onset, stop at
impact), release speed and elevation angle, vacuum ballistic
self-consistency check, clipping audit, and attitude truth metric.

Usage:
    python analyze_field_throw.py [logfile.csv] [--diagnostic]

With no logfile the most recent capture in Data Logs is used.
--diagnostic renders the variance tuning plot (|accel| plus log-scale
rolling variance with phase overlays), saves it as a PNG next to the log,
and shows it. Use it on the first real throw of field day to tune
SpearParams.flight_var_thresh (the summary prints a suggested value).

All analysis logic lives in spear_analysis.py; this script only wires it up.
"""

import argparse
import os

import spear_analysis as sa


def main():
    parser = argparse.ArgumentParser(
        description='Analyze a javelin throw log (phases, release, audits).')
    parser.add_argument('logfile', nargs='?', default=None,
                        help='CSV in Data Logs (default: most recent)')
    parser.add_argument('--diagnostic', action='store_true',
                        help='Render and save the variance tuning plot')
    args = parser.parse_args()

    # Resolve the log path up front (same rule as load_log: most recent when
    # unspecified) so the diagnostic PNG can be named after the actual file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'Data Logs')
    if args.logfile is None:
        candidates = []
        for pattern in ('imu_log_*.csv', 'session_*.csv',
                        os.path.join('throws', 'throw_*.csv')):
            candidates += sa.glob.glob(os.path.join(data_dir, pattern))
        if not candidates:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        log_path = max(candidates, key=os.path.getmtime)
    elif os.path.isabs(args.logfile):
        log_path = args.logfile
    else:
        log_path = os.path.join(data_dir, args.logfile)

    # --- Load and bias-correct, same calibration priority as Step 1/2 ---
    df = sa.load_log(log_path)
    cal_path = os.path.join(script_dir, 'imu_calibration.json')
    if os.path.exists(cal_path):
        calibration = sa.load_calibration(cal_path)
        print(f"Calibration source: imu_calibration.json  "
              f"(characterized {calibration['characterized_at'][:10]}, "
              f"gravity {calibration['gravity_axis']})")
        df = sa.apply_bias(df, calibration=calibration)
    else:
        print("Calibration source: per-log stationary window "
              "(no imu_calibration.json found)")
        bias = sa.calibrate_bias(df, t_start=0.0, t_end=3.0)
        df = sa.apply_bias(df, bias)

    # --- Parameters: constructed once here, passed down everywhere ---
    params = sa.SpearParams()

    # --- Run the pipeline and report ---
    res = sa.analyze_field_log(df, params,
                               log_name=os.path.basename(log_path))
    sa.print_field_summary(res)

    if args.diagnostic:
        # PNG lands next to the log file so field notes and plots stay together
        base = os.path.splitext(os.path.basename(log_path))[0]
        png_path = os.path.join(os.path.dirname(log_path),
                                f"{base}_diagnostic.png")
        sa.plot_field_diagnostic(res, png_path, show=True)


if __name__ == '__main__':
    main()
