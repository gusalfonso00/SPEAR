"""
SPEAR field throw analysis.

Runs the full throw pipeline on one log: phase detection (QUIET -> ACTIVE ->
FLIGHT -> IMPACT), arbitrary-orientation attitude initialization from the
first quiet window, complementary filter with per-sample dt, windowed
velocity integration (v=0 at the last quiet window before onset, stop at
impact), release speed and elevation angle, vacuum ballistic
self-consistency check, clipping audit, and attitude truth metric.

Usage:
    python analyze_field_throw.py [logfile] [--diagnostic] [--list]

Selecting a log:
    (nothing)           most recent capture: session logs and decoded
                        flash dumps both count
    throw_001           a flash dump by short name (.csv optional; found
                        in Data Logs/throws/ automatically)
    session_...csv      a live-stream session log by name
    --list              show every analyzable log, newest first, and exit

The variance diagnostic PNG is ALWAYS saved next to the analyzed log
(<logname>_diagnostic.png). --diagnostic additionally opens it in an
interactive window. Use that on the first real throw of field day to tune
SpearParams.flight_var_thresh (the summary prints a suggested value).

All analysis logic lives in spear_analysis.py; this script only wires it up.
"""

import argparse
import datetime
import glob
import os

import spear_analysis as sa

# Every place an analyzable CSV can live, relative to Data Logs
LOG_PATTERNS = ('imu_log_*.csv', 'session_*.csv',
                os.path.join('throws', 'throw_*.csv'))


def all_logs(data_dir):
    """Every analyzable CSV, newest first."""
    found = []
    for pattern in LOG_PATTERNS:
        found += glob.glob(os.path.join(data_dir, pattern))
    return sorted(found, key=os.path.getmtime, reverse=True)


def resolve_log(arg, data_dir):
    """Turn whatever the operator typed into a real path.

    Accepts an absolute path, a filename in Data Logs, a filename in
    Data Logs/throws, and the .csv extension is optional, so all of these
    work: throw_001, throw_001.csv, session_2026-07-14_20-32-22.csv.
    """
    if os.path.isabs(arg):
        return arg
    names = [arg] if arg.endswith('.csv') else [arg, arg + '.csv']
    for name in names:
        for base in (data_dir, os.path.join(data_dir, 'throws')):
            p = os.path.join(base, name)
            if os.path.exists(p):
                return p
    available = [os.path.relpath(p, data_dir) for p in all_logs(data_dir)[:8]]
    raise FileNotFoundError(
        f"'{arg}' not found in Data Logs or Data Logs/throws. "
        f"Most recent logs: {', '.join(available) if available else '(none)'}. "
        f"Run with --list to see everything.")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze a javelin throw log (phases, release, audits).')
    parser.add_argument('logfile', nargs='?', default=None,
                        help='Log to analyze: throw_001, session_*.csv, or a '
                             'path (default: most recent)')
    parser.add_argument('--diagnostic', action='store_true',
                        help='Open the variance tuning plot interactively '
                             '(the PNG is saved either way)')
    parser.add_argument('--list', action='store_true',
                        help='List analyzable logs, newest first, and exit')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'Data Logs')

    if args.list:
        logs = all_logs(data_dir)
        if not logs:
            print("No logs found in Data Logs.")
            return
        print(f"{'log':<42} {'modified':<17} {'size':>8}")
        for p in logs:
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(p))
            print(f"{os.path.relpath(p, data_dir):<42} "
                  f"{mtime:%Y-%m-%d %H:%M}  {os.path.getsize(p):>8}")
        return

    if args.logfile is None:
        logs = all_logs(data_dir)
        if not logs:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        log_path = logs[0]
    else:
        log_path = resolve_log(args.logfile, data_dir)

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

    # The diagnostic PNG is always saved next to the log so every analyzed
    # capture has its plot on disk; --diagnostic additionally opens it in an
    # interactive window for threshold tuning in the field.
    base = os.path.splitext(os.path.basename(log_path))[0]
    png_path = os.path.join(os.path.dirname(log_path),
                            f"{base}_diagnostic.png")
    sa.plot_field_diagnostic(res, png_path, show=args.diagnostic)


if __name__ == '__main__':
    main()
