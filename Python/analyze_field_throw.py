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

The diagnostic PNG is ALWAYS saved next to the analyzed log
(<logname>_diagnostic.png), zoomed to the throw (onset - 5 s to impact
+ 3 s). --diagnostic additionally opens it in an interactive window.
Detection is peak-based (pull peak to impact rising edge) and needs no
threshold tuning; see SpearParams comments in spear_analysis.py.

All analysis logic lives in spear_analysis.py; this script only wires it up.
"""

import argparse
import datetime
import glob
import os

import spear_analysis as sa

# ===========================================================================
# TRIM WINDOW - edit these to change what the trimmed copy keeps
# ===========================================================================
TRIM_PRE_ONSET_S = 10.0    # seconds kept BEFORE throw onset (window start)
TRIM_POST_IMPACT_S = 2.0  # seconds kept AFTER impact

# V=0 WINDOW - where velocity integration starts. The stillest stretch of
# V0_STILL_DUR seconds (minimum |accel| variance) inside the V0_LOOKBACK_S
# seconds before onset anchors v=0 at its end. This is the green area on
# the diagnostic plot. Raise V0_LOOKBACK_S if your hold happens earlier;
# raise V0_STILL_DUR to demand a longer hold.
V0_LOOKBACK_S = 4.0       # s before onset searched for the hold
V0_STILL_DUR = 1.0        # s of stillness that anchors v=0
# ===========================================================================

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
                        help='Open the diagnostic plot interactively '
                             '(the PNG is saved either way)')
    parser.add_argument('--distance', type=float, default=None, metavar='M',
                        help='Tape-measured throw distance in meters; the '
                             'summary prints the ballistic prediction error '
                             'against it')
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

    # --- Parameters: constructed once here, passed down everywhere.
    # Trim and v=0 windows come from the config block at the top of this
    # file. ---
    params = sa.SpearParams(trim_pre_onset_s=TRIM_PRE_ONSET_S,
                            trim_post_impact_s=TRIM_POST_IMPACT_S,
                            v0_lookback_s=V0_LOOKBACK_S,
                            v0_still_dur=V0_STILL_DUR)

    cal_path = os.path.join(script_dir, 'imu_calibration.json')
    calibration = sa.load_calibration(cal_path) if os.path.exists(cal_path) \
        else None

    def load_and_correct(path):
        df_raw = sa.load_log(path)
        if calibration is not None:
            return df_raw, sa.apply_bias(df_raw, calibration=calibration)
        return df_raw, sa.apply_bias(
            df_raw, bias=sa.calibrate_bias(df_raw, t_start=0.0, t_end=3.0))

    if calibration is not None:
        print(f"Calibration source: imu_calibration.json  "
              f"(characterized {calibration['characterized_at'][:10]}, "
              f"gravity {calibration['gravity_axis']})")
    else:
        print("Calibration source: per-log stationary window "
              "(no imu_calibration.json found)")

    # --- Trim first, analyze the trimmed copy ---
    # The 60 s ring is mostly dead time. Locate the throw in the source,
    # write <log>_trimmed.csv (window set by SpearParams.trim_pre_onset_s /
    # trim_post_impact_s), then run the full analysis on the TRIMMED file,
    # so every timestamp in the summary and plots starts near 0. Sources
    # that are already trimmed, and logs with no throw (bench captures),
    # are analyzed directly.
    df_raw, df = load_and_correct(log_path)
    if not log_path.endswith('_trimmed.csv'):
        pre_phases = sa.detect_phases(df, params)
        trimmed = sa.write_trimmed_copy(df_raw, pre_phases, params, log_path)
        if trimmed is not None:
            t = df['t_s'].values
            t0 = max(t[0], t[pre_phases.onset_idx] - params.trim_pre_onset_s)
            print(f"Trimmed {os.path.basename(log_path)} -> "
                  f"{os.path.basename(trimmed)}  "
                  f"(t=0 was {t0:.2f} s in the source)")
            log_path = trimmed
            df_raw, df = load_and_correct(log_path)

    # --- Run the pipeline and report ---
    res = sa.analyze_field_log(df, params,
                               log_name=os.path.basename(log_path))
    sa.print_field_summary(res, measured_distance_m=args.distance)

    # The diagnostic PNG is always saved next to the analyzed file;
    # --diagnostic additionally opens it in an interactive window.
    base = os.path.splitext(os.path.basename(log_path))[0]
    png_path = os.path.join(os.path.dirname(log_path),
                            f"{base}_diagnostic.png")
    sa.plot_field_diagnostic(res, png_path, show=args.diagnostic)


if __name__ == '__main__':
    main()
