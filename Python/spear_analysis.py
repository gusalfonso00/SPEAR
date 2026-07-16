"""
SPEAR analysis function library.
Functions for loading IMU logs, calibrating bias, integrating to velocity,
and plotting results.

Import as: import spear_analysis as sa
"""

import json
import math
import os
import glob
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

import spear_filter as sf

# Standard gravity, m/s^2. Unit conversion ONLY (sensor full scale is defined
# in g units, mg is defined against standard gravity). Never use this for
# gravity subtraction; that is what SpearParams.local_g is for.
G_STD = 9.80665


# ---------------------------------------------------------------------------
# Central parameter block: every tunable in the throw pipeline lives here.
# Nothing elsewhere in this module hardcodes a threshold. Main scripts build
# one SpearParams and pass it down.
# ---------------------------------------------------------------------------

@dataclass
class SpearParams:
    """All tunables for the field throw pipeline, with units.

    Change a value here (or construct SpearParams(field=value) in a script);
    do not edit thresholds inside functions.
    """

    # Local gravitational acceleration, m/s^2. Boulder CO default; must match
    # the value used by characterize_bias.py. Change when running elsewhere.
    local_g: float = 9.7966

    # QUIET detection. Both conditions must hold per sample (bias-corrected):
    # | |accel| - local_g | <= quiet_accel_tol AND every gyro axis magnitude
    # < quiet_gyro_tol. Loosened for handheld tremor: the thrower holds the
    # javelin roughly still in hand, not on a bench. Tighten for bench work.
    quiet_accel_tol: float = 0.3    # m/s^2
    quiet_gyro_tol: float = 10.0    # dps, per axis
    quiet_min_dur: float = 0.5      # s of consecutive quiet to count as a window

    # Throw onset: first sample with |accel| > onset_thresh_g * local_g.
    # 3 g is far above handheld motion and far below any real throw pull.
    onset_thresh_g: float = 3.0     # multiples of local_g

    # v = 0 point: velocity integration starts at the END of the stillest
    # v0_still_dur stretch found within v0_lookback_s before onset (chosen
    # by minimum rolling variance, no absolute tolerance - handheld tremor
    # fails the strict quiet test but is fine for zeroing velocity).
    # Only data from right before the throw feeds the velocity integral;
    # nothing earlier leaks in. THESE are the knobs for where integration
    # starts: raise v0_lookback_s if your pre-throw hold happens earlier
    # than 4 s before the pull.
    v0_lookback_s: float = 4.0      # s before onset to search for the hold
    v0_still_dur: float = 1.0       # s of stillness to anchor v=0

    # Minimum in-gate samples for the v=0 window attitude check to print a
    # mean; below this the summary prints n/a with the count instead of a
    # number averaged over almost nothing.
    attitude_check_min_samples: int = 10

    # Trimmed copy: analyze_field_throw.py writes <log>_trimmed.csv next to
    # the source, holding only the good part of the record. The values used
    # come from the TRIM WINDOW block at the TOP of analyze_field_throw.py,
    # which overrides these defaults - edit there.
    trim_pre_onset_s: float = 5.0   # s kept before onset
    trim_post_impact_s: float = 2.0 # s kept after impact

    # FLIGHT detection: peak-based, sized for short field throws (10-30 m,
    # flight under ~2 s). The impact is the loudest event in the log (all
    # real throws railed or nearly railed the accel: 322-478 m/s^2), so:
    #   impact anchor = global max of |accel|
    #   onset         = first >3g sample within throw_lookback_s before it
    #   flight start  = the pull peak (max |accel| within release_window_s
    #                   after onset) - throw_013 example: 56.127 s
    #   flight end    = rising edge of the impact peak (walk back from the
    #                   max while |accel| >= impact_rise_frac of the peak)
    # No variance thresholds to tune. A variance-band detector was tried
    # first and needed per-session tuning (flight vibration 89-278 vs pull
    # 343-1760: thin margins). Revisit for long throws where flight
    # exceeds throw_lookback_s or impact is quieter than the pull.
    flight_var_window: float = 0.3  # s, rolling variance window (stats and
                                    # diagnostic plots only, not detection)
    throw_lookback_s: float = 3.0   # s before the impact peak to search for
                                    # onset; must exceed max flight time
    release_window_s: float = 0.5   # s after onset containing the pull peak
    impact_rise_frac: float = 0.2   # walk-back stops when |accel| falls
                                    # below this fraction of the impact peak
    flight_sane_min: float = 0.4    # s; flight shorter = needs review
    flight_sane_max: float = 4.0    # s; flight longer = needs review

    # Clipping audit: a sample counts as near-rail when any axis is within
    # clip_rail_frac of full scale. 0.98 rather than 1.0 because the sensor
    # rails a hair below the nominal range.
    clip_rail_frac: float = 0.98    # fraction of full scale

    # Height of the release point above the landing plane, m. Used only by
    # the vacuum ballistic prediction. Thrower-dependent: measure shoulder
    # height plus arm extension at release for your thrower.
    release_height_m: float = 2.0   # m

    # Sensor full scale, matching firmware register settings. Change only if
    # the firmware ranges change.
    accel_fs_g: float = 32.0        # g
    gyro_fs_dps: float = 2000.0     # dps


def load_log(filename=None):
    """
    Load an IMU log CSV from Data Logs.

    Parameters
    ----------
    filename : str, optional
        Specific CSV to load. If None, loads the most recent log.

    Returns
    -------
    df : pandas.DataFrame
        Sensor data with an added 't_s' column (seconds from start).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "Data Logs")

    if filename is None:
        # Three capture sources share the CSV format: legacy imu_log_*
        # captures, live session_* logs from log_imu.py, and decoded flash
        # dumps in throws/. "Most recent" searches all of them.
        csv_files = []
        for pattern in ("imu_log_*.csv", "session_*.csv",
                        os.path.join("throws", "throw_*.csv")):
            csv_files += glob.glob(os.path.join(data_dir, pattern))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        filename = max(csv_files, key=os.path.getmtime)
        print(f"Loading most recent log: {os.path.basename(filename)}")
    else:
        if not os.path.isabs(filename):
            filename = os.path.join(data_dir, filename)
        print(f"Loading: {os.path.basename(filename)}")

    # log_imu.py writes a header row, but headerless captures exist too.
    # Sniff the first line: if it starts with 'seq' it's a header, otherwise
    # assume the standard 9-column order and assign names ourselves.
    columns = ['seq', 'ms', 'temp_C', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
    with open(filename) as f:
        has_header = f.readline().lstrip().startswith('seq')
    if has_header:
        df = pd.read_csv(filename)
    else:
        df = pd.read_csv(filename, header=None, names=columns)

    if len(df) == 0:
        raise ValueError(f"{os.path.basename(filename)} is empty "
                         f"(header only, no samples). Was the ESP32 streaming?")

    df['t_s'] = (df['ms'] - df['ms'].iloc[0]) / 1000.0
    return df


def load_calibration(filename=None):
    """
    Load imu_calibration.json written by characterize_bias.py.

    Parameters
    ----------
    filename : str, optional
        Path to JSON file. Defaults to imu_calibration.json next to this module.

    Returns
    -------
    dict
        Calibration data including accel_bias_mps2, gyro_bias_dps, etc.
    """
    if filename is None:
        filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'imu_calibration.json')
    with open(filename) as f:
        return json.load(f)


def calibrate_bias(df, t_start=0.0, t_end=3.0):
    """
    Compute average sensor readings during a stationary window.
    Use this to estimate sensor bias before applying corrections.

    Parameters
    ----------
    df : pandas.DataFrame
        Sensor data with 't_s' column.
    t_start, t_end : float
        Time window in seconds for bias estimation. The IMU should be
        stationary during this window.

    Returns
    -------
    bias : dict
        Mean values for each accel and gyro axis.
    """
    mask = (df['t_s'] >= t_start) & (df['t_s'] <= t_end)
    window = df.loc[mask]

    if len(window) < 10:
        raise ValueError(
            f"Calibration window {t_start}-{t_end}s contains only "
            f"{len(window)} samples. Need a longer stationary period."
        )

    bias = {axis: window[axis].mean() for axis in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']}
    print(f"Calibrated over {len(window)} samples ({t_start}-{t_end}s)")
    return bias


def apply_bias(df, bias=None, calibration=None):
    """
    Subtract bias values from raw sensor readings.

    Parameters
    ----------
    df : pandas.DataFrame
        Sensor data.
    bias : dict, optional
        Bias values from calibrate_bias(). Keys: ax, ay, az, gx, gy, gz.
        Used when no calibration file is available (per-log fallback).
    calibration : dict, optional
        Calibration dict from load_calibration(). Takes precedence over bias.
        Gyro bias is stored in dps and converted to rad/s internally.

    Returns
    -------
    df : pandas.DataFrame
        Same DataFrame with new '_corr' columns added.
    """
    if calibration is not None:
        # Build bias dict in sensor units (m/s² for accel, rad/s for gyro)
        c = calibration
        bias = {
            'ax': c['accel_bias_mps2']['x'],
            'ay': c['accel_bias_mps2']['y'],
            'az': c['accel_bias_mps2']['z'],
            'gx': c['gyro_bias_dps']['x'] * math.pi / 180.0,
            'gy': c['gyro_bias_dps']['y'] * math.pi / 180.0,
            'gz': c['gyro_bias_dps']['z'] * math.pi / 180.0,
        }
    if bias is None:
        raise ValueError("Provide either bias (from calibrate_bias) or calibration (from load_calibration).")
    df = df.copy()
    for axis in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']:
        df[f'{axis}_corr'] = df[axis] - bias[axis]
    return df


def integrate_velocity(df, axes=('ax_corr', 'ay_corr', 'az_corr')):
    """
    Cumulative trapezoidal integration of acceleration to velocity.

    Parameters
    ----------
    df : pandas.DataFrame
        Sensor data with 't_s' column and the specified accel columns.
    axes : tuple of str
        Names of acceleration columns to integrate.

    Returns
    -------
    df : pandas.DataFrame
        Same DataFrame with 'vx', 'vy', 'vz' columns added.
    """
    df = df.copy()
    vel_names = ['vx', 'vy', 'vz']
    t = df['t_s'].values

    for accel_col, vel_col in zip(axes, vel_names):
        a = df[accel_col].values
        # cumulative_trapezoid returns one fewer element than input;
        # pad with leading 0 (velocity at t=0 is 0)
        v = integrate.cumulative_trapezoid(a, t, initial=0.0)
        df[vel_col] = v

    return df


def plot_acc_and_vel(df, title=None):
    """
    Two-row plot: corrected acceleration on top, integrated velocity below.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 't_s', 'ax_corr', 'ay_corr', 'az_corr', 'vx', 'vy', 'vz'.
    title : str, optional
        Figure title.
    """
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    for col, label in zip(['ax_corr', 'ay_corr', 'az_corr'], ['ax', 'ay', 'az']):
        axes[0].plot(df['t_s'], df[col], label=label)
    axes[0].set_ylabel('Accel, bias-corrected (m/s²)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    for col, label in zip(['vx', 'vy', 'vz'], ['vx', 'vy', 'vz']):
        axes[1].plot(df['t_s'], df[col], label=label)
    axes[1].set_ylabel('Velocity (m/s)')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    duration_min = df['t_s'].iloc[-1] / 60.0
    fig.text(0.98, 0.98, f'Duration: {duration_min:.1f} min',
             ha='right', va='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray'))

    if title:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Link characterization: UDP packet loss from sequence numbers
# ---------------------------------------------------------------------------
# UDP is fire-and-forget: a lost packet leaves no row and no timestamp in the
# log. All loss math therefore comes from the seq column. A gap between
# consecutive received seq values is attributed to the arrival time (ms) of
# the packet AFTER the gap - the only timestamp we have for it.


def compute_loss(df):
    """
    Overall packet loss for one recording.

    Expected count comes from the seq span, not the row count: seq is
    monotonic within a recording, so (last - first + 1) is how many packets
    the ESP32 sent while we were listening.

    Parameters
    ----------
    df : pandas.DataFrame
        Log with a 'seq' column.

    Returns
    -------
    dict with keys:
        expected  : int, packets sent during the recording window
        received  : int, packets actually logged
        loss_pct  : float, percentage lost
    """
    expected = int(df['seq'].iloc[-1]) - int(df['seq'].iloc[0]) + 1
    received = len(df)
    loss_pct = (expected - received) / expected * 100.0
    return {'expected': expected, 'received': received, 'loss_pct': loss_pct}


def find_gaps(df):
    """
    Locate every sequence gap in a recording.

    A diff of 1 between consecutive seq values means no loss; any diff > 1
    is a gap of (diff - 1) consecutive lost packets. Gap size is what
    separates scattered single drops (RF noise, buffer hiccups) from burst
    events (signal blockage, antenna null).

    Parameters
    ----------
    df : pandas.DataFrame
        Log with 'seq' and 'ms' columns.

    Returns
    -------
    pandas.DataFrame with one row per gap:
        gap_start_seq : first missing sequence number
        gap_size      : number of consecutive packets lost
        millis_at_gap : ms timestamp of the packet received AFTER the gap
                        (lost packets have no timestamp of their own)
    """
    seq_diff = df['seq'].diff()
    gap_rows = df.loc[seq_diff > 1]                    # packet right after each gap
    gap_size = seq_diff.loc[gap_rows.index] - 1

    return pd.DataFrame({
        'gap_start_seq': (gap_rows['seq'] - gap_size).astype(int).values,
        'gap_size':      gap_size.astype(int).values,
        'millis_at_gap': gap_rows['ms'].values,
    })


def compute_loss_timeseries(df, bin_seconds=1.0):
    """
    Packet loss percentage binned over time.

    For each consecutive received pair, the interval carries
    (seq_diff) expected packets and (seq_diff - 1) lost packets, both
    attributed to the time bin of the LATER packet - lost packets have no
    timestamp, so the arrival after the gap is the best time anchor.
    Note: a burst longer than one bin is still attributed entirely to the
    bin where reception resumed, so long blockages show as one tall spike.

    Bins where nothing arrived at all (link fully down) have no information
    and get loss_pct = NaN, which breaks the plotted line instead of drawing
    a false zero.

    Parameters
    ----------
    df : pandas.DataFrame
        Log with 'seq' and 't_s' columns (t_s from load_log).
    bin_seconds : float
        Bin width in seconds.

    Returns
    -------
    pandas.DataFrame with one row per bin:
        bin_time_s : bin start time, seconds from start of recording
        loss_pct   : lost / expected * 100 within the bin (NaN if no data)
        lost_count : packets lost within the bin
    """
    # Per-interval loss, attributed to the later packet of each pair.
    # First row has no prior packet, so it contributes nothing.
    seq_diff = df['seq'].diff()
    lost     = (seq_diff - 1).fillna(0)
    expected = seq_diff.fillna(0)

    # Assign each interval to the time bin of its later packet
    bin_idx = np.floor(df['t_s'] / bin_seconds).astype(int)

    per_bin = pd.DataFrame({'bin': bin_idx, 'lost': lost, 'expected': expected})
    agg = per_bin.groupby('bin').sum()

    # Reindex to cover every bin in the recording, so silent stretches
    # (link fully down, nothing received) appear as NaN rather than vanish
    all_bins = np.arange(0, bin_idx.max() + 1)
    agg = agg.reindex(all_bins, fill_value=0)

    with np.errstate(invalid='ignore', divide='ignore'):
        loss_pct = np.where(agg['expected'] > 0,
                            agg['lost'] / agg['expected'] * 100.0,
                            np.nan)

    return pd.DataFrame({
        'bin_time_s': agg.index.values * bin_seconds,
        'loss_pct':   loss_pct,
        'lost_count': agg['lost'].astype(int).values,
    })


# ---------------------------------------------------------------------------
# Field throw pipeline: phase detection, release metrics, audits
# ---------------------------------------------------------------------------
# Protocol per throw: start log, hold roughly still ~1 s (any orientation),
# throw from standing, javelin flies and lands, stop log. One throw per log.
# Post-processing state machine: QUIET -> ACTIVE -> FLIGHT -> IMPACT -> DONE.


@dataclass
class ThrowPhases:
    """Result of detect_phases: indices are into the log arrays."""
    quiet_windows: list          # [(i_start, i_end)] inclusive, time-ordered
    onset_idx: int               # first |accel| > onset threshold, or None
    v0_idx: int                  # velocity zero point (end of the stillest
                                 # pre-throw stretch), or 0 on fallback
    v0_window: tuple             # (i0, i1) of that stillest stretch, or None
    v0_fallback: bool            # True when no onset found (bench log)
    flight_start: int            # FLIGHT entry index, or None
    flight_confirm: int          # index where FLIGHT reached min duration
    flight_end: int              # last FLIGHT sample: impact when found,
                                 # else the end of the low-variance run (so
                                 # post-flight handling in a no-impact log
                                 # cannot pollute FLIGHT statistics)
    impact_idx: int              # first impact sample, or None
    needs_review: bool
    review_reasons: list
    stuck_state: str             # deepest state reached: QUIET/ACTIVE/FLIGHT/DONE
    accel_mag: np.ndarray        # | bias-corrected accel |, m/s^2
    rolling_var: np.ndarray      # rolling variance of accel_mag, (m/s^2)^2
    phase_labels: np.ndarray     # per-sample label: QUIET/ACTIVE/FLIGHT/IMPACT


def _runs_of(mask):
    """Start/end index pairs (inclusive) of consecutive True runs in a bool array."""
    idx = np.flatnonzero(np.diff(np.concatenate(([False], mask, [False])).astype(int)))
    return list(zip(idx[0::2], idx[1::2] - 1))


def rolling_accel_variance(accel_mag, t, window_s):
    """Trailing rolling variance of |accel| over a time window.

    Window length is converted to samples with the median dt; leading samples
    (before one full window) are NaN. NaN compares False against any
    threshold, so detection logic needs no special casing.
    """
    dt_med = float(np.median(np.diff(t)))
    nsamp = max(3, int(round(window_s / dt_med)))
    return pd.Series(accel_mag).rolling(nsamp, min_periods=nsamp).var().values


def detect_phases(df, params):
    """Throw phase state machine over a complete log (post-processing).

    Requires bias-corrected columns (run apply_bias first). Uses time-based
    durations throughout, so UDP packet drops shorten a window's sample count
    but not its measured duration.

    QUIET  : | |accel|-local_g | <= quiet_accel_tol and all |gyro| <
             quiet_gyro_tol, sustained >= quiet_min_dur.
    ACTIVE : end of last quiet window to FLIGHT entry. Nothing armed here;
             throw spikes are ignored.
    FLIGHT : peak-based, anchored on the impact. The impact is the loudest
             event in a short field throw, so: impact anchor = global max
             of |accel|; onset = first >3g within throw_lookback_s before
             it (earlier bumps/drops in the 60 s ring are ignored because
             they are not the loudest event); flight starts at the PULL
             PEAK (max |accel| within release_window_s of onset); flight
             ends at the impact peak's rising edge.
    IMPACT : the rising edge of the global |accel| peak (walk back from the
             max while |accel| stays above impact_rise_frac of it).
             Everything after impact is ignored.

    Sized for easy 10-30 m throws. A log with no sample above the onset
    threshold is a bench log: no onset, no flight, no impact.
    """
    for col in ('ax_corr', 'ay_corr', 'az_corr', 'gx_corr', 'gy_corr', 'gz_corr'):
        if col not in df.columns:
            raise ValueError("detect_phases needs bias-corrected columns; "
                             "call apply_bias(df, ...) first.")

    t = df['t_s'].values
    accel = df[['ax_corr', 'ay_corr', 'az_corr']].values
    gyro_dps = np.degrees(df[['gx_corr', 'gy_corr', 'gz_corr']].values)

    accel_mag = np.linalg.norm(accel, axis=1)
    rolling_var = rolling_accel_variance(accel_mag, t, params.flight_var_window)

    review_reasons = []

    # --- QUIET windows (time-based duration) ---
    quiet_mask = ((np.abs(accel_mag - params.local_g) <= params.quiet_accel_tol) &
                  (np.abs(gyro_dps) < params.quiet_gyro_tol).all(axis=1))
    quiet_windows = [(i0, i1) for i0, i1 in _runs_of(quiet_mask)
                     if t[i1] - t[i0] >= params.quiet_min_dur]

    # --- Peak-based throw detection, anchored on the impact ---
    # The impact is the loudest event in the log for these short throws.
    # Anchoring everything on the global |accel| max makes earlier >3g
    # events in the 60 s ring (bumps, drops, the previous throw's echo)
    # irrelevant without any extra logic: they are not the loudest event.
    onset_thresh = params.onset_thresh_g * params.local_g
    above = np.flatnonzero(accel_mag > onset_thresh)
    onset_idx = None
    flight_start = None
    flight_confirm = None
    flight_run_end = None

    if len(above):
        peak_idx = int(np.argmax(accel_mag))

        # Onset: first >3g sample in the lookback window before the peak
        cand = above[(t[above] >= t[peak_idx] - params.throw_lookback_s) &
                     (above <= peak_idx)]
        onset_idx = int(cand[0]) if len(cand) else int(above[0])

        # Flight starts at the pull peak: the first obvious |accel| peak
        # of the throw (throw_013: 56.127 s)
        j1 = int(np.searchsorted(t, t[onset_idx] + params.release_window_s))
        flight_start = onset_idx + int(np.argmax(accel_mag[onset_idx:j1]))

        # Release-search bound: peak integrated speed sits at the end of
        # the pull, within a fraction of a second of the pull peak
        flight_confirm = int(np.searchsorted(t, t[flight_start] + 0.2))

        # Flight ends at the impact peak's rising edge: walk back from the
        # max while |accel| stays above impact_rise_frac of it
        # (throw_013: peak 478 at 57.64, rising edge 57.62)
        rise = max(onset_thresh, params.impact_rise_frac * accel_mag[peak_idx])
        k = peak_idx
        while k > flight_start + 1 and accel_mag[k - 1] >= rise:
            k -= 1
        flight_run_end = k

    # --- v = 0 point: stillest stretch right before the throw ---
    # Deliberately NOT the strict quiet detector: handheld tremor before a
    # throw (measured ~0.8 m/s^2, ~23 dps on throw_013) fails the absolute
    # tolerances, but the stillest second of the pre-throw hold is exactly
    # where velocity is closest to zero. Anchoring here means only data
    # from right before the throw feeds the velocity integral.
    v0_fallback = False
    v0_window = None
    if onset_idx is not None:
        j_lo = int(np.searchsorted(t, t[onset_idx] - params.v0_lookback_s))
        v0_window = find_stillest_window(rolling_var, t, j_lo, onset_idx,
                                         params.v0_still_dur)
        if v0_window is None:
            raise ValueError(
                f"No {params.v0_still_dur:.1f} s stretch found in the "
                f"{params.v0_lookback_s:.1f} s before onset "
                f"(t={t[onset_idx]:.2f} s) - the log starts too close to "
                f"the throw. Hold still for a second right before throwing, "
                f"or raise SpearParams.v0_lookback_s.")
        v0_idx = v0_window[1]
    else:
        # Bench log: no throw. Integrate from sample 0 and say so.
        v0_idx = 0
        v0_fallback = True
        print("NOTE: no throw onset found (|accel| never exceeded "
              f"{params.onset_thresh_g:.1f} g). Falling back to integrating "
              "from sample 0. Expected for bench logs.")

    # --- IMPACT: the rising edge found by the walk-back above ---
    # A degenerate span (impact right at the pull peak) means the global
    # max WAS the pull: no flight happened (fake throw, held-on swing).
    # The duration sanity check below flags it for review.
    impact_idx = None
    flight_end = None
    if flight_start is not None:
        impact_idx = flight_run_end
        flight_end = impact_idx

    # --- Flight duration sanity check (detector span) ---
    if flight_start is not None:
        flight_dur = t[flight_end] - t[flight_start]
        if not (params.flight_sane_min <= flight_dur <= params.flight_sane_max):
            print("=" * 60)
            print(f"WARNING: flight duration {flight_dur:.2f} s is outside the "
                  f"sane range [{params.flight_sane_min}, "
                  f"{params.flight_sane_max}] s. Results marked NEEDS REVIEW.")
            print("=" * 60)
            review_reasons.append(
                f"flight duration {flight_dur:.2f} s outside "
                f"[{params.flight_sane_min}, {params.flight_sane_max}] s")

    # --- Per-sample phase labels ---
    # Default ACTIVE covers handling before the first quiet window too;
    # phase statistics only distinguish QUIET / ACTIVE / FLIGHT / IMPACT.
    phase_labels = np.full(len(t), 'ACTIVE', dtype='<U6')
    for i0, i1 in quiet_windows:
        phase_labels[i0:i1 + 1] = 'QUIET'
    if flight_start is not None:
        end = impact_idx if impact_idx is not None else flight_end + 1
        phase_labels[flight_start:end] = 'FLIGHT'
    if impact_idx is not None:
        phase_labels[impact_idx:] = 'IMPACT'

    if impact_idx is not None:
        stuck_state = 'DONE'
    elif flight_start is not None:
        stuck_state = 'FLIGHT'
    elif onset_idx is not None:
        stuck_state = 'ACTIVE'
    elif quiet_windows:
        stuck_state = 'QUIET'
    else:
        stuck_state = 'NONE'

    return ThrowPhases(
        quiet_windows=quiet_windows, onset_idx=onset_idx,
        v0_idx=v0_idx, v0_window=v0_window, v0_fallback=v0_fallback,
        flight_start=flight_start, flight_confirm=flight_confirm,
        flight_end=flight_end, impact_idx=impact_idx,
        needs_review=bool(review_reasons), review_reasons=review_reasons,
        stuck_state=stuck_state, accel_mag=accel_mag,
        rolling_var=rolling_var, phase_labels=phase_labels)


def integrate_throw_velocity(t, lin_accel_world, phases):
    """Integrate world-frame linear accel to velocity over the throw window.

    v = 0 at phases.v0_idx (end of the last quiet window before onset);
    integration stops at impact. Samples outside [v0, impact] are NaN so
    plots cannot silently show pre-hold or post-impact velocity as real.

    Returns (vel Nx3, speed N) arrays.
    """
    N = len(t)
    i0 = phases.v0_idx
    i1 = phases.impact_idx if phases.impact_idx is not None else N

    vel = np.full((N, 3), np.nan)
    for ax in range(3):
        vel[i0:i1, ax] = integrate.cumulative_trapezoid(
            lin_accel_world[i0:i1, ax], t[i0:i1], initial=0.0)
    speed = np.linalg.norm(vel, axis=1)   # NaN propagates outside the window
    return vel, speed


def find_release(t, vel, speed, phases):
    """Release sample and release-state numbers.

    Release = the PEAK SPEED sample of the integrated velocity. Speed rises
    through the pull and decays after the hand lets go, so the peak marks
    release to within a sample or two. FLIGHT entry is deliberately not used
    AS the release sample: the variance detector confirms flight late by
    design.

    The peak search is bounded at FLIGHT confirmation (when FLIGHT was
    found). Over the full flight, speed can EXCEED the release value: the
    drop from release height adds v_land^2 = v_rel^2 + 2*g*h of energy, and
    on gentle flat throws that beats the drag loss; velocity integration
    drift inflates late-flight speed further. Physical release always
    precedes confirmed flight, so bounding there keeps the true peak in the
    window while late-flight growth can never win the argmax.

    Returns dict(idx, t_s, speed, elev_deg) or None when no onset was found.
    """
    if phases.onset_idx is None:
        return None
    i0 = phases.v0_idx
    if phases.flight_confirm is not None:
        i1 = phases.flight_confirm + 1
    elif phases.impact_idx is not None:
        i1 = phases.impact_idx
    else:
        i1 = len(t)
    window = speed[i0:i1]
    if not np.any(np.isfinite(window)):
        return None
    rel = i0 + int(np.nanargmax(window))
    vx, vy, vz = vel[rel]
    elev_deg = math.degrees(math.atan2(vz, math.hypot(vx, vy)))
    return {'idx': rel, 't_s': float(t[rel]),
            'speed': float(speed[rel]), 'elev_deg': elev_deg}


def ballistic_predict(speed, elev_deg, height_m, local_g):
    """Vacuum ballistic flight time and range from release state.

    Solves h + v*sin(th)*t - 0.5*g*t^2 = 0 for the positive root. IGNORES
    aerodynamics entirely (a javelin glides); this is a self-consistency
    sanity check against the measured flight time and taped distance, not a
    performance model.

    Returns dict(t_flight_s, range_m).
    """
    th = math.radians(elev_deg)
    vz0 = speed * math.sin(th)
    vh = speed * math.cos(th)
    disc = vz0 * vz0 + 2.0 * local_g * height_m
    t_flight = (vz0 + math.sqrt(disc)) / local_g
    return {'t_flight_s': t_flight, 'range_m': vh * t_flight}


def clipping_audit(df, phases, params):
    """Count near-full-scale samples per phase, accel and gyro separately.

    Runs on RAW columns: clipping happens at the sensor, before bias
    correction. IMPACT-phase accel clipping is expected (the javelin hits
    the ground) and is informational only. Accel clipping inside the release
    window (v0 to FLIGHT entry) degrades release speed/angle to a lower
    bound; gyro saturation before or at release makes attitude suspect.

    Returns dict with per-phase counts and the two degradation flags.
    """
    accel_rail = params.clip_rail_frac * params.accel_fs_g * G_STD
    gyro_rail = math.radians(params.clip_rail_frac * params.gyro_fs_dps)

    accel_clip = (np.abs(df[['ax', 'ay', 'az']].values) >= accel_rail).any(axis=1)
    gyro_clip = (np.abs(df[['gx', 'gy', 'gz']].values) >= gyro_rail).any(axis=1)

    phases_seen = ['QUIET', 'ACTIVE', 'FLIGHT', 'IMPACT']
    per_phase = {}
    for ph in phases_seen:
        mask = phases.phase_labels == ph
        if mask.any():
            per_phase[ph] = {'n': int(mask.sum()),
                             'accel_clip': int(accel_clip[mask].sum()),
                             'gyro_clip': int(gyro_clip[mask].sum())}

    # Release window: v0 through the release-search bound (flight_confirm,
    # just past the pull peak). Must INCLUDE the pull peak: that is where
    # accel is largest and clipping actually happens, and clipping there
    # corrupts the integrated release speed.
    i0 = phases.v0_idx
    if phases.flight_confirm is not None:
        i1 = phases.flight_confirm
    elif phases.impact_idx is not None:
        i1 = phases.impact_idx
    else:
        i1 = len(accel_clip)
    release_accel_clip = int(accel_clip[i0:i1].sum())

    # Gyro saturation anywhere up to the end of the release window
    gyro_sat_before_release = int(gyro_clip[:i1].sum())

    return {'per_phase': per_phase,
            'release_accel_clip': release_accel_clip,
            'gyro_sat_before_release': gyro_sat_before_release,
            'total_accel_clip': int(accel_clip.sum()),
            'total_gyro_clip': int(gyro_clip.sum())}


def attitude_truth_angle(accel, attitude, gate_active):
    """Per-sample angle (deg) between predicted and measured gravity direction.

    Predicted body-frame up = world [0,0,1] rotated into body frame by the
    current quaternion; measured = normalized accel. Only meaningful when the
    accel gate is closed (|accel| near 1 g), NaN otherwise. This is the
    attitude truth metric: it bypasses the ZYX Euler singularity at pitch 90
    entirely because it never touches Euler angles.
    """
    N = len(accel)
    ang = np.full(N, np.nan)
    up_world = np.array([0.0, 0.0, 1.0])
    for i in range(N):
        if not gate_active[i]:
            continue
        a_norm = np.linalg.norm(accel[i])
        if a_norm < 1e-9:
            continue
        R = sf.quat_to_rotation_matrix(attitude[i])
        g_body_pred = R.T @ up_world
        cosang = np.clip(np.dot(accel[i] / a_norm, g_body_pred), -1.0, 1.0)
        ang[i] = math.degrees(math.acos(cosang))
    return ang


def yaw_excursion_and_drift(t, attitude, quiet_windows):
    """Separate yaw excursion from yaw drift.

    Excursion: total unwrapped yaw range over the whole log. This includes
    real rotation (throw spin, handling) and must NOT be called drift.
    Drift: linear yaw slope fitted within quiet windows only, where any yaw
    change IS error (nothing is rotating), duration-weighted across windows.

    Returns dict(excursion_deg, drift_deg_per_min or None, n_quiet_windows).
    """
    yaw = np.array([sf.quat_to_euler(q)[2] for q in attitude])
    yaw_unwrapped = np.degrees(np.unwrap(np.radians(yaw)))
    excursion = float(yaw_unwrapped.max() - yaw_unwrapped.min())

    slopes, weights = [], []
    for i0, i1 in quiet_windows:
        dur = t[i1] - t[i0]
        if dur > 0.2 and i1 - i0 >= 5:   # need enough span for a stable fit
            slope = np.polyfit(t[i0:i1 + 1], yaw_unwrapped[i0:i1 + 1], 1)[0]
            slopes.append(slope)
            weights.append(dur)

    drift = None
    if slopes:
        drift = float(np.average(slopes, weights=weights) * 60.0)  # deg/min

    return {'excursion_deg': excursion, 'drift_deg_per_min': drift,
            'n_quiet_windows': len(quiet_windows)}


def phase_variance_stats(phases):
    """Median and 90th percentile of rolling variance per detected phase.

    Informational: characterizes each phase's vibration level. Detection is
    peak-based and does not use these numbers.
    """
    stats = {}
    for ph in ('QUIET', 'ACTIVE', 'FLIGHT', 'IMPACT'):
        mask = phases.phase_labels == ph
        v = phases.rolling_var[mask]
        v = v[np.isfinite(v)]
        if len(v):
            stats[ph] = {'median': float(np.median(v)),
                         'p90': float(np.percentile(v, 90)),
                         'n': int(len(v))}
    return stats


def find_stillest_window(rolling_var, t, i0, i1, dur_s=1.0):
    """The stillest stretch of duration dur_s inside [i0, i1).

    Relaxed alternative to the strict quiet detector: instead of absolute
    tolerances (which handheld tremor can fail), take the dur_s window with
    the lowest mean rolling variance. Useful for eyeballing the pre-throw
    hold when the strict detector found nothing there.

    Returns (j0, j1) inclusive index pair, or None if the span is shorter
    than dur_s.
    """
    dt_med = float(np.median(np.diff(t)))
    n = max(2, int(round(dur_s / dt_med)))
    if i1 - i0 < n:
        return None
    # NaN (the rolling-variance warmup at the start of a file) becomes a
    # LARGE FINITE penalty, not inf: inf poisons the cumulative sum (every
    # later window mean turns into inf - inf = NaN and argmin returns the
    # first NaN, silently picking the warmup region as "stillest").
    v = np.nan_to_num(rolling_var[i0:i1], nan=1e12, posinf=1e12)
    # Mean variance over every candidate window via cumulative sum
    c = np.concatenate(([0.0], np.cumsum(v)))
    means = (c[n:] - c[:-n]) / n
    j0 = i0 + int(np.argmin(means))
    return (j0, j0 + n - 1)


@dataclass
class FieldAnalysis:
    """Everything analyze_field_log computes, bundled for summary and plots."""
    log_name: str
    params: SpearParams
    t: np.ndarray
    phases: ThrowPhases
    attitude: np.ndarray         # Nx4 quaternions
    gate_active: np.ndarray      # N bool
    lin_accel_world: np.ndarray  # Nx3 m/s^2, gravity removed
    vel: np.ndarray              # Nx3 m/s, NaN outside integration window
    speed: np.ndarray            # N m/s
    release: dict                # from find_release, or None
    ballistic: dict              # from ballistic_predict, or None
    audit: dict                  # from clipping_audit
    truth_angle: np.ndarray      # N deg, NaN when gate open
    yaw_stats: dict              # from yaw_excursion_and_drift
    var_stats: dict              # from phase_variance_stats
    q0_source: str               # where the initial attitude came from


def analyze_field_log(df, params, log_name=''):
    """Full field-throw pipeline over one bias-corrected log.

    Orchestrates: phase detection, quiet-window attitude initialization,
    complementary filter with per-sample dt (UDP drop handling), gravity
    removal, windowed velocity integration, release detection, ballistic
    prediction, clipping audit, and truth metrics. Main scripts call this
    once and then print/plot; no analysis logic lives in the scripts.
    """
    t = df['t_s'].values
    accel = df[['ax_corr', 'ay_corr', 'az_corr']].values
    gyro_dps = np.degrees(df[['gx_corr', 'gy_corr', 'gz_corr']].values)

    phases = detect_phases(df, params)

    # Initial attitude from the FIRST quiet window's mean accel direction.
    # Works at ANY tilt (cocked behind the shoulder included): the quaternion
    # maps measured gravity to world down via the shortest arc. Yaw is
    # unobservable without a magnetometer; it initializes to zero, so all
    # reported headings are relative to the starting orientation.
    if phases.quiet_windows:
        i0, i1 = phases.quiet_windows[0]
        q0 = sf.quat_from_accel(accel[i0:i1 + 1].mean(axis=0))
        q0_source = f"quiet window {t[i0]:.2f}-{t[i1]:.2f} s"
    else:
        q0 = None   # filter falls back to first-100-sample init and warns
        q0_source = "fallback: first 100 samples (no quiet window found)"

    # Per-sample dt: UDP drops make some gaps 2-3 nominal periods and the
    # gyro must integrate across the real elapsed time
    dt_med = float(np.median(np.diff(t)))
    dt_array = np.diff(t, prepend=t[0])
    dt_array[0] = dt_med
    dt_array = np.clip(dt_array, 1e-4, None)   # guard duplicate timestamps

    attitude, lin_accel_world, _, gate_active = sf.complementary_filter(
        accel, gyro_dps, dt_med, local_g=params.local_g,
        q0=q0, dt_array=dt_array)

    vel, speed = integrate_throw_velocity(t, lin_accel_world, phases)
    release = find_release(t, vel, speed, phases)

    ballistic = None
    if release is not None:
        ballistic = ballistic_predict(release['speed'], release['elev_deg'],
                                      params.release_height_m, params.local_g)

    audit = clipping_audit(df, phases, params)
    truth_angle = attitude_truth_angle(accel, attitude, gate_active)
    yaw_stats = yaw_excursion_and_drift(t, attitude, phases.quiet_windows)
    var_stats = phase_variance_stats(phases)

    return FieldAnalysis(
        log_name=log_name, params=params, t=t, phases=phases,
        attitude=attitude, gate_active=gate_active,
        lin_accel_world=lin_accel_world, vel=vel, speed=speed,
        release=release, ballistic=ballistic, audit=audit,
        truth_angle=truth_angle, yaw_stats=yaw_stats, var_stats=var_stats,
        q0_source=q0_source)


def print_field_summary(res, measured_distance_m=None):
    """Single summary block per log: phases, release, prediction, audits.

    measured_distance_m: optional tape-measured throw distance; when given,
    the ballistic section prints the error of the vacuum prediction against
    it, in meters and percent.
    """
    p = res.params
    ph = res.phases
    t = res.t
    w = 66

    print()
    print('=' * w)
    print(f"Field Throw Summary: {res.log_name}")
    if ph.needs_review:
        print(">>> NEEDS REVIEW <<<")
        for r in ph.review_reasons:
            print(f"    reason: {r}")
    print('=' * w)

    # --- Phase timeline ---
    print("Phase timeline:")
    for i, (i0, i1) in enumerate(ph.quiet_windows):
        print(f"  QUIET  {i+1}:      {t[i0]:7.2f} - {t[i1]:.2f} s "
              f"({t[i1]-t[i0]:.2f} s)")
    if ph.onset_idx is not None:
        print(f"  onset:          {t[ph.onset_idx]:7.2f} s  "
              f"(|accel| > {p.onset_thresh_g:.1f} g)")
    else:
        print("  onset:          not found (bench log?)")
    if ph.flight_start is not None:
        fs = t[ph.flight_start]
        fe = t[ph.flight_end]
        # Pull peak to impact rising edge. The ballistic section reports
        # flight time from RELEASE (peak integrated speed), which sits at
        # the end of the pull, within a few samples of the pull peak.
        print(f"  FLIGHT:         {fs:7.2f} - {fe:.2f} s  "
              f"(pull peak to impact rise, {fe-fs:.2f} s)")
    else:
        print("  FLIGHT:         not detected")
    if ph.impact_idx is not None:
        print(f"  IMPACT:         {t[ph.impact_idx]:7.2f} s  "
              "(everything after ignored)")
    else:
        print("  IMPACT:         not detected")
    print(f"  machine state:  {ph.stuck_state}")

    # --- Integration window ---
    print()
    print("Integration window:")
    if ph.v0_fallback:
        print("  v=0 point:      sample 0 (FALLBACK: no onset found)")
    else:
        vw = ph.v0_window
        print(f"  v=0 point:      t={t[ph.v0_idx]:.2f} s (end of stillest "
              f"pre-throw stretch, {t[vw[0]]:.2f}-{t[vw[1]]:.2f} s)")
        print("                  velocity integrates ONLY from here to "
              "impact; nothing earlier is used")
    if ph.onset_idx is not None:
        gap = t[ph.onset_idx] - t[ph.v0_idx]
        print(f"  onset:          index {ph.onset_idx}, "
              f"t={t[ph.onset_idx]:.2f} s  (gap {gap:.2f} s)")
        if gap > 3.0:
            print(f"  NOTE: {gap:.1f} s between v=0 and onset. Velocity "
                  "integrates over that whole gap, so bias and attitude "
                  "error leak into the release numbers. Shorten the hold.")

    # --- Release ---
    print()
    degraded = res.audit['release_accel_clip'] > 0
    if res.release is not None:
        r = res.release
        if degraded:
            n = res.audit['release_accel_clip']
            print(f"Release (t={r['t_s']:.2f} s):")
            print(f"  speed:          >= {r['speed']:.1f} m/s  "
                  f"(LOWER BOUND: {n} accel samples at rail during release)")
            print(f"  elevation:      {r['elev_deg']:+.1f} deg  "
                  "(degraded: rail-clipped accel in release window)")
        else:
            print(f"Release (t={r['t_s']:.2f} s):")
            print(f"  speed:          {r['speed']:.2f} m/s")
            print(f"  elevation:      {r['elev_deg']:+.1f} deg "
                  "(velocity angle above horizontal)")
    else:
        print("Release: not computed (no throw detected)")

    # --- Ballistic self-consistency (vacuum model, ignores aero) ---
    if res.ballistic is not None:
        b = res.ballistic
        print()
        print(f"Ballistic prediction (VACUUM model, no aero, "
              f"release height {p.release_height_m:.1f} m):")
        if degraded:
            print("  CAVEAT: inputs are lower bounds (release-window "
                  "clipping); prediction inherits that.")
        print(f"  predicted flight time:  {b['t_flight_s']:.2f} s")
        # Measured from RELEASE to impact: the javelin is flying from the
        # moment speed peaks. FLIGHT entry is only when the detector
        # confirms it, ~0.3 s late by construction.
        if res.release is not None and ph.impact_idx is not None:
            meas = t[ph.impact_idx] - res.release['t_s']
            print(f"  measured  flight time:  {meas:.2f} s  "
                  f"(release to impact, difference "
                  f"{b['t_flight_s']-meas:+.2f} s)")
        else:
            print("  measured  flight time:  unavailable "
                  "(release/impact not both found)")
        if measured_distance_m is not None:
            err_m = b['range_m'] - measured_distance_m
            err_pct = err_m / measured_distance_m * 100.0
            print(f"  predicted range:        {b['range_m']:.1f} m")
            print(f"  measured  range:        {measured_distance_m:.1f} m "
                  "(tape)")
            print(f"  range error:            {err_m:+.1f} m ({err_pct:+.0f}%)"
                  "  (vacuum model; javelin aero makes some error expected)")
        else:
            print(f"  predicted range:        {b['range_m']:.1f} m  "
                  "(compare against tape measure, or re-run with "
                  "--distance)")

    # --- Data quality: only the things that change how much to trust the
    # numbers above. Clipping in the release window makes speed a lower
    # bound (already formatted above); gyro saturation before release makes
    # the elevation angle suspect (also tracks spin headroom, an open
    # question); the attitude check is filter health, which feeds
    # elevation and therefore the range prediction.
    print()
    print("Data quality:")
    rel_clip = res.audit['release_accel_clip']
    imp_clip = res.audit['per_phase'].get('IMPACT', {}).get('accel_clip', 0)
    gyro_sat = res.audit['gyro_sat_before_release']
    if rel_clip == 0 and gyro_sat == 0:
        note = f"  (impact railed {imp_clip} samples, expected)" \
            if imp_clip else ""
        print(f"  clipping:       none before impact{note}")
    else:
        if rel_clip:
            print(f"  clipping:       {rel_clip} accel samples at rail in "
                  "the release window (speed above is a LOWER BOUND)")
        if gyro_sat:
            print(f"  gyro saturation: {gyro_sat} samples before release "
                  "(elevation angle suspect; spin near 2000 dps limit)")

    end = ph.impact_idx if ph.impact_idx is not None else len(res.truth_angle)
    finite = res.truth_angle[:end]
    finite = finite[np.isfinite(finite)]
    if len(finite):
        print(f"  attitude check (whole record, convergence-sensitive): "
              f"{np.mean(finite):.1f} deg mean error, predicted vs "
              f"measured gravity")

    # Attitude check over ONLY the v=0 still window. The whole-record mean
    # above is convergence-contaminated: it averages the filter's early
    # settling from its initial guess and any pre-throw handling together
    # with the part that matters, so it moves when the trim window moves
    # (measured: TRIM_PRE 5 -> 10 shifted it 6.9 -> 4.7 on one throw and
    # 5.5 -> 6.5 on another while every release number stayed
    # bit-identical). The v=0 window is the stillest stretch right before
    # the throw - the attitude the filter carries INTO the pull - so its
    # error is the one that actually feeds release elevation.
    if ph.v0_window is not None:
        j0, j1 = ph.v0_window
        vwin = res.truth_angle[j0:j1 + 1]
        vwin = vwin[np.isfinite(vwin)]      # in-gate samples only (NaN = gate open)
        if len(vwin) >= p.attitude_check_min_samples:
            print(f"  attitude check (v=0 window):   {np.mean(vwin):.1f} deg "
                  f"mean error over {len(vwin)} in-gate samples "
                  f"(healthy < ~5; feeds release elevation)")
        else:
            print(f"  attitude check (v=0 window):   n/a "
                  f"({len(vwin)} in-gate samples, need "
                  f">= {p.attitude_check_min_samples})")

    print('=' * w)


def write_trimmed_copy(raw_df, phases, params, src_path):
    """Write <log>_trimmed.csv next to the source: only the good part.

    Window: trim_pre_onset_s before onset through trim_post_impact_s after
    impact (both SpearParams fields; edit trim_pre_onset_s to move the
    start). RAW columns only, same format as the source, so every loader
    and analysis script consumes the trimmed copy unchanged.

    Returns the path written, or None when there is no throw to trim to.
    """
    if phases.onset_idx is None:
        return None
    t = raw_df['t_s'].values
    i0 = int(np.searchsorted(t, t[phases.onset_idx] - params.trim_pre_onset_s))
    if phases.impact_idx is not None:
        i1 = int(np.searchsorted(t, t[phases.impact_idx]
                                 + params.trim_post_impact_s))
    else:
        i1 = len(t)

    cols = [c for c in ('seq', 'ms', 'temp_C', 'ax', 'ay', 'az',
                        'gx', 'gy', 'gz') if c in raw_df.columns]
    out_path = os.path.splitext(src_path)[0] + '_trimmed.csv'
    raw_df.iloc[i0:i1][cols].to_csv(out_path, index=False)
    return out_path


def overlay_phase_markers(ax, res):
    """Draw the detected phase markers on any time-axis plot.

    The single source of truth for phase overlays: onset (orange), the v=0
    still window (green span) and anchor (green dotted), FLIGHT entry
    (blue dashed), IMPACT (red) or FLIGHT end when no impact. Used by the
    field diagnostic and by presentation figures, so the two can never
    drift apart. Strict QUIET windows are deliberately NOT shaded: the
    only green area is the stretch the velocity integral actually uses.
    """
    ph = res.phases
    t = res.t
    if ph.onset_idx is not None:
        ax.axvline(t[ph.onset_idx], color='orange', linewidth=1.2,
                   label='onset')
    if not ph.v0_fallback:
        if ph.v0_window is not None:
            ax.axvspan(t[ph.v0_window[0]], t[ph.v0_window[1]],
                       alpha=0.25, color='limegreen',
                       label='v=0 window (integration starts at its end)')
        ax.axvline(t[ph.v0_idx], color='green', linewidth=1.2,
                   linestyle=':', label='v=0')
    if ph.flight_start is not None:
        ax.axvline(t[ph.flight_start], color='blue', linewidth=1.2,
                   linestyle='--', label='FLIGHT entry')
    if ph.impact_idx is not None:
        ax.axvline(t[ph.impact_idx], color='red', linewidth=1.2,
                   label='IMPACT')
    elif ph.flight_end is not None:
        # No impact: show where the low-variance run (and therefore the
        # FLIGHT span) actually ends, so the phase extent is readable
        ax.axvline(t[ph.flight_end], color='blue', linewidth=1.0,
                   linestyle=':', label='FLIGHT end (no impact)')


def plot_field_diagnostic(res, save_path, show=True):
    """The field tuning instrument: |accel| and its rolling variance vs time.

    Two stacked subplots, shared time axis. The variance subplot uses a LOG
    y axis: quiet (~1e-3), flight (~1e-1?), and throw (~1e2+) variances span
    orders of magnitude and a linear axis makes the flight plateau invisible.
    Phase boundaries overlay both subplots. Renders correctly when the state
    machine found nothing (that is when this plot matters most): whatever
    boundaries exist are drawn and the stuck state is printed on the figure.
    """
    ph = res.phases
    t = res.t
    p = res.params

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle(f"Throw diagnostic: {res.log_name}   "
                 f"(machine state: {ph.stuck_state})", fontsize=12)

    ax1.plot(t, ph.accel_mag, color='steelblue', linewidth=0.8)
    ax1.set_ylabel('|accel| (m/s^2)')
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(t, ph.rolling_var, color='steelblue', linewidth=0.8)
    ax2.set_ylabel(f'rolling var of |accel| '
                   f'({p.flight_var_window:.1f} s win, log scale)')
    ax2.set_xlabel('Time (s)')
    ax2.grid(True, alpha=0.3, which='both')

    # Overlay phase boundaries on BOTH subplots (shared helper, also used
    # by presentation figures)
    for ax in (ax1, ax2):
        overlay_phase_markers(ax, res)

    # When detection failed, say so ON the figure: this plot exists to debug
    # exactly that case in the field
    if ph.flight_start is None:
        ax2.text(0.02, 0.95,
                 f"No throw found (machine stuck at {ph.stuck_state}): "
                 "no |accel| sample above "
                 f"{p.onset_thresh_g:.1f} g anywhere in the log.",
                 transform=ax2.transAxes, va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightyellow',
                           edgecolor='orange'))

    # Zoom to the throw: a 60 s ring is mostly dead time and the whole
    # event lives in the last few seconds. Context: 5 s before onset to
    # 3 s after impact. The full record is still in the data; use
    # plot_throw_window.py or pan the interactive window to see more.
    if ph.onset_idx is not None:
        x_lo = max(t[0], t[ph.onset_idx] - 5.0)
        x_hi = t[-1] if ph.impact_idx is None else min(t[-1],
                                                       t[ph.impact_idx] + 3.0)
        ax1.set_xlim(x_lo, x_hi)

    ax1.legend(loc='upper right', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    plt.tight_layout()

    fig.savefig(save_path, dpi=150)
    print(f"Diagnostic plot saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)