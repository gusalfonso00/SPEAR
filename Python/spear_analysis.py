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

    # FLIGHT detection: rolling variance of |accel| below flight_var_thresh
    # sustained for flight_min_dur. Variance, not magnitude: free flight can
    # read several g (drag plus centripetal from off-axis mounting) but it is
    # SMOOTH. flight_var_window is the rolling window length.
    flight_var_window: float = 0.3  # s

    # >>> PROVISIONAL AND UNVALIDATED <<<
    # flight_var_thresh has never seen a real throw. The default below is set
    # from bench data only: comfortably above stationary and handheld-quiet
    # variance, far below throw/handling dynamics. It WILL be retuned on field
    # day. Field procedure (two lines):
    #   Throw one javelin, run analyze_field_throw.py --diagnostic on its log.
    #   Copy the SUGGESTED flight_var_thresh from the summary into this field.
    flight_var_thresh: float = 0.5  # (m/s^2)^2, rolling variance of |accel|

    flight_min_dur: float = 0.5     # s below threshold to confirm FLIGHT
    flight_sane_min: float = 1.0    # s; flight shorter than this = needs review
    flight_sane_max: float = 4.0    # s; flight longer than this = needs review

    # IMPACT detection (armed only after FLIGHT is confirmed): any raw accel
    # axis beyond impact_rail_frac of full scale, or a step in |accel| larger
    # than impact_step_mps2 between consecutive samples.
    impact_rail_frac: float = 0.90  # fraction of accel full scale
    impact_step_mps2: float = 50.0  # m/s^2 jump in |accel| sample to sample

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
    v0_idx: int                  # velocity zero point (end of last quiet
                                 # window before onset), or 0 on fallback
    v0_fallback: bool            # True when no onset found (bench log)
    flight_start: int            # FLIGHT entry index, or None
    flight_confirm: int          # index where FLIGHT reached min duration
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
    FLIGHT : rolling variance of |accel| < flight_var_thresh sustained
             >= flight_min_dur, searched only AFTER onset. Variance, not
             magnitude: flight can read several g but it is smooth.
    IMPACT : armed only after FLIGHT confirms. Any raw accel axis beyond
             impact_rail_frac of full scale, or an |accel| step larger than
             impact_step_mps2. Everything after the first impact is ignored.
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

    # --- Onset: first sample above the throw threshold ---
    onset_thresh = params.onset_thresh_g * params.local_g
    above = np.flatnonzero(accel_mag > onset_thresh)
    onset_idx = int(above[0]) if len(above) else None

    # --- v = 0 point: end of the LAST quiet window before onset ---
    v0_fallback = False
    if onset_idx is not None:
        pre = [w for w in quiet_windows if w[1] <= onset_idx]
        if not pre:
            raise ValueError(
                f"No quiet window found before throw onset at index {onset_idx} "
                f"(t={t[onset_idx]:.2f} s). Thresholds used: quiet_accel_tol="
                f"{params.quiet_accel_tol} m/s^2, quiet_gyro_tol="
                f"{params.quiet_gyro_tol} dps, quiet_min_dur="
                f"{params.quiet_min_dur} s. Was the pre-throw hold too short "
                f"or too shaky? Loosen the quiet tolerances or re-log.")
        v0_idx = pre[-1][1]
    else:
        # Bench log: no throw. Integrate from sample 0 and say so.
        v0_idx = 0
        v0_fallback = True
        print("NOTE: no throw onset found (|accel| never exceeded "
              f"{params.onset_thresh_g:.1f} g). Falling back to integrating "
              "from sample 0. Expected for bench logs.")

    # --- FLIGHT: sustained low variance, searched only after onset ---
    flight_start = None
    flight_confirm = None
    if onset_idx is not None:
        low = rolling_var < params.flight_var_thresh   # NaN -> False
        low[:onset_idx + 1] = False
        for i0, i1 in _runs_of(low):
            if t[i1] - t[i0] >= params.flight_min_dur:
                flight_start = i0
                # Confirmation sample: where the run has lasted min_dur
                flight_confirm = i0 + int(np.searchsorted(
                    t[i0:i1 + 1], t[i0] + params.flight_min_dur))
                break

    # --- IMPACT: armed only once FLIGHT is confirmed ---
    impact_idx = None
    if flight_confirm is not None:
        rail = params.impact_rail_frac * params.accel_fs_g * G_STD
        raw = df[['ax', 'ay', 'az']].values
        near_rail = (np.abs(raw) >= rail).any(axis=1)
        step = np.abs(np.diff(accel_mag, prepend=accel_mag[0]))
        hits = np.flatnonzero((near_rail | (step > params.impact_step_mps2))
                              & (np.arange(len(t)) > flight_confirm))
        impact_idx = int(hits[0]) if len(hits) else None
        if impact_idx is None:
            review_reasons.append("FLIGHT found but no impact detected; "
                                  "flight measured to end of log")

    # --- Flight duration sanity check ---
    if flight_start is not None:
        flight_end = impact_idx if impact_idx is not None else len(t) - 1
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
        end = impact_idx if impact_idx is not None else len(t)
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
        v0_idx=v0_idx, v0_fallback=v0_fallback,
        flight_start=flight_start, flight_confirm=flight_confirm,
        impact_idx=impact_idx,
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

    # Release window: v0 to FLIGHT entry (or impact/end when FLIGHT missing)
    i0 = phases.v0_idx
    if phases.flight_start is not None:
        i1 = phases.flight_start
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

    The field tuning aid: read the FLIGHT plateau level here and set
    flight_var_thresh from it (the summary suggests 4x the FLIGHT median).
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


def print_field_summary(res):
    """Single summary block per log: phases, release, prediction, audits."""
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
        fe = t[ph.impact_idx] if ph.impact_idx is not None else t[-1]
        print(f"  FLIGHT:         {fs:7.2f} - {fe:.2f} s  "
              f"(measured {fe-fs:.2f} s)")
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
        print(f"  v=0 point:      index {ph.v0_idx}, t={t[ph.v0_idx]:.2f} s "
              "(end of last quiet window)")
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
        if ph.flight_start is not None and ph.impact_idx is not None:
            meas = t[ph.impact_idx] - t[ph.flight_start]
            print(f"  measured  flight time:  {meas:.2f} s  "
                  f"(difference {b['t_flight_s']-meas:+.2f} s)")
        else:
            print("  measured  flight time:  unavailable "
                  "(flight/impact not both detected)")
        print(f"  predicted range:        {b['range_m']:.1f} m  "
              "(compare against tape measure)")

    # --- Variance statistics and threshold suggestion ---
    print()
    print(f"Rolling variance of |accel| per phase "
          f"({p.flight_var_window:.1f} s window, (m/s^2)^2):")
    for name, s in res.var_stats.items():
        print(f"  {name:<7} median {s['median']:>12.4g}   "
              f"p90 {s['p90']:>12.4g}   ({s['n']} samples)")
    if 'FLIGHT' in res.var_stats:
        fmed = res.var_stats['FLIGHT']['median']
        if 'ACTIVE' in res.var_stats:
            ratio = res.var_stats['ACTIVE']['median'] / fmed
            print(f"  ACTIVE/FLIGHT median ratio: {ratio:.0f}x")
        print(f"  SUGGESTED flight_var_thresh = {4.0*fmed:.4g}  "
              f"(4x FLIGHT median, source: {res.log_name})")
        print("  Copy that value into SpearParams.flight_var_thresh and re-run.")
    elif ph.onset_idx is not None:
        # No flight found: print post-onset percentiles so the plateau can
        # be read off manually
        post = res.phases.rolling_var[ph.onset_idx:]
        post = post[np.isfinite(post)]
        if len(post):
            print("  FLIGHT not found. Post-onset variance percentiles for "
                  "manual threshold setting:")
            for q in (10, 25, 50, 75, 90):
                print(f"    p{q:<3} {np.percentile(post, q):>12.4g}")

    # --- Clipping / saturation audit ---
    print()
    print(f"Clipping audit (rail = {p.clip_rail_frac:.0%} of "
          f"{p.accel_fs_g:.0f} g / {p.gyro_fs_dps:.0f} dps):")
    for name, s in res.audit['per_phase'].items():
        note = ""
        if name == 'IMPACT' and s['accel_clip'] > 0:
            note = "  (expected at ground strike, informational)"
        print(f"  {name:<7} accel {s['accel_clip']:>5} / {s['n']:<6} "
              f"gyro {s['gyro_clip']:>5} / {s['n']:<6}{note}")
    if res.audit['gyro_sat_before_release'] > 0:
        print(f"  NOTE: {res.audit['gyro_sat_before_release']} gyro samples "
              "saturated before or at release. Attitude and elevation angle "
              "are suspect for this log.")

    # --- Attitude truth metric ---
    # Stats stop at impact: post-impact attitude is meaningless (that data is
    # ignored by the whole pipeline) and tumbling wreckage samples that
    # happen to fall in-gate would otherwise dominate the max.
    print()
    end = ph.impact_idx if ph.impact_idx is not None else len(res.truth_angle)
    finite = res.truth_angle[:end]
    finite = finite[np.isfinite(finite)]
    if len(finite):
        print(f"Attitude truth metric (predicted vs measured gravity "
              f"direction, in-gate samples, pre-impact):")
        print(f"  mean {np.mean(finite):.2f} deg   max {np.max(finite):.2f} deg   "
              f"({len(finite)} samples in gate)")
    else:
        print("Attitude truth metric: no in-gate samples before impact")

    # --- Yaw: excursion and drift are different things ---
    print()
    ys = res.yaw_stats
    print(f"Yaw (relative heading, unobservable absolute reference):")
    print(f"  excursion:      {ys['excursion_deg']:.2f} deg total range "
          "(includes real rotation, NOT drift)")
    if ys['drift_deg_per_min'] is not None:
        print(f"  drift:          {ys['drift_deg_per_min']:+.3f} deg/min "
              f"(fitted across {ys['n_quiet_windows']} quiet windows)")
    else:
        print("  drift:          not estimable (no usable quiet window)")

    print()
    print(f"Attitude init:    {res.q0_source}")
    print('=' * w)


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
    ax2.axhline(p.flight_var_thresh, color='purple', linewidth=1.2,
                linestyle='--',
                label=f'flight_var_thresh = {p.flight_var_thresh:g}')
    ax2.set_ylabel(f'rolling var of |accel| '
                   f'({p.flight_var_window:.1f} s win, log scale)')
    ax2.set_xlabel('Time (s)')
    ax2.grid(True, alpha=0.3, which='both')

    # Overlay phase boundaries on BOTH subplots
    for ax in (ax1, ax2):
        for i, (i0, i1) in enumerate(ph.quiet_windows):
            ax.axvspan(t[i0], t[i1], alpha=0.15, color='green',
                       label='quiet window' if i == 0 else None)
        if ph.onset_idx is not None:
            ax.axvline(t[ph.onset_idx], color='orange', linewidth=1.2,
                       label='onset')
        if not ph.v0_fallback:
            ax.axvline(t[ph.v0_idx], color='green', linewidth=1.2,
                       linestyle=':', label='v=0')
        if ph.flight_start is not None:
            ax.axvline(t[ph.flight_start], color='blue', linewidth=1.2,
                       linestyle='--', label='FLIGHT entry')
        if ph.impact_idx is not None:
            ax.axvline(t[ph.impact_idx], color='red', linewidth=1.2,
                       label='IMPACT')

    # When detection failed, say so ON the figure: this plot exists to debug
    # exactly that case in the field
    if ph.flight_start is None:
        ax2.text(0.02, 0.95,
                 f"No FLIGHT phase found (machine stuck at {ph.stuck_state}).\n"
                 "Read the flight plateau off this plot and set "
                 "flight_var_thresh above it.",
                 transform=ax2.transAxes, va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='lightyellow',
                           edgecolor='orange'))

    ax1.legend(loc='upper right', fontsize=8)
    ax2.legend(loc='upper right', fontsize=8)
    plt.tight_layout()

    fig.savefig(save_path, dpi=150)
    print(f"Diagnostic plot saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)