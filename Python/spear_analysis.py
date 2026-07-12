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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


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
        csv_files = glob.glob(os.path.join(data_dir, "imu_log_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        filename = max(csv_files, key=os.path.getmtime)
        print(f"Loading most recent log: {os.path.basename(filename)}")
    else:
        if not os.path.isabs(filename):
            filename = os.path.join(data_dir, filename)
        print(f"Loading: {os.path.basename(filename)}")

    # log_imu_udp.py writes a header row, but headerless captures exist too.
    # Sniff the first line: if it starts with 'seq' it's a header, otherwise
    # assume the standard 9-column order and assign names ourselves.
    columns = ['seq', 'ms', 'temp_C', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
    with open(filename) as f:
        has_header = f.readline().lstrip().startswith('seq')
    if has_header:
        df = pd.read_csv(filename)
    else:
        df = pd.read_csv(filename, header=None, names=columns)

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