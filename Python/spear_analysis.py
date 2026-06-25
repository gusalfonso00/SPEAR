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

    df = pd.read_csv(filename)
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