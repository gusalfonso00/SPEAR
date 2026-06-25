# SPEAR - Javelin Telemetry System

Real-time 6-axis inertial telemetry for a thrown javelin. An ESP32 microcontroller streams accelerometer and gyroscope data at 100 Hz over Wi-Fi UDP to a host computer, where a Python pipeline handles bias calibration, quaternion-based attitude estimation, and gravity removal to recover linear acceleration and integrated velocity in the world frame.

Built to practice sensor fusion techniques directly relevant to flight systems: IMU calibration, attitude tracking under dynamic loading, and the engineering tradeoffs between complementary and Kalman filter architectures.

## Hardware

| Component | Part | Notes |
|---|---|---|
| IMU | ST LSM6DSO32 | ±32 g accel, ±2000 DPS gyro |
| MCU | ESP32 HiLetgo 38-pin | I2C bus, Wi-Fi UDP transport |
| Host | macOS | Python 3.13 capture and analysis |

The sensor ODR is set to 208 Hz while the firmware polls at 100 Hz. This intentional oversampling avoids beat-frequency aliasing between the sensor clock and the software scheduler - a subtle failure mode that appears when the two rates are close rather than comfortably separated.

## Signal Processing Pipeline

**Step 0 - Bias characterization** (`characterize_bias.py`)

Loads a stationary capture, trims settling transients, and characterizes static bias and noise floor per axis. Writes `imu_calibration.json` for reuse across sessions. Uses local gravitational acceleration (9.7966 m/s² for Boulder, CO) rather than standard gravity (9.80665 m/s²) - a ~10 mg difference that matters at the precision target for this sensor.

**Step 1 - Raw integration** (`analyze_throw.py`)

Applies stored calibration, integrates bias-corrected acceleration with the trapezoidal rule, and plots the result. Velocity drifts visibly over seconds because gravity is not subtracted - this is expected and motivates Step 2.

**Step 2 - Attitude tracking and gravity removal** (`analyze_throw_step2.py`)

Complementary filter over gyro integration and accelerometer tilt correction. Quaternion representation throughout (scalar-first, body-to-world). Euler angles are computed only at the output for plotting - they are never used inside the filter, where gimbal lock at ±90° pitch would be a problem.

Accel correction gate (0.7-1.3 g): suppresses tilt correction during the throw phase when linear acceleration swamps the gravity reference. Gyro integrates freely during that window. After re-convergence on stationary test data: peak speed < 0.02 m/s, linear acceleration residual ~2 mg (within raw sensor noise).

### Filter design rationale

A complementary filter was chosen over an EKF for this stage:

- It is transparent - every line of the correction step is a direct physics statement, which makes it easier to validate against known ground truth (stationary logs).
- It has no process/measurement noise matrices to tune, which removes a free-parameter problem when there is no validated dynamic model for javelin aerodynamics.
- For a ~0.5 s flight window, gyro drift during the accel gate interval is small enough that it does not dominate the velocity estimate.

The fundamental observability limit is yaw: a 6-axis IMU cannot resolve rotations around the gravity axis. Yaw drifts at the residual gyro bias rate (~0.04 deg/min after calibration on test hardware). This is a physical constraint, not a tuning issue, and is documented rather than papered over.

A full EKF with a javelin flight dynamics model would improve yaw observability during the throw (aerodynamic forces have a known direction relative to the body axis) and better constrain the velocity estimate during the accel gate interval. That is the planned next step.

## Repo Structure

```
Arduino/spear_imu_udp/
    spear_imu_udp.ino       ESP32 firmware - 100 Hz UDP telemetry
    secrets.h.example       Wi-Fi credential template (copy to secrets.h, never commit)

Python/
    log_imu_udp.py          UDP listener, writes timestamped CSV logs
    imu_plot.py             Quick 3-panel raw data plot
    characterize_bias.py    Bias/noise characterization, writes imu_calibration.json
    spear_analysis.py       Analysis library (load, calibrate, integrate, plot)
    spear_filter.py         Quaternion complementary filter and gravity removal
    analyze_throw.py        Step 1 script
    analyze_throw_step2.py  Step 2 script
    imu_calibration.json    Example calibration from stationary characterization run

Documentation/
    esp32_pinout.png        HiLetgo ESP32 pin reference
    output.png              Example Step 1 plot
```

## Setup

**Firmware**
1. Copy `Arduino/spear_imu_udp/secrets.h.example` to `secrets.h` in the same directory.
2. Fill in Wi-Fi SSID, password, and host IP address.
3. Flash to ESP32 with Arduino IDE. Requires the Adafruit LSM6DSO32 library.

**Python**
```
pip install pandas numpy matplotlib scipy
```

**Workflow**
```bash
# Characterize bias from a stationary session
python log_imu_udp.py                              # 120 s default capture
python characterize_bias.py --gravity-axis +z      # adjust axis to match mounting

# Capture and analyze a throw
python log_imu_udp.py
python analyze_throw.py          # Step 1: raw integration
python analyze_throw_step2.py    # Step 2: attitude tracking and gravity removal
```

Pass `--local-g <value>` to `characterize_bias.py` if running outside Boulder, CO.

## Status and Scope

This is a personal instrumentation project, not production flight software. Current limitations:

- Yaw is not observable from a 6-axis IMU.
- The accel gate is a fixed magnitude threshold, not a dynamically estimated noise model.
- UDP telemetry is fire-and-forget; packet loss during a throw is expected, tracked by sequence number, but not recovered.
- Velocity integration has been validated against stationary data. Throw analysis is ongoing as hardware testing continues.

## Skills Demonstrated

- Embedded systems: ESP32 firmware, I2C sensor interface, fixed-rate scheduler with micros() wraparound handling
- Sensor fusion: quaternion attitude estimation, complementary filter design and validation
- IMU calibration: static bias characterization, local gravity correction, noise floor quantification
- Signal processing: trapezoidal integration, gravity projection and removal, sensor gating
- Python scientific stack: NumPy, SciPy, Matplotlib, Pandas
- Engineering judgment: filter architecture tradeoffs, observability analysis, phased development with testable intermediate outputs
