# SPEAR - Sensor Pack for Evaluating Athletic Releases

Inertial telemetry and analysis for a thrown javelin. An ESP32 samples a 6-axis IMU at 100 Hz, streams live over Wi-Fi UDP, and continuously records the last 60 seconds into an onboard ring buffer that is frozen to flash after each throw - so the throw data survives even when the RF link does not. A Python ground station manages capture, and an analysis pipeline handles bias calibration, quaternion attitude estimation, gravity removal, throw phase detection, and release-state estimation.

Field validation so far: on the first three tape-measured throws, release velocity integrated from the IMU predicted the landing point within 5% of the measured distance (vacuum ballistics, 7-16 m throws).

Built to practice sensor fusion and test-like engineering discipline directly relevant to flight systems: calibration, attitude tracking under dynamic loading, telemetry link budgeting, data-of-record design, and validating estimates against independent ground truth.

## Hardware

| Component | Part | Notes |
|---|---|---|
| IMU | ST LSM6DSO32 | ±32 g accel, ±2000 DPS gyro |
| MCU | ESP32 HiLetgo 38-pin | I2C to IMU; Wi-Fi UDP + TCP; LittleFS flash storage |
| Host | macOS | Python 3.13 capture and analysis |

The sensor ODR is 208 Hz while the firmware polls at 100 Hz: intentional oversampling that avoids beat-frequency aliasing between the sensor clock and the software scheduler.

## Two data paths

Controlled link testing showed why streaming can't be trusted with a throw: rotating the javelin through 90-degree holds at 30 m found a single orientation - an antenna null - that loses 78% of packets while the other orientations lose under 2%, and a throw's release window is about 100 ms. So the live UDP stream (port 4210) is a health monitor, and the data of record is onboard: an 18-byte packed sample struct rings through a 108 KB buffer (60 s), frozen to LittleFS on command (UDP port 4211), then downloaded and verified over TCP (port 4212) once the javelin is retrieved. Downloads are integrity-checked (sequence continuity, timestamp monotonicity, byte counts) and auto-decoded to the same CSV format as the live stream, so the whole analysis pipeline consumes either path unchanged.

The ring buffer is heap-allocated once at boot, before Wi-Fi initializes: a 108 KB static array does not fit the ESP32's static-data segment alongside the radio stack's own statics, while the pre-Wi-Fi heap is a single unfragmented block.

## Analysis pipeline

**Calibration.** Two paths writing one `imu_calibration.json`: a bench characterization (sensor flat, gravity on one axis) and an in-situ sphere fit for the mounted sensor - hold the javelin still in 8-10 arbitrary orientations and solve for the bias vector that puts every pose on a sphere of radius local gravity. No orientation assumption, no unmounting, warns when the pose set is degenerate (coplanar). Local gravity (9.7966 m/s² at this site) is used for physics; standard gravity only for unit definitions.

**Attitude and gravity removal.** A quaternion complementary filter (scalar-first, body-to-world, written from scratch in NumPy) integrates the gyro and corrects tilt against the accelerometer only when |accel| is near 1 g, with per-sample timesteps so packet gaps integrate over true elapsed time. Gravity is projected out in the world frame; yaw is unobservable without a magnetometer and is documented as relative.

**Throw analysis** (`analyze_field_throw.py`). A phase state machine finds the throw in a 60 s record and anchors on its two loudest features: impact = the global |accel| peak (landings hit 33-49 g), flight = from the pull peak to the impact's rising edge. Velocity integrates from a v=0 anchor at the stillest second of the pre-throw hold - chosen by minimum variance rather than absolute thresholds, because a handheld hold never passes bench-grade stillness tests. The tool trims each record to the throw automatically, reports release speed and elevation angle, compares a vacuum-ballistics range prediction against the tape-measured distance, and audits data quality (sensor clipping, gyro saturation, attitude-estimate health).

**Supporting tools**: a throw window viewer for quick inspection of any capture.

### Design notes

The flight detector is the third design; real data killed the first two. A "flight is smooth" variance threshold turned out backwards - a javelin shaft rings hard after release (measured flight variance 100-1000x the resting level), so the detector preferred the post-impact grass. A variance band fixed that but left thin margins between gentle pulls and hard flights. Peak anchoring replaced both with no tunable thresholds, at the documented cost of assuming the landing is louder than the pull. The failure history and rationale live in the code comments.

A complementary filter was chosen over an EKF deliberately: every line of the correction step is a physics statement that can be validated against stationary ground truth, and there are no noise matrices to tune without a validated aerodynamic model. The upgrade path (EKF with a javelin dynamics model, improving yaw observability during the throw) is understood and intentional.

## Repo structure

```
Arduino/spear_imu_udp/
    spear_imu_udp.ino          firmware: 100 Hz sampling, UDP stream, boot sequencing
    throw_buffer.h/.cpp        ring buffer, FREEZE-to-flash, TCP dump server
    secrets.h.example          Wi-Fi credential template (copy to secrets.h, never commit)

Python/
    log_imu.py                 ground station: live logging + flash freeze/dump/clear
    characterize_bias.py       bench bias characterization (sensor flat)
    characterize_bias_mounted.py  in-situ sphere-fit calibration (sensor mounted)
    spear_analysis.py          analysis library: params, phase detection, audits, plots
    spear_filter.py            quaternion complementary filter, gravity removal
    analyze_field_throw.py     throw analysis: trim, release state, ballistic check
    plot_throw_window.py       clipped throw viewer
    imu_calibration.json       current calibration (sphere fit, mounted)
```

## Setup

**Firmware**: copy `secrets.h.example` to `secrets.h`, fill in Wi-Fi credentials and host IP, flash with Arduino IDE (needs the Adafruit LSM6DSO32 library and ESP32 board support).

**Python**: `pip install pandas numpy matplotlib scipy`

**Field workflow**
```bash
python log_imu.py
python analyze_field_throw.py --distance 16.2
```
Run the logger, throw, press F to freeze the buffer, retrieve, press D to download and verify. The analyzer picks up the newest capture, trims it to the throw, and reports release state and prediction error against the taped distance.

## Status and scope

Personal instrumentation project, not production flight software. Current state and limits:

- Release-state estimation validated on short throws (7-16 m): range predictions within 5% of tape.
- Impact detection assumes the landing outreads the pull (true on grass; a net catch would not anchor).
- Accelerometer scale factor is not yet calibrated (bias only); the 3+ g throw pull leans on it.
- Yaw is not observable from six axes; headings are relative.
- Flight-time predictions run ~0.2-0.3 s long while range lands within 5%: the vacuum model's vertical-axis error, visible because everything else got accurate enough to expose it.

## Skills demonstrated

- Embedded systems: ESP32 firmware, I2C, fixed-rate scheduling, memory-segment budgeting, LittleFS storage, custom TCP/UDP protocols
- Sensor fusion: quaternion attitude estimation, complementary filtering, observability analysis
- Calibration: static and in-situ sphere-fit bias estimation, local gravity correction, noise characterization
- Signal processing: windowed integration, event detection in noisy series, saturation auditing
- Verification discipline: synthetic-truth tests, bench regression suites, hardware checklists, independent ground-truth validation (tape measure vs prediction)
- Python scientific stack: NumPy, SciPy, Matplotlib, Pandas
