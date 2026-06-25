# CLAUDE.md - SPEAR

## Project overview
SPEAR is an instrumented javelin telemetry system. An LSM6DSO32 IMU mounted on a javelin streams 6-axis inertial data plus die temperature (3-axis accel, 3-axis gyro) over Wi-Fi UDP to a Mac host at 100 Hz. A Python logger captures packets to timestamped CSV files. A Python analysis library then loads those logs, calibrates sensor bias from a stationary pre-throw window, integrates acceleration to velocity, and plots results. The current pipeline is Step 1 of a planned multi-step analysis: raw integration without gravity removal, so velocity drift is expected and motivates Step 2 (orientation tracking + gravity subtraction).

## Hardware
- **IMU**: ST LSM6DSO32 - accelerometer at ±32 g range, gyroscope at ±2000 DPS. ODR set to 208 Hz internally; firmware polls and transmits at 100 Hz via a fixed-rate scheduler. The oversampling is intentional - it avoids beat-frequency aliasing between sensor and scheduler.
- **MCU**: HiLetgo ESP32 DevKit (38-pin). Communicates with the IMU over **I2C** (`dso32.begin_I2C()`).
- **Transport**: Wi-Fi UDP, target port 4210. The Mac's IP is hardcoded in `secrets.h` (see below).

## Repo layout
```
Arduino/          - ESP32 firmware (Arduino IDE sketch)
Documentation/    - ESP32 pinout diagram, sample output screenshot
Python/           - All host-side code
Python/Data Logs/ - Timestamped CSV captures (gitignored)
```

## Current pipeline
1. **Firmware** (`Arduino/spear_imu_udp/spear_imu_udp.ino`) - polls IMU at 100 Hz, sends one UDP packet per sample as a 9-field CSV: `seq,ms,temp_C,ax,ay,az,gx,gy,gz`.
2. **Logger** (`Python/log_imu_udp.py`) - binds UDP port 4210, validates packets, writes rows to `Python/Data Logs/imu_log_YYYYMMDD_HHMMSS.csv`. Tracks sequence gaps and reports drop rate.
3. **Quick plot** (`Python/imu_plot.py`) - loads most recent CSV, renders 3-panel plot (temp, accel XYZ, gyro XYZ) with NaN breaks at gaps >200 ms.
4. **Analysis library** (`Python/spear_analysis.py`) - functions: `load_log`, `calibrate_bias`, `apply_bias`, `integrate_velocity`, `plot_acc_and_vel`.
5. **Analysis script** (`Python/analyze_throw.py`) - Step 1 pipeline: load -> calibrate bias (0-3 s stationary window) -> subtract bias -> trapezoidal integrate to velocity -> plot.

## Dev environment
- Firmware: Arduino IDE on macOS, with the Adafruit LSM6DSO32 library and standard ESP32 board support.
- Python: macOS, Python 3.13. Dependencies: `pandas`, `numpy`, `matplotlib`, `scipy`.
- Data collection: `python log_imu_udp.py` (120 s default), then `python analyze_throw.py` or `python imu_plot.py`.

## Credentials
Wi-Fi SSID, password, and target IP live in `Arduino/spear_imu_udp/secrets.h`, which is gitignored. Copy `secrets.h.example` and fill in your values. Never commit `secrets.h`.

## Coding preferences
- Default to working code over pseudocode; comment generously - this is an embedded + signal-processing project with non-obvious behavior.
- For sensor debugging, walk through the physics first (what should the signal look like, what are the expected magnitudes?) before diving into code fixes.
- Flag LSM6DSO32 gotchas:
  - FIFO is not used here - raw polling only. If FIFO is added, ODR and watermark interaction matters a lot.
  - ODR (208 Hz) vs. output poll rate (100 Hz) - the firmware intentionally oversamples to avoid beat-frequency aliasing. Changing ODR without updating the scheduler will alias or starve the pipeline.
  - The ±32 g range is set via `setAccelRange(LSM6DSO32_ACCEL_RANGE_32_G)` - confirm this register write actually took if readings look saturated or scaled wrong (read back the range register).
  - Temperature output has a fixed offset; don't use it for absolute temperature without calibration.
- Flag ESP32 gotchas:
  - I2C SDA/SCL pin defaults on this devkit - double-check against `Documentation/esp32_pinout.png` if the IMU isn't found at boot.
  - `micros()` wraps at ~71 minutes; the fixed-rate scheduler uses `(int32_t)` cast to handle wrap correctly - preserve that cast if modifying the scheduler.
  - Wi-Fi increases power draw significantly; battery life will be much shorter than a bare ESP32 sketch.
  - UDP is fire-and-forget - packet loss is expected, especially during a throw. The logger tracks sequence gaps; don't paper over drops in analysis.
- Push back on bad designs, don't just validate. If an approach has a fundamental flaw (drift accumulation, gravity projection error, aliasing), say so before implementing.
- No em dashes - use plain hyphens or restructure the sentence.

## Known issues / TODOs
- Step 2: orientation tracking (Madgwick or complementary filter) + gravity subtraction in inertial frame, then re-integrate. Current Step 1 drift is expected, not a bug.
