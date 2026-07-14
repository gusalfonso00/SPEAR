# SPEAR Field Test Guide

Procedure for javelin throw capture and analysis with the field pipeline
(`analyze_field_throw.py`). Written for field day: short commands, what to
look for, what can go wrong.

## Before leaving the house

- [ ] `secrets.h` has the hotspot SSID and the laptop's IP on that hotspot.
      Check the IP after connecting to the hotspot, not at home: it changes.
      Run `ifconfig en0` and update `TARGET_IP` if needed, reflash.
- [ ] Verify calibration is current. With the sensor mounted on the javelin,
      use the sphere-fit script: capture one log holding the javelin still
      in 8-10 different orientations (~10 s each, hands off, mix in tip-up
      and tip-down poses, do not only roll it), then run
      `python characterize_bias_mounted.py`. The capture protocol is in that
      script's docstring. Do it at field temperature if possible.
      (`characterize_bias.py` is the bench alternative and needs the sensor
      flat and unmounted.)
- [ ] Set `release_height_m` in `SpearParams` (top of `spear_analysis.py`)
      to the thrower's actual release height above the landing plane.
      Shoulder height plus extended arm, roughly. Default is 2.0 m.
- [ ] Full charge on the power bank. Modem sleep is disabled in firmware, so
      draw is a steady ~170 mA; expect hours, not days.
- [ ] Bring a tape measure. The ballistic range prediction is only useful
      if there is a measured distance to compare against.
- [ ] Bench sanity check the night before: power everything, capture 30 s
      stationary, run `python analyze_field_throw.py` and confirm one QUIET
      window, no onset, zero clipping.

## Per-throw protocol

One throw per log. Every throw:

1. Start the logger:

```bash
python log_imu_udp.py
```

2. Thrower holds the javelin roughly still for 1-2 seconds. Any orientation
   is fine, including cocked behind the shoulder. Do not hold much longer
   than that: the summary flags gaps over 3 s because bias leaks into the
   velocity integral over the wait.
3. Throw from standing, no run-up.
4. Let the javelin land and settle. Stop the logger (Ctrl+C).
5. Tape-measure the distance and write it down with the log timestamp.

## First throw of the day: tune the flight threshold

`flight_var_thresh` in `SpearParams` cannot be known until a real throw has
been logged. It ships with a provisional bench-derived value (0.5) that has
never seen a javelin fly.

```bash
python analyze_field_throw.py --diagnostic
```

This analyzes the most recent log, prints the summary, shows the diagnostic
plot, and saves the plot as a PNG next to the log file.

**If FLIGHT was detected:** the summary prints a line like

```
SUGGESTED flight_var_thresh = 0.0038  (4x FLIGHT median, source: ...)
```

Copy that number into `flight_var_thresh` in `SpearParams` (top of
`spear_analysis.py`), save, and re-run. Done for the day.

**If FLIGHT was NOT detected:** the plot shows a yellow box with the state
the machine got stuck in, and the summary prints post-onset variance
percentiles. On the log-scale variance panel, find the flat low plateau
between the throw spike and the impact spike: that is flight. Set
`flight_var_thresh` a factor of a few above the plateau level and re-run.

## Every subsequent throw

```bash
python analyze_field_throw.py
```

Add `--diagnostic` whenever a result looks off. To analyze an older log:

```bash
python analyze_field_throw.py imu_log_YYYYMMDD_HHMMSS.csv --diagnostic
```

## Reading the summary: what to watch for

| Line | Meaning | Action |
|---|---|---|
| `>>> NEEDS REVIEW <<<` | Flight duration outside the 1-4 s sanity band | Look at the diagnostic plot before trusting anything |
| `speed: >= X m/s (LOWER BOUND: ...)` | Accel railed during the pull | True release speed is higher; note it as a bound |
| Gyro saturation note | Gyro exceeded 2000 dps before release | Elevation angle suspect for this throw |
| `gap X.X s` note in integration window | Held still too long before throwing | Start the throw sooner after settling |
| `No quiet window found before throw onset` (error) | Hold was too short or too shaky | Re-throw with a deliberate 1-2 s hold |
| Onset found but no FLIGHT | Threshold not tuned, or flight shorter than 0.5 s | Run `--diagnostic`, read the plateau |
| IMPACT accel clipping | Ground strike railed the sensor | Expected, informational only |
| Large difference, measured vs predicted flight time | Aero effects (the prediction is a vacuum model) | Normal for a javelin; note the gap, it is data |

Packet loss checks still apply in the field: `log_imu_udp.py` prints the
drop rate live. If loss goes above a few percent, move the laptop closer or
re-orient it (see the link characterization results: line of sight was 0.2%
loss at 100 m, a body/car obstruction pushed it to ~9%).

## Physical tests wanted (in priority order)

These validate specific assumptions the pipeline currently takes on faith.
Each is one short capture.

1. **Handheld quiet hold, throwing grip, cocked position.** 10 s holding
   the javelin still-ish behind the shoulder. Validates that real arm
   tremor passes the quiet detector (`quiet_accel_tol` 0.3 m/s^2,
   `quiet_gyro_tol` 10 dps). If no QUIET window is found, those tolerances
   need loosening before any throw analysis works.

2. **Throw motion WITHOUT release.** Full arm swing, hold onto the javelin,
   follow through. Should produce an onset but NO flight and NO impact.
   This is the false-positive test for the flight detector: follow-through
   motion must not read as flight.

3. **Gentle underhand toss onto grass, 2-3 m.** Lowest-severity end-to-end
   test of the full chain (onset, flight, impact). Two things checked at
   once: whether a soft grass landing actually rails the accelerometer
   (impact detection currently assumes 90% of 32 g OR a 50 m/s^2 step; a
   soft landing may only trigger the step path), and whether short flights
   get flagged needs-review correctly (a 2-3 m toss flies under 1 s, so
   the review flag SHOULD fire).

4. **Drop test from measured height.** Hold at a measured height (e.g.
   2.0 m to the tip), release with zero velocity, land on grass. Vacuum
   flight time is exactly sqrt(2h/g), about 0.64 s for 2 m, no aero
   ambiguity. Checks measured flight time and impact timing against ground
   truth. Expect the needs-review flag (under 1 s); that is correct
   behavior, ignore it for this test.

5. **Real throw with tape-measured distance.** The main event. Compare
   summary range prediction against tape. The vacuum model should
   overpredict a flat throw and the gap is the aero signal.

6. **Spin check during a real throw.** After the first throw, look at the
   clipping audit gyro column for FLIGHT. A javelin can spin about its long
   axis at release; if gyro samples saturate (2000 dps) during flight,
   attitude through flight is compromised and that changes how much to
   trust the elevation angle on later throws.

7. **Multi-pose calibration capture (mounted, any day).** The 8-10 pose
   sphere-fit capture described in the pre-departure checklist, consumed by
   `characterize_bias_mounted.py`. This separates true accel bias from
   mounting tilt without unmounting anything (the old single-orientation
   calibration confounds them on X/Y, and the rotation bench log shows
   about +0.3 m/s^2 of it). Re-do after any remount or big temperature
   change. Scale factor is still not fitted; that would need a bench
   six-face set with the sensor unmounted.

8. **Wi-Fi loss at throw geometry.** One capture with the javelin at the
   far end of the expected flight path, laptop at the throwing line. The
   link tests covered 100 m line of sight (0.2% loss), but a javelin in
   grass is at ground level where Fresnel-zone loss is worst. If loss is
   bad at ground level, expect the tail of each log (post-landing) to be
   gappy and rely on the pre-landing data.

## Files produced per session

- `Data Logs/imu_log_*.csv` - raw captures (gitignored)
- `Data Logs/imu_log_*_diagnostic.png` - diagnostic plots, saved next to
  their log (gitignored with the folder)
- Your notebook: log timestamp, taped distance, wind notes. The pipeline
  cannot recover these later.
