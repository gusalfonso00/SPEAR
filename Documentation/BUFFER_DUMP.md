# SPEAR Onboard Buffer and Flash Dump

Why this exists: field link tests showed burst packet loss over 100 ms in
bad javelin orientations, and the release window of a throw is about
100 ms. The live UDP stream therefore cannot be the data of record. The
ESP32 now keeps the most recent 60 s in RAM continuously; a FREEZE command
persists it to flash, and a TCP dump retrieves it once the javelin is back
in range. The UDP stream is unchanged and serves as a live health monitor.

## Ports

| Port | Transport | Direction | Purpose |
|---|---|---|---|
| 4210 | UDP | ESP32 -> host | Live sample stream (unchanged) |
| 4211 | UDP | host -> ESP32 | FREEZE command, ACK reply |
| 4212 | TCP | host <-> ESP32 | LIST / GET / CLEAR / QUIT dump protocol |

Command traffic has its own UDP port so it never mixes with the outbound
sample stream on 4210.

## Sample format and ring sizing

Each buffered sample is 18 bytes, packed, little-endian:

```
uint32  t_ms    millis() at the sensor read
uint16  seq     low 16 bits of the same counter the UDP stream uses
int16   ax, ay, az   raw accel words (+/-32 g full scale, 0.976 mg/LSB)
int16   gx, gy, gz   raw gyro words (+/-2000 dps full scale, 70 mdps/LSB)
```

Ring buffer: 6000 samples x 18 B = 108,000 B. 6000 samples at 100 Hz =
60 s of history. The 60 s window is the operator contract: a throw must be
frozen within 60 s or it scrolls out.

Allocation: one heap_caps_malloc at the very top of setup(), before Wi-Fi
initializes, hard-checked with a halt-and-print on failure, never freed.
A static array was the first choice and does not link on this chip: the
ESP32's static-data segment (dram0_0_seg, ~160 KB) also carries the
Arduino core and Wi-Fi stack statics, and 108 KB overflows it by ~32 KB.
The heap reaches DRAM the static segment cannot, and pre-Wi-Fi it is one
unfragmented block, so the boot-time allocation is deterministic. Boot
prints free heap twice: after the ring allocation and again after Wi-Fi,
so both costs are visible (expect very roughly 180 KB then 120-150 KB;
below ~60 KB after Wi-Fi, investigate before flying).

Raw int16 words are stored instead of floats: half the size, zero rounding,
and the host decoder applies the exact scale factors from the Adafruit
driver source (`Adafruit_LSM6DSO32.cpp::_read()`), so decoded CSVs match
the UDP stream values. The struct has no temperature field; decoded CSVs
carry `nan` in the temp_C column.

## FREEZE (UDP port 4211)

Host sends the ASCII line `FREEZE\n` at 10 Hz until acked (15 s timeout).
On the first FREEZE the ESP32:

1. Increments the persistent counter in `/throw_counter.txt` and writes it
   back BEFORE writing data, so a number is never reused even if power
   dies mid-flush. CLEAR never touches this file, so numbering climbs
   forever and old local copies are never shadowed by new files.
2. Writes the ring oldest-sample-first to `/throw_NNN.bin` on LittleFS.
3. Replies `ACK FREEZE throw_NNN.bin <byte_count>\n` to the sender.
4. Resumes ring buffering.

Snapshot correctness needs no locking: sampling and the flush both run
cooperatively inside `loop()`, so no sample write can interleave with the
flush. This is the simpler correct option; double buffering would buy
nothing because the flush stall is acceptable anyway (the event is already
captured, and the stream is only a health monitor).

Duplicate handling: any FREEZE arriving within 2 s of a completed flush
gets the same ACK line and creates no new file. The host's 10 Hz retries
queue up during the multi-second flush and drain harmlessly through this
window.

Failure path (not part of the nominal protocol): if the flash write fails
the ESP32 replies `ERR FREEZE flash write failed\n`, which the host prints
instead of retrying silently into a timeout.

## TCP dump protocol (port 4212)

Line-based commands from the client; one client at a time.

| Command | Reply |
|---|---|
| `LIST\n` | one `<filename> <size_bytes>\n` per throw file, then `END\n` |
| `GET <filename>\n` | `SIZE <size_bytes>\n` then exactly that many raw bytes; connection stays open |
| `CLEAR\n` | deletes all `throw_*.bin` (never the counter file), replies `CLEARED <n>\n` |
| `QUIT\n` | closes the connection |

`GET` of a name not in LIST replies `ERR NOFILE\n` (protocol extension for
diagnosability; the host only requests names it got from LIST).

## What stalls when

| Activity | Effect on sampling and streaming |
|---|---|
| Ring buffering (always on) | none - one 18 B struct copy per sample |
| FREEZE flash flush | both stall for a few seconds; acceptable, the event is already in the snapshot |
| TCP file transfer | both stall for the transfer duration; acceptable, dumps happen between throws |
| First-ever LittleFS mount | setup() blocks several seconds while formatting; happens once per chip, never lazily |

## Host workflow

Operator instructions live in the `log_imu.py` docstring. Summary: launch,
throw, press F within 60 s, repeat; carry javelin back, press D to download
and verify (PASS/FAIL per file, decoder writes `throw_NNN.csv` beside each
verified bin); press C (y/N confirm) to clear flash only after PASS lines.
D never deletes anything and re-running it skips files already downloaded.

Verification per file: byte count matches LIST and is a multiple of 18;
seq increments by exactly 1 (at most one uint16 wraparound); t_ms strictly
monotonic with steps of at least 5 ms. Forward t_ms gaps LARGER than
nominal with continuous seq are sampling stalls, not corruption: a freeze
flush pauses sampling ~0.3 s, so any freeze within 60 s of a previous one
contains that pause. Verified files report their stalls in the PASS line
(e.g. "1 sampling stall(s), max 335 ms"); corrupted timestamps still fail
because a mangled value breaks monotonicity at the next sample. Decoded
CSVs use the same column format as session CSVs, so the whole analysis
pipeline consumes them unchanged.

## Bench-test checklist (hardware required, in order)

1. **Boot heap prints.** Flash, open serial monitor. Confirm the "Ring
   buffer: 6000 samples ... allocated pre-WiFi" line, then the free-heap
   line after Wi-Fi connects. Record both numbers; the post-Wi-Fi figure
   must comfortably clear the radio's working needs (expect > 60 KB).
2. **First-mount format.** On a chip that has never run LittleFS: confirm
   the "mount failed, formatting" then "formatted and mounted" serial
   lines, and that the boot takes several extra seconds only this once.
3. **FREEZE ack round trip.** Run `log_imu.py`, confirm stream packets
   arriving, press F. Expect the ACK with `throw_001.bin 108000` (or
   fewer bytes if the ESP32 has been up less than 60 s - the ring is not
   full yet).
4. **Duplicate-FREEZE suppression.** Press F twice within 2 s. Second
   press must return the SAME filename immediately, and LIST must show
   only one file.
5. **D dump of two files.** Freeze twice (more than 2 s apart), press D.
   Both files download, both print PASS, both decoded CSVs appear in
   `Data Logs/throws/`.
6. **PASS verification content.** Open a decoded CSV, confirm ~10 ms t_ms
   spacing and plausible accel values (about 9.8 m/s^2 magnitude at rest).
   Run `python analyze_field_throw.py "Data Logs/throws/throw_001.csv"`
   end to end.
7. **C clear.** Press C, confirm y, expect `CLEARED 2`. Press D again:
   "no throw files on ESP32 flash".
8. **Counter persistence.** Power-cycle the ESP32, freeze again: the new
   file number must continue from before the clear (e.g. throw_003.bin),
   never restart at 001.
