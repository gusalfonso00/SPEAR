"""
SPEAR ground station: live UDP logging plus onboard flash dump control.

WHAT THIS DOES
    The ESP32 streams live samples over UDP (lossy, health monitor only)
    and also keeps the most recent 60 seconds in an onboard RAM buffer
    (lossless, the data of record). This script logs the live stream to a
    CSV and sends the commands that freeze and retrieve the onboard data.

THE ONE HARD RULE
    A throw must be FROZEN (press F) within 60 seconds of happening, or it
    scrolls out of the onboard buffer and is gone. Freeze first, ask
    questions later.

FIELD WORKFLOW
    1. Launch:  python log_imu.py
       Live logging starts immediately. The status line shows packet count
       and drop rate. Drops here are fine; flash has the real data.
    2. Throw.
    3. Press F within 60 s. The script sends FREEZE until the ESP32 acks
       with a filename (e.g. throw_007.bin). The javelin can lie in the
       grass; the file is safe on its flash.
    4. More throws: repeat 2-3. Each freeze makes a new file.
    5. Carry the javelin back near the ground station, press D.
       Downloads every throw file not already on this laptop, verifies
       each one, prints PASS or FAIL per file, and writes an analysis
       ready CSV next to each .bin. Nothing is deleted from the ESP32.
    6. After confirming PASS lines, press C to erase the ESP32 flash
       (asks y/N first). Only do this once local copies are verified.
    7. Ctrl-C ends the session.

OUTPUT FILES (all under "Data Logs/")
    session_YYYY-MM-DD_HH-MM-SS.csv   live UDP capture of this session
    throws/throw_NNN.bin              raw flash dump (lossless throw data)
    throws/throw_NNN.csv              decoded copy, same format as session
                                      CSVs; feed it to the analysis scripts

KEYS
    F  freeze the last 60 s to ESP32 flash        (do this after EVERY throw)
    D  download + verify + decode all throw files (javelin near station)
    C  clear ESP32 flash                          (only after PASS lines)
    Ctrl-C  end session
"""

import csv
import glob
import os
import select
import socket
import struct
import sys
import time
from datetime import datetime

# --- Ports: must match the firmware (see Documentation/BUFFER_DUMP.md) ---
UDP_DATA_PORT = 4210   # ESP32 -> here, live sample stream
UDP_CMD_PORT  = 4211   # here -> ESP32, FREEZE commands
TCP_DUMP_PORT = 4212   # LIST / GET / CLEAR / QUIT

# --- Decoder constants ---
# 18-byte packed little-endian sample written by the firmware ring buffer:
# uint32 t_ms, uint16 seq, int16 ax ay az gx gy gz (raw sensor words).
SAMPLE = struct.Struct('<IH6h')
SAMPLE_BYTES = SAMPLE.size   # 18

# Scale factors copied from the Adafruit_LSM6DSO32 driver source (_read()),
# so decoded values match the UDP stream exactly. Do not "simplify" these:
# they are the library's numbers, not datasheet ideals.
#   accel +/-32 g:    0.976 mg/LSB  -> m/s^2 via standard gravity
#   gyro +/-2000 dps: 70.0 mdps/LSB -> rad/s
ACCEL_SCALE = 0.976 * 9.80665 / 1000.0      # raw word -> m/s^2
GYRO_SCALE  = 70.0 * 0.017453293 / 1000.0   # raw word -> rad/s

# Verification tolerances: the firmware samples on a 10 ms fixed-rate
# scheduler, so consecutive t_ms steps should sit right at 10 ms.
T_STEP_MIN_MS = 5
T_STEP_MAX_MS = 15

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir   = os.path.join(script_dir, "Data Logs")
throws_dir = os.path.join(data_dir, "throws")


# ---------------------------------------------------------------------------
# Decoder: throw_NNN.bin -> throw_NNN.csv (importable, no side effects)
# ---------------------------------------------------------------------------

def decode_throw_bin(bin_path, csv_path=None):
    """Convert a flash dump to the same CSV format the UDP logger writes.

    Columns: seq, ms, temp_C, ax, ay, az, gx, gy, gz. The bin stores no
    temperature (18-byte samples, IMU words only), so temp_C is written as
    nan. seq in the bin is uint16 (low half of the stream counter); it is
    unwrapped here so the CSV seq increases monotonically like the stream's.

    Returns (n_samples, duration_s). csv_path defaults to the bin path with
    a .csv extension.
    """
    if csv_path is None:
        csv_path = os.path.splitext(bin_path)[0] + '.csv'

    with open(bin_path, 'rb') as f:
        blob = f.read()
    if len(blob) % SAMPLE_BYTES != 0:
        raise ValueError(f"{os.path.basename(bin_path)}: {len(blob)} bytes "
                         f"is not a multiple of {SAMPLE_BYTES}")

    n = len(blob) // SAMPLE_BYTES
    t_first = t_last = None
    seq_offset = 0        # accumulates 65536 per uint16 wraparound
    prev_seq16 = None

    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seq', 'ms', 'temp_C', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'])
        for i in range(n):
            t_ms, seq16, ax, ay, az, gx, gy, gz = SAMPLE.unpack_from(
                blob, i * SAMPLE_BYTES)
            if prev_seq16 is not None and seq16 < prev_seq16:
                seq_offset += 65536
            prev_seq16 = seq16
            if t_first is None:
                t_first = t_ms
            t_last = t_ms
            w.writerow([seq16 + seq_offset, t_ms, 'nan',
                        f"{ax * ACCEL_SCALE:.6f}",
                        f"{ay * ACCEL_SCALE:.6f}",
                        f"{az * ACCEL_SCALE:.6f}",
                        f"{gx * GYRO_SCALE:.6f}",
                        f"{gy * GYRO_SCALE:.6f}",
                        f"{gz * GYRO_SCALE:.6f}"])

    duration = (t_last - t_first) / 1000.0 if n > 1 else 0.0
    return n, duration


def verify_throw_bin(bin_path, expected_size=None):
    """Integrity check of a downloaded flash dump.

    Checks, in order:
      1. Byte count matches the LIST size (when given) and is an exact
         multiple of 18.
      2. seq increases by exactly 1 between consecutive samples, allowing
         at most one uint16 wraparound across the file.
      3. t_ms is strictly monotonic; steps shorter than the sampler could
         produce are corruption.

    Forward t_ms gaps LARGER than nominal with continuous seq are NOT
    failures: they are sampling stalls the firmware makes on purpose (a
    FREEZE flash flush or a TCP transfer pauses sampling, and any freeze
    taken within 60 s of one contains that pause). They get counted and
    reported in the PASS message. Genuine t_ms corruption still fails: a
    mangled timestamp breaks monotonicity at the next sample.

    Returns (True, "n samples X s (stalls...)") or (False, "reason naming
    the first violation").
    """
    size = os.path.getsize(bin_path)
    if expected_size is not None and size != expected_size:
        return False, f"size {size} does not match LIST size {expected_size}"
    if size % SAMPLE_BYTES != 0:
        return False, f"size {size} is not a multiple of {SAMPLE_BYTES}"
    if size == 0:
        return False, "file is empty"

    with open(bin_path, 'rb') as f:
        blob = f.read()
    n = size // SAMPLE_BYTES

    wraps = 0
    stalls = []
    prev_t = prev_seq = None
    for i in range(n):
        t_ms, seq16, *_ = SAMPLE.unpack_from(blob, i * SAMPLE_BYTES)
        if prev_seq is not None:
            if seq16 == (prev_seq + 1) & 0xFFFF:
                if seq16 < prev_seq:       # legitimate uint16 wraparound
                    wraps += 1
                    if wraps > 1:
                        return False, (f"sample {i}: more than one seq "
                                       f"wraparound")
            else:
                return False, (f"sample {i}: seq jumped {prev_seq} -> "
                               f"{seq16}, expected {(prev_seq + 1) & 0xFFFF}")
            step = t_ms - prev_t
            if step <= 0:
                return False, (f"sample {i}: t_ms not monotonic "
                               f"({prev_t} -> {t_ms})")
            if step < T_STEP_MIN_MS:
                return False, (f"sample {i}: t_ms step {step} ms below "
                               f"{T_STEP_MIN_MS} ms (sampler cannot run "
                               f"that fast; corrupt timestamp)")
            if step > T_STEP_MAX_MS:
                stalls.append(step)
        prev_t, prev_seq = t_ms, seq16

    duration = (prev_t - SAMPLE.unpack_from(blob, 0)[0]) / 1000.0
    msg = f"{n} samples {duration:.1f} s"
    if stalls:
        msg += (f" ({len(stalls)} sampling stall(s), max {max(stalls)} ms - "
                f"freeze/dump pauses, expected)")
    return True, msg


# ---------------------------------------------------------------------------
# ESP32 command paths
# ---------------------------------------------------------------------------

def send_freeze(esp_ip):
    """Send FREEZE at 10 Hz until ACKed or 15 s elapse. Returns ack or None.

    The firmware suppresses duplicate freezes for 2 s after completing one
    and re-sends the same ACK, so hammering at 10 Hz is safe by design.
    """
    cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cmd.settimeout(0.1)
    deadline = time.time() + 15.0
    print(f"\nFREEZE -> {esp_ip}:{UDP_CMD_PORT} (retrying up to 15 s)...")
    try:
        while time.time() < deadline:
            cmd.sendto(b"FREEZE\n", (esp_ip, UDP_CMD_PORT))
            try:
                reply, _ = cmd.recvfrom(256)
            except socket.timeout:
                continue
            line = reply.decode('utf-8', errors='ignore').strip()
            if line.startswith("ACK FREEZE"):
                print(f"  {line}")
                return line
            if line.startswith("ERR"):
                print(f"  ESP32 reported: {line}")
        print("  TIMEOUT: no ack in 15 s. Is the javelin in Wi-Fi range? "
              "The buffer keeps rolling; retry F as soon as link returns.")
        return None
    finally:
        cmd.close()


def _read_line(sock_file):
    line = sock_file.readline()
    if not line:
        raise ConnectionError("connection closed by ESP32")
    return line.decode('utf-8', errors='ignore').strip()


def dump_throws(esp_ip):
    """Connect over TCP, download every throw file we lack, verify, decode.

    Never sends CLEAR. Retries the connect at 1 Hz until it succeeds or the
    operator gives up with Ctrl-C (the javelin may still be being carried
    back; keep walking).
    """
    os.makedirs(throws_dir, exist_ok=True)

    tcp = None
    print(f"\nConnecting to {esp_ip}:{TCP_DUMP_PORT} (Ctrl-C to abort)...")
    while tcp is None:
        try:
            tcp = socket.create_connection((esp_ip, TCP_DUMP_PORT), timeout=2.0)
        except (socket.timeout, OSError):
            sys.stdout.write("\r  waiting for javelin in range...")
            sys.stdout.flush()
            time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n  dump aborted")
            return
    print("\r  connected                          ")

    tcp.settimeout(10.0)
    tf = tcp.makefile('rb')
    try:
        tcp.sendall(b"LIST\n")
        remote = []                      # [(filename, size), ...]
        while True:
            line = _read_line(tf)
            if line == "END":
                break
            name, size = line.rsplit(' ', 1)
            remote.append((name, int(size)))

        if not remote:
            print("  no throw files on ESP32 flash")
            return

        have = {os.path.basename(p) for p in
                glob.glob(os.path.join(throws_dir, 'throw_*.bin'))}
        todo = [(n, s) for n, s in remote if n not in have]
        print(f"  {len(remote)} file(s) on flash, {len(todo)} new")

        # Reconcile already-downloaded files that never got a decoded CSV
        # (e.g. an earlier verify version failed them): re-verify locally,
        # no re-download needed - the bytes are already here.
        for name, size in remote:
            if name in have:
                local = os.path.join(throws_dir, name)
                if not os.path.exists(os.path.splitext(local)[0] + '.csv'):
                    ok, msg = verify_throw_bin(local, expected_size=size)
                    if ok:
                        print(f"PASS {name} {msg} (local re-verify)")
                        decode_throw_bin(local)
                        print(f"     decoded -> {os.path.splitext(name)[0]}.csv")
                    else:
                        print(f"FAIL {name}: {msg} (local re-verify)")

        for name, size in todo:
            tcp.sendall(f"GET {name}\n".encode())
            header = _read_line(tf)
            if not header.startswith("SIZE "):
                print(f"FAIL {name}: unexpected reply {header!r}")
                continue
            nbytes = int(header.split()[1])
            blob = tf.read(nbytes)
            if len(blob) != nbytes:
                print(f"FAIL {name}: short read {len(blob)}/{nbytes} bytes")
                continue

            local = os.path.join(throws_dir, name)
            with open(local, 'wb') as f:
                f.write(blob)

            ok, msg = verify_throw_bin(local, expected_size=size)
            if ok:
                print(f"PASS {name} {msg}")
                # Decode immediately so the analysis-ready CSV exists
                # without a separate manual step
                decode_throw_bin(local)
                print(f"     decoded -> {os.path.splitext(name)[0]}.csv")
            else:
                print(f"FAIL {name}: {msg}")

        tcp.sendall(b"QUIT\n")
    except (ConnectionError, socket.timeout, OSError) as e:
        print(f"  dump interrupted: {e}. Downloaded files are kept; press D "
              "to resume (already-verified files are skipped).")
    finally:
        tf.close()
        tcp.close()


def clear_flash(esp_ip):
    """Explicit, confirmed CLEAR of all throw files on ESP32 flash."""
    sys.stdout.write("\nreally clear flash? [y/N] ")
    sys.stdout.flush()
    ch = sys.stdin.read(1)
    print(ch)
    if ch.lower() != 'y':
        print("  not cleared")
        return
    try:
        tcp = socket.create_connection((esp_ip, TCP_DUMP_PORT), timeout=5.0)
        tf = tcp.makefile('rb')
        tcp.sendall(b"CLEAR\n")
        print(f"  {_read_line(tf)}")
        tcp.sendall(b"QUIT\n")
        tf.close()
        tcp.close()
    except (OSError, ConnectionError) as e:
        print(f"  CLEAR failed: {e}")


# ---------------------------------------------------------------------------
# Main session loop: live UDP logging + non-blocking keys
# ---------------------------------------------------------------------------

def main():
    os.makedirs(data_dir, exist_ok=True)
    filename = os.path.join(
        data_dir, f"session_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', UDP_DATA_PORT))
    sock.setblocking(False)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)

    # Raw keypresses without Enter: cbreak mode, restored on exit. The
    # stream socket and stdin are watched by one select() loop.
    interactive = sys.stdin.isatty()
    if interactive:
        import termios
        import tty
        old_tty = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    received = drops = 0
    last_seq = None
    esp_ip = None          # learned from the stream's source address
    last_status = 0.0
    last_event = "waiting for stream"

    print(f"Logging live stream to {os.path.basename(filename)}")
    print("Keys: F freeze | D dump | C clear flash | Ctrl-C end session")
    print("Hard rule: press F within 60 s of a throw or it scrolls away.\n")

    try:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['seq', 'ms', 'temp_C',
                             'ax', 'ay', 'az', 'gx', 'gy', 'gz'])

            while True:
                readable, _, _ = select.select(
                    [sock, sys.stdin] if interactive else [sock], [], [], 0.2)

                if sock in readable:
                    # Drain everything queued so the status stays honest
                    while True:
                        try:
                            data, addr = sock.recvfrom(256)
                        except BlockingIOError:
                            break
                        esp_ip = addr[0]
                        parts = data.decode('utf-8',
                                            errors='ignore').strip().split(',')
                        if len(parts) != 9:
                            continue
                        try:
                            seq_v = int(parts[0])
                            [float(p) for p in parts[1:]]
                        except ValueError:
                            continue
                        writer.writerow(parts)
                        received += 1
                        if last_seq is not None and seq_v - last_seq - 1 > 0:
                            drops += seq_v - last_seq - 1
                        last_seq = seq_v

                if interactive and sys.stdin in readable:
                    key = sys.stdin.read(1).lower()
                    if key not in ('f', 'd', 'c'):
                        pass
                    elif esp_ip is None:
                        print("\nESP32 address unknown (no stream packet "
                              "received yet); cannot send commands")
                    elif key == 'f':
                        ack = send_freeze(esp_ip)
                        last_event = ack.strip() if ack else "freeze TIMEOUT"
                    elif key == 'd':
                        f.flush()
                        dump_throws(esp_ip)
                        last_event = "dump done"
                    elif key == 'c':
                        clear_flash(esp_ip)
                        last_event = "clear requested"

                # One-line status, refreshed at 2 Hz
                now = time.time()
                if now - last_status > 0.5:
                    last_status = now
                    total = received + drops
                    loss = 100.0 * drops / total if total else 0.0
                    sys.stdout.write(
                        f"\r  {received} pkts  {drops} dropped ({loss:.1f}%)"
                        f"  esp32={esp_ip or '?'}  last: {last_event}    ")
                    sys.stdout.flush()

    except KeyboardInterrupt:
        pass
    finally:
        if interactive:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty)
        sock.close()

    total = received + drops
    loss = 100.0 * drops / total if total else 0.0
    print(f"\n\nSession saved: {filename}")
    print(f"Received: {received}, Dropped: {drops}, Loss: {loss:.2f}%")
    print("Reminder: flash dumps in Data Logs/throws/ are the data of "
          "record; the session CSV is the health log.")


if __name__ == '__main__':
    main()
