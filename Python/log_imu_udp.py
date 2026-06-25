import socket
import csv
import os
from datetime import datetime

LISTEN_PORT = 4210
DURATION_SEC = 120

# Build path relative to where this script lives
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "Data Logs")
os.makedirs(data_dir, exist_ok=True)  # creates folder if it doesn't exist
filename = os.path.join(data_dir, f"imu_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', LISTEN_PORT))
sock.settimeout(2.0)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)

with open(filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['seq', 'ms', 'temp_C', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'])

    start = datetime.now()
    print(f"Listening on UDP port {LISTEN_PORT}, logging to {filename}...")
    print("Ctrl+C to stop early.")

    received = 0
    last_seq = None
    drops = 0

    try:
        while (datetime.now() - start).total_seconds() < DURATION_SEC:
            try:
                data, addr = sock.recvfrom(256)
                line = data.decode('utf-8', errors='ignore').strip()
                parts = line.split(',')
                if len(parts) == 9:
                    try:
                        seq = int(parts[0])
                        [float(p) for p in parts[1:]]
                        writer.writerow(parts)
                        received += 1

                        if last_seq is not None:
                            gap = seq - last_seq - 1
                            if gap > 0:
                                drops += gap
                        last_seq = seq

                        if received % 500 == 0:
                            print(f"  {received} packets, {drops} dropped ({100*drops/(received+drops):.1f}%)")
                    except ValueError:
                        pass
            except socket.timeout:
                print("  (no data — ESP32 connected?)")
    except KeyboardInterrupt:
        print("Stopped.")

sock.close()
total = received + drops
loss_pct = 100 * drops / total if total else 0
print(f"\nSaved: {filename}")
print(f"Received: {received}, Dropped: {drops}, Loss: {loss_pct:.2f}%")