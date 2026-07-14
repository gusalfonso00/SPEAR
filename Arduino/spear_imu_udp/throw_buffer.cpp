// Onboard throw capture implementation. Protocol and design rationale in
// Documentation/BUFFER_DUMP.md.

#include "throw_buffer.h"
#include <WiFi.h>
#include <WiFiUdp.h>
#include <LittleFS.h>
#include <esp_heap_caps.h>

// --- Ports (also documented in BUFFER_DUMP.md) ---
// Command traffic is on its own UDP port so it never mixes with the
// outbound 4210 sample stream. The TCP dump gets a third port.
static const uint16_t CMD_UDP_PORT  = 4211;   // host -> ESP32: FREEZE
static const uint16_t DUMP_TCP_PORT = 4212;   // LIST / GET / CLEAR / QUIT

// --- Ring buffer ---
// 6000 samples = 60 s at 100 Hz = 108,000 bytes.
// Allocated ONCE from the heap at boot (throwBufferAlloc), not static and
// not stack. A static array was tried first and does not link on this
// chip: the ESP32's static-data segment (dram0_0_seg, ~160 KB) also holds
// the Arduino core and Wi-Fi stack statics, and 108 KB overflows it by
// ~32 KB. The heap can reach DRAM the static segment cannot. To keep the
// original no-malloc intent (no silent runtime allocation failure), the
// allocation happens at the very top of setup(), BEFORE Wi-Fi initializes
// (heap is empty and unfragmented there), is hard-checked with a halt on
// failure, and is never freed or resized afterward.
static const size_t RING_N = 6000;
static BufSample *ring = nullptr;
static size_t ring_head = 0;      // next write slot
static bool   ring_full = false;  // true once the ring has wrapped

// --- Freeze bookkeeping ---
static uint32_t throw_counter = 0;        // persisted in /throw_counter.txt
static char     last_ack[64]  = "";       // last ACK line, for duplicate FREEZEs
static uint32_t last_freeze_done_ms = 0;  // when the last flush finished

static WiFiUDP    cmdUdp;
static WiFiServer dumpServer(DUMP_TCP_PORT);
static WiFiClient dumpClient;

static const char *COUNTER_FILE = "/throw_counter.txt";


void throwBufferAlloc() {
  // Must run before WiFi.begin(): the heap is one big unfragmented block
  // here, so this either succeeds deterministically or the board is the
  // wrong part. There is no recovering from a missing ring buffer, so a
  // failure halts loudly rather than flying without the data of record.
  ring = (BufSample *)heap_caps_malloc(RING_N * sizeof(BufSample),
                                       MALLOC_CAP_8BIT);
  if (ring == nullptr) {
    Serial.printf("FATAL: ring buffer alloc failed (%u bytes). Halting.\n",
                  (unsigned)(RING_N * sizeof(BufSample)));
    while (true) delay(1000);
  }
  Serial.printf("Ring buffer: %u samples (%u bytes) allocated pre-WiFi, "
                "free heap now %u\n",
                (unsigned)RING_N, (unsigned)(RING_N * sizeof(BufSample)),
                ESP.getFreeHeap());
}


void throwBufferInit() {
  // Mount without auto-format first so a format is a reported event, not a
  // silent one. First-ever mount on a fresh chip WILL fail and format here;
  // that takes several seconds, which is why this lives in setup() and is
  // never done lazily from the sampling loop.
  bool fs_ok = LittleFS.begin(false);
  if (fs_ok) {
    Serial.println("LittleFS mounted");
  } else {
    Serial.println("LittleFS mount failed, formatting (first boot takes a few s)...");
    LittleFS.format();
    fs_ok = LittleFS.begin(false);
    Serial.println(fs_ok ? "LittleFS formatted and mounted"
                         : "LittleFS UNUSABLE: freezes will reply ERR");
  }
  // Even with no filesystem, the command and dump ports still start below:
  // a FREEZE then fails loudly with an ERR reply on the host instead of
  // disappearing into a dead port.

  // Persistent throw counter: read at boot, incremented and written back on
  // every freeze, so numbers are never reused across reboots.
  if (fs_ok) {
    File cf = LittleFS.open(COUNTER_FILE, "r");
    if (cf) {
      throw_counter = cf.parseInt();
      cf.close();
    }
    Serial.printf("Throw counter at boot: %lu\n", (unsigned long)throw_counter);
  }

  cmdUdp.begin(CMD_UDP_PORT);
  dumpServer.begin();
  Serial.printf("Command UDP on %u, dump TCP on %u\n", CMD_UDP_PORT, DUMP_TCP_PORT);
}


void throwBufferWrite(const BufSample &s) {
  ring[ring_head] = s;              // 18-byte struct copy, nothing else
  if (++ring_head == RING_N) {
    ring_head = 0;
    ring_full = true;
  }
}


// Write the ring to /throw_NNN.bin, oldest sample first. Blocks for the
// duration of the flash write (seconds). Safe without any locking: sampling
// and this flush both run cooperatively in loop(), so no new sample can be
// written while this function has the CPU. That is the "simpler correct
// option" for snapshotting - the alternative (double buffering) buys
// nothing here because the stall is acceptable anyway.
static bool freezeToFlash() {
  throw_counter++;

  // Persist the counter BEFORE writing the data file: if power dies mid
  // flush, the number is burned and never reused.
  File cf = LittleFS.open(COUNTER_FILE, "w");
  if (!cf) {
    Serial.println("FREEZE: cannot write counter file");
    return false;
  }
  cf.printf("%lu", (unsigned long)throw_counter);
  cf.close();

  char path[24];
  snprintf(path, sizeof(path), "/throw_%03lu.bin", (unsigned long)throw_counter);

  File f = LittleFS.open(path, "w");
  if (!f) {
    Serial.printf("FREEZE: cannot open %s\n", path);
    return false;
  }

  // Oldest-first: a wrapped ring is two contiguous regions
  size_t written = 0;
  if (ring_full) {
    written += f.write((const uint8_t *)&ring[ring_head],
                       (RING_N - ring_head) * sizeof(BufSample));
    written += f.write((const uint8_t *)&ring[0], ring_head * sizeof(BufSample));
  } else {
    written += f.write((const uint8_t *)&ring[0], ring_head * sizeof(BufSample));
  }
  f.close();

  snprintf(last_ack, sizeof(last_ack), "ACK FREEZE throw_%03lu.bin %u\n",
           (unsigned long)throw_counter, (unsigned)written);
  Serial.printf("FREEZE done: %s", last_ack);
  return true;
}


static void handleCommandUdp() {
  int len = cmdUdp.parsePacket();
  if (len <= 0) return;

  char buf[32];
  int n = cmdUdp.read(buf, sizeof(buf) - 1);
  buf[n > 0 ? n : 0] = '\0';

  // Save the sender before the (long) flush; remote info would survive
  // anyway until the next parsePacket, but being explicit costs nothing.
  IPAddress rip = cmdUdp.remoteIP();
  uint16_t rport = cmdUdp.remotePort();

  if (strncmp(buf, "FREEZE", 6) != 0) return;

  // The host retries FREEZE at 10 Hz until acked, so several duplicates
  // queue up while the flush is running. Any FREEZE within 2 s of a
  // completed flush is the same request: do not create a new file, but DO
  // resend the same ACK so the host stops retrying.
  if (last_ack[0] != '\0' && millis() - last_freeze_done_ms < 2000) {
    cmdUdp.beginPacket(rip, rport);
    cmdUdp.print(last_ack);
    cmdUdp.endPacket();
    return;
  }

  bool ok = freezeToFlash();
  last_freeze_done_ms = millis();

  cmdUdp.beginPacket(rip, rport);
  if (ok) {
    cmdUdp.print(last_ack);
  } else {
    // Not part of the nominal protocol: the host treats anything that is
    // not "ACK FREEZE" as a retry-until-timeout, and this line makes the
    // failure diagnosable on the host side instead of silent.
    cmdUdp.print("ERR FREEZE flash write failed\n");
  }
  cmdUdp.endPacket();
}


// True for names like "throw_007.bin", with or without a leading slash
// (ESP32 core versions differ on what File::name() returns).
static bool isThrowFile(const char *name) {
  if (name[0] == '/') name++;
  return strncmp(name, "throw_", 6) == 0 && strstr(name, ".bin") != nullptr;
}


static void handleTcpLine(const String &line) {
  if (line == "LIST") {
    File root = LittleFS.open("/");
    File f = root.openNextFile();
    while (f) {
      if (!f.isDirectory() && isThrowFile(f.name())) {
        const char *nm = f.name();
        if (nm[0] == '/') nm++;
        dumpClient.printf("%s %u\n", nm, (unsigned)f.size());
      }
      f = root.openNextFile();
    }
    dumpClient.print("END\n");

  } else if (line.startsWith("GET ")) {
    String fname = line.substring(4);
    fname.trim();
    String path = fname.startsWith("/") ? fname : "/" + fname;
    File f = LittleFS.open(path, "r");
    if (!f || f.isDirectory()) {
      dumpClient.print("ERR NOFILE\n");   // host only GETs names from LIST
      return;
    }
    dumpClient.printf("SIZE %u\n", (unsigned)f.size());
    // Stream in chunks. This blocks loop() for the whole transfer, pausing
    // sampling and streaming: acceptable, dumps happen between throws and
    // the data of record is already on flash.
    uint8_t chunk[1024];
    while (f.available()) {
      size_t n = f.read(chunk, sizeof(chunk));
      dumpClient.write(chunk, n);
    }
    f.close();
    // Connection stays open for further commands

  } else if (line == "CLEAR") {
    // Deletes throw files only, never the counter file: numbering must
    // keep climbing across clears so old local copies are never shadowed.
    int cleared = 0;
    File root = LittleFS.open("/");
    File f = root.openNextFile();
    String doomed[64];
    while (f && cleared < 64) {
      if (!f.isDirectory() && isThrowFile(f.name())) {
        const char *nm = f.name();
        doomed[cleared++] = (nm[0] == '/') ? String(nm) : "/" + String(nm);
      }
      f = root.openNextFile();
    }
    for (int i = 0; i < cleared; i++) LittleFS.remove(doomed[i]);
    dumpClient.printf("CLEARED %d\n", cleared);

  } else if (line == "QUIT") {
    dumpClient.stop();
  }
}


static void handleTcpServer() {
  // Adopt a new client when idle. One client at a time is plenty: there is
  // exactly one ground station.
  if (!dumpClient || !dumpClient.connected()) {
    WiFiClient c = dumpServer.available();
    if (c) {
      dumpClient = c;
      dumpClient.setTimeout(2000);
      Serial.printf("Dump client connected: %s\n",
                    dumpClient.remoteIP().toString().c_str());
    }
    return;
  }
  if (dumpClient.available()) {
    String line = dumpClient.readStringUntil('\n');
    line.trim();
    if (line.length()) handleTcpLine(line);
  }
}


void throwBufferService() {
  handleCommandUdp();
  handleTcpServer();
}
