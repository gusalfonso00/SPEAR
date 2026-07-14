// Onboard throw capture: RAM ring buffer, FREEZE-to-flash, TCP dump.
// See Documentation/BUFFER_DUMP.md for the protocol and sizing math.

#pragma once
#include <Arduino.h>

// One buffered IMU sample. Raw sensor words, not floats: 18 bytes instead
// of 36, and the exact ADC output survives to disk with no rounding. The
// host decoder applies the same scale factors the Adafruit library uses,
// so decoded CSVs match the UDP stream values.
struct __attribute__((packed)) BufSample {
  uint32_t t_ms;   // millis() at the sensor read
  uint16_t seq;    // low 16 bits of the UDP stream's sequence counter
  int16_t ax, ay, az;   // raw accelerometer words (+/-32 g full scale)
  int16_t gx, gy, gz;   // raw gyro words (+/-2000 dps full scale)
};
static_assert(sizeof(BufSample) == 18, "BufSample must pack to exactly 18 bytes");

// Allocate the 108 KB ring from the heap. MUST be the first thing setup()
// does, before Serial is even a nicety and strictly before WiFi.begin():
// the heap is unfragmented at that point, so the allocation is
// deterministic. Halts with a serial message if it fails. (A static array
// does not link: it overflows the ESP32's static-data segment, which the
// Wi-Fi stack's own statics share.)
void throwBufferAlloc();

// Mount LittleFS (formats on first boot), load the persistent throw
// counter, open the command UDP port and TCP dump server. Call once in
// setup(), after Wi-Fi is up.
void throwBufferInit();

// Copy one sample into the ring. Called at 100 Hz from the sampling loop;
// a struct copy and an index increment, nothing that can block.
void throwBufferWrite(const BufSample &s);

// Poll the command UDP port (FREEZE) and the TCP dump server (LIST / GET /
// CLEAR / QUIT). Call every pass through loop(). Normally returns in
// microseconds; during a FREEZE flush or a TCP file transfer it blocks for
// seconds, which stalls sampling and streaming. That is acceptable by
// design: a freeze means the event is already in the buffer, and dumps
// happen between throws.
void throwBufferService();
