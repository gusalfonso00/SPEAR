// Minimal UDP echo test to verify laptop-to-ESP32 packets arrive through the hotspot
#include <WiFi.h>
// UDP object for receiving test packets
#include <WiFiUdp.h>
// Pull SSID and password from the same gitignored secrets file the main firmware uses
#include "secrets.h"

// UDP listener instance
WiFiUDP udp;
// Buffer for incoming packet contents
char buf[64];

void setup() {
  // Serial for printing the IP and received packets
  Serial.begin(115200);
  // Join the same hotspot the main firmware uses
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  // Block until associated
  while (WiFi.status() != WL_CONNECTED) { delay(200); }
  // Disable modem sleep, same as main firmware
  WiFi.setSleep(false);
  // Print the address the laptop should send to
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
  // Listen on the same port number planned for the command channel
  udp.begin(9001);
}

void loop() {
  // Check for an incoming packet
  int n = udp.parsePacket();
  // If one arrived, read and print it
  if (n > 0) {
    // Read up to buffer size minus null terminator
    int len = udp.read(buf, 63);
    // Terminate the string
    buf[len] = 0;
    // Show what arrived and from where
    Serial.printf("Got %d bytes from %s: %s\n", n, udp.remoteIP().toString().c_str(), buf);
  }
}