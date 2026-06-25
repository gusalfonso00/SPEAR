// LSM6DSO32 over Wi-Fi UDP
// Same CSV format as serial version

#include <WiFi.h>
#include <WiFiUdp.h>
#include <Adafruit_LSM6DSO32.h>
#include "secrets.h"  // defines WIFI_SSID, WIFI_PASSWORD, TARGET_IP (gitignored)

const uint16_t TARGET_PORT = 4210;

Adafruit_LSM6DSO32 dso32;
WiFiUDP udp;
uint32_t seq = 0;

void setup() {
  Serial.begin(115200);
  delay(500);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Connected. ESP32 IP: ");
  Serial.println(WiFi.localIP());

  if (!dso32.begin_I2C()) {
    Serial.println("IMU not found!");
    while (1) delay(10);
  }

  // Full range for throw dynamics
  dso32.setAccelRange(LSM6DSO32_ACCEL_RANGE_32_G);
  dso32.setGyroRange(LSM6DS_GYRO_RANGE_2000_DPS);

  // ODR 208 Hz (sensor internal rate) > 100 Hz (scheduler poll rate).
  // Intentional oversampling: avoids beat-frequency aliasing if the two rates
  // were close or equal. Do not lower ODR to match poll rate.
  dso32.setAccelDataRate(LSM6DS_RATE_208_HZ);
  dso32.setGyroDataRate(LSM6DS_RATE_208_HZ);

  Serial.println("IMU ready, streaming at 100 Hz (ODR 208 Hz)...");
}

void loop() {
  static uint32_t next_us = 0;
  uint32_t now_us = micros();

  // Fixed-rate scheduler: fire every 10 ms = 100 Hz
  if ((int32_t)(now_us - next_us) < 0) return;
  next_us = now_us + 10000;

  sensors_event_t accel, gyro, temp;
  dso32.getEvent(&accel, &gyro, &temp);

  char buf[180];
  snprintf(buf, sizeof(buf),
           "%lu,%lu,%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
           seq++,
           millis(),
           temp.temperature,
           accel.acceleration.x, accel.acceleration.y, accel.acceleration.z,
           gyro.gyro.x, gyro.gyro.y, gyro.gyro.z);

  udp.beginPacket(TARGET_IP, TARGET_PORT);
  udp.print(buf);
  udp.endPacket();
}