// AI Corti: ESP32 DevKit + MAX30102 + Grove GSR, USB 115200 baud.
// Install "SparkFun MAX3010x Pulse and Proximity Sensor Library" in Arduino IDE.
// MAX30105 is the library's class name; it also supports MAX30102 (not MAX30100).
#include <Arduino.h>
#include <Wire.h>
#include <MAX30105.h>
#include <math.h>

// Defaults for a classic ESP32 DevKit. Check/change these for your actual board.
const int SDA_PIN = 21, SCL_PIN = 22, GSR_PIN = 34;
const float GSR_SUPPLY_MV = 3300.0f; // Measure actual Grove supply for calibration.
const float GSR_CALIBRATION_10BIT = NAN; // No-contact value; see LIVE_CAPTURE.md.
const bool GSR_CALIBRATION_VERIFIED = false; // Verify with known resistors first.
const byte LED_CURRENT = 0x1F; // Adjust for signal strength; keep fixed during sessions.
const uint32_t MIN_IR_CONTACT = 10000; // Initial heuristic; tune with your module.
const uint32_t PULSE_MAX = 262143; // 18-bit MAX30102 ADC at pulse width 411 us.
MAX30105 pulseSensor;
bool sensorReady = false, capturing = false;
uint32_t sequence = 0, originUs = 0, lastCheckUs = 0;

bool calibrated() {
  return GSR_CALIBRATION_VERIFIED && isfinite(GSR_CALIBRATION_10BIT)
         && GSR_CALIBRATION_10BIT > 0 && GSR_CALIBRATION_10BIT < 1023;
}

float gsrMicrosiemens(float reading) {
  // Seeed's equation, using equivalent 10-bit counts at the supply reference.
  // https://wiki.seeedstudio.com/Grove-GSR_Sensor/
  if (!calibrated() || reading <= 0 || reading >= GSR_CALIBRATION_10BIT) return NAN;
  const float resistance = ((1024.0f + 2.0f * reading) * 10000.0f)
                           / (GSR_CALIBRATION_10BIT - reading);
  return resistance > 0 ? 1000000.0f / resistance : NAN;
}

void setup() {
  Serial.begin(115200);
  Wire.begin(SDA_PIN, SCL_PIN);
  analogReadResolution(12);
  analogSetPinAttenuation(GSR_PIN, ADC_11db);
  sensorReady = pulseSensor.begin(Wire, I2C_SPEED_FAST);
  if (sensorReady) {
    // Native 100 Hz, no FIFO averaging, red + IR, 18-bit ADC. Server resamples to 64 Hz.
    pulseSensor.setup(LED_CURRENT, 1, 2, 100, 411, 4096);
    pulseSensor.disableFIFORollover();
    pulseSensor.shutDown();
  }
}

void loop() {
  while (Serial.available()) {
    const char command = Serial.read();
    if (command == 'X') { capturing = false; if (sensorReady) pulseSensor.shutDown(); }
    if (command == 'S') {
      if (!sensorReady) { Serial.println(F("ERROR,MAX30102 not found; check I2C wiring")); continue; }
      pulseSensor.shutDown(); pulseSensor.clearFIFO();
      while (pulseSensor.available()) pulseSensor.nextSample();
      Serial.print(F("CORTI,1,100,4,")); Serial.print(calibrated() ? 1 : 0);
      Serial.print(','); Serial.println(PULSE_MAX);
      sequence = 0; originUs = 0; lastCheckUs = micros(); capturing = true;
      pulseSensor.wakeUp();
    }
  }
  if (!capturing) { delay(1); return; }
  // Library FIFO ring is small. Stop on a service stall rather than silently losing timing.
  const uint32_t now = micros();
  if (now - lastCheckUs > 20000) {
    Serial.println(F("ERROR,Sensor service stalled; reconnect"));
    capturing = false; pulseSensor.shutDown(); return;
  }
  lastCheckUs = now;
  pulseSensor.check();
  while (pulseSensor.available()) {
    const uint32_t ir = pulseSensor.getFIFOIR();
    pulseSensor.nextSample();
    const uint32_t stamp = micros();
    if (sequence == 0) originUs = stamp;
    float gsr = NAN, eda = NAN;
    if (sequence % 25 == 0) {
      const uint32_t mv = analogReadMilliVolts(GSR_PIN);
      if (mv > 150 && mv < 3050) { // Classic ESP32 usable ADC range at 11 dB.
        gsr = mv * 1024.0f / GSR_SUPPLY_MV;
        eda = gsrMicrosiemens(gsr);
      }
    }
    Serial.print(sequence); Serial.print(','); Serial.print(stamp - originUs); Serial.print(',');
    if (ir >= MIN_IR_CONTACT && ir < PULSE_MAX) Serial.print(ir); else Serial.print(F("nan"));
    Serial.print(',');
    if (isfinite(eda) && eda >= 0 && eda <= 100) Serial.print(eda, 5); else Serial.print(F("nan"));
    Serial.print(',');
    if (isfinite(gsr)) Serial.print(gsr, 3); else Serial.print(F("nan"));
    Serial.println();
    if (++sequence >= 60000) { capturing = false; pulseSensor.shutDown(); break; }
  }
}
