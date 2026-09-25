# AI Corti · Live ESP32 capture

**ESP32 + MAX30102 + Grove GSR → USB → Chrome/Edge → AI Corti → Stress / No stress.**

Arduino IDE uploads the firmware. After upload, the website reads the USB serial port directly; the IDE is not a relay. Close Serial Monitor and Serial Plotter before connecting the website.

## 1. Wire the sensors

The sketch defaults below are for a **classic ESP32 DevKit**. Change the constants for your actual board and wiring, especially if using an ESP32-C3/S3.

| Sensor connection | ESP32 default |
|---|---|
| MAX30102 SDA | GPIO 21 |
| MAX30102 SCL | GPIO 22 |
| MAX30102 breakout power | 3.3 V, if supported by your breakout |
| MAX30102 GND | GND |
| Grove GSR SIG (yellow) | GPIO 34 / ADC1 |
| Grove GSR VCC (red) | 3.3 V |
| Grove GSR GND (black) | GND |

Leave the Grove's unused wire disconnected. The MAX30102 **breakout board**, not a bare sensor IC, must support your chosen supply and 3.3 V I²C levels. Do not feed a 5 V signal into ESP32 GPIO. Connect the ESP32 to the PC using a USB data cable.

## 2. Upload the sketch

1. In Arduino IDE, install **esp32 by Espressif Systems** through Boards Manager.
2. Install **SparkFun MAX3010x Pulse and Proximity Sensor Library** through Library Manager.
3. Open `arduino/corti_capture/corti_capture.ino`, select your board and USB port, then upload.
4. Close Serial Monitor / Plotter.
5. Open AI Corti directly in desktop Chrome or Edge. Select **Live Arduino → Connect Arduino**, then select the ESP32 port.

The library calls its class `MAX30105`, but it supports the MAX30102. The firmware reads the raw infrared waveform, **not BPM or SpO₂**. Without GSR calibration, the page shows live readings but does not produce a stress label.

## 3. Calibrate Grove GSR once

The model expects EDA in **microsiemens (µS)**. Raw ADC values cannot be passed as µS. Grove's analog output encodes resistance and depends on its potentiometer setting.

1. Leave electrodes off the skin. Follow [Seeed's no-contact calibration procedure](https://wiki.seeedstudio.com/Grove-GSR_Sensor/), using the live page's **GSR ADC** number. This is an equivalent 10-bit value calculated from the ESP32's calibrated millivolt reading and the Grove supply voltage.
2. Enter that number as `GSR_CALIBRATION_10BIT` in the sketch. Measure/update `GSR_SUPPLY_MV` if it differs from 3300 mV. Do not move the potentiometer afterward.
3. Verify the conversion with known resistors across the disconnected electrodes (for example, 100 kΩ should give approximately 10 µS; 1 MΩ approximately 1 µS). Calculate using the live GSR reading and the equation below before enabling inference.
4. Only after verification, set `GSR_CALIBRATION_VERIFIED = true` and upload again. If the denominator is zero/negative or the resistor checks fail, keep this false and check the Grove revision, potentiometer, wiring, and conversion. Do not take an absolute value or invent an offset to force an output.

The sketch implements Seeed's published formula, with `x` the equivalent 10-bit reading and `c` the no-contact calibration:

```text
R_ohms = (1024 + 2*x) * 10000 / (c - x)
EDA_µS = 1,000,000 / R_ohms
```

ESP32 ADC calibration does not itself calibrate the Grove circuit. The code rejects voltages outside 150–3050 mV, based on the classic ESP32 ADC range at 11 dB attenuation; other ESP32 variants need their own limits. Check `MIN_IR_CONTACT` and `LED_CURRENT` for a stable, unclipped MAX30102 waveform; the contact threshold is a starting heuristic, not a calibrated quality measure.

## 4. Run a live check-in

Wear the sensors, keep still, and leave the tab visible. The first result normally arrives after **about 45 seconds**, then updates approximately every **5 seconds** using the latest completed **30-second window**. Poor contact, invalid units, timing errors, stale data, and disconnects suppress the current label.

Select **Disconnect** when finished. Switching away from the tab stops capture to avoid browser background throttling. Reconnecting starts a new session and warmup. Sessions stop at **10 minutes** to bound memory, upload traffic, and preprocessing time.

The reference implementation sends the complete session approximately every five seconds and reuses the existing model's full-history preprocessing. Longer sessions should use an incremental transport and stateful filters; simply dropping old history changes the EDA tonic/filter state.

## What reaches the model

- MAX30102 samples IR at its supported **100 Hz** rate. Python performs anti-aliasing polyphase resampling to the model's **64 Hz**; it withholds 0.25 seconds to avoid using artificial future padding.
- Grove is sampled every 25 pulse samples: **4 Hz**, aligned to the same session. The sketch sends calibrated µS and the raw equivalent ADC reading.
- Sequence numbers preserve missing samples as NaN. Actual ESP32 timestamps are checked for order, timing jitter, and sampling-rate drift; timing failures require a fresh connection.
- The unchanged `stress_inference.py` filters, applies warmup and quality checks, constructs the existing BVP + four EDA channels, and runs the saved model.
- **No stress = P(Baseline) + P(Amusement)**. Compare that with **P(Stress)**. Live mode also rejects windows that trigger the predictor's pulse-quality warning.

The model was trained on Empatica E4 wrist signals. MAX30102 infrared finger PPG and Grove finger GSR are different sensors/sites. A successful live prediction does not establish accuracy on this hardware; collect labeled recordings and evaluate before claiming performance.

## Deployment and troubleshooting

Deploy `live_capture.py`, the complete `serial_component/` folder, and the updated `streamlit_app.py` alongside the existing model bundle. Python requirements are unchanged; no pyserial, local bridge, API key, or new server is needed. Include the sketch and this guide for setup.

USB access uses the visitor's browser, so it can work with a hosted HTTPS app or localhost. Open the app directly in Chrome/Edge, not inside a third-party iframe. An embedding site's permissions policy or managed-browser policy can block serial access. A blocked browser shows an explanatory message.

Readings travel to the Streamlit server for inference and remain in session memory; this feature writes no recordings to disk. Disconnecting clears the visible live result. Reloading clears the browser buffer. Each visitor has a separate session; the model resource is shared.

| Symptom | Check |
|---|---|
| Port will not open | Close IDE Serial Monitor/Plotter and any other serial application. |
| MAX30102 not found | I²C pins, power, ground and sensor model. |
| Calibration needed | Complete the Grove calibration; never substitute ADC counts for µS. |
| Signal not ready / weak pulse | Finger contact, movement, LED current, GSR calibration and electrode contact. |
| Timing error / service stalled | Reconnect; remove extra firmware delays/debug prints. The sketch stops rather than silently losing FIFO samples. |
| Live data paused | Keep the tab visible; check USB and network, then reconnect. |

Software checks: `python check_ui.py`, `python check_live.py`, and `node serial_component/check_serial.cjs`. These use synthetic data; they do not replace a physical sensor test.

Verified locally: the sketch compiles for `esp32:esp32:esp32` with Espressif core 3.3.3 and SparkFun MAX3010x 1.1.2. Protocol, resampling, real-model inference, calibration gating, stale-result removal, and existing recording UI checks passed. A synthetic ten-minute session took approximately 0.25 seconds to validate, preprocess and predict after model load on the development PC; hosted startup/network time will differ. Physical wiring, calibration and sensor accuracy have not been tested.

References: [Web Serial](https://developer.chrome.com/docs/capabilities/serial), [SparkFun MAX3010x library](https://github.com/sparkfun/SparkFun_MAX3010x_Sensor_Library), [ESP32 ADC API](https://docs.espressif.com/projects/arduino-esp32/en/latest/api/adc.html), [Grove GSR](https://wiki.seeedstudio.com/Grove-GSR_Sensor/).
