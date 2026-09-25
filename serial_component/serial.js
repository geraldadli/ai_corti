/* Native Web Serial + the Streamlit v1 message protocol; no frontend dependencies. */
class CortiSession {
  constructor() { this.rows = []; this.header = null; }
  accept(line) {
    const parts = line.trim().split(",");
    if (parts[0] === "ERROR") throw Error(parts.slice(1).join(","));
    if (parts[0] === "CORTI") {
      if (this.header || parts.length !== 6 || parts[1] !== "1" || !["64", "100"].includes(parts[2]) || parts[3] !== "4" || !["0", "1"].includes(parts[4]))
        throw Error("Wrong sketch or device restarted. Reconnect Arduino.");
      const adcMax = Number(parts[5]);
      if (!Number.isInteger(adcMax) || adcMax < 255 || adcMax > 262143) throw Error("Invalid ADC range.");
      this.header = {protocol: 1, bvp_fs: Number(parts[2]), eda_fs: 4, calibrated: parts[4] === "1", adc_max: adcMax};
      return false;
    }
    if (!this.header) return false; // Ignore bootloader text until the explicit handshake.
    if (parts.length !== 5) return false; // A corrupt line leaves a sequence gap, never compressed time.
    const row = parts.map(value => value === "nan" || value === "" ? null : Number(value));
    if (row.some(value => value !== null && !Number.isFinite(value))) return false;
    const [seq, time] = row;
    if (!Number.isInteger(seq) || seq < 0 || seq >= 600 * this.header.bvp_fs || !Number.isInteger(time) || time < 0)
      throw Error("Invalid device sequence or session limit reached.");
    const previous = this.rows.at(-1);
    if ((!previous && seq !== 0) || (previous && (seq <= previous[0] || time <= previous[1])))
      throw Error("Device restarted or samples are out of order. Reconnect.");
    this.rows.push(row);
    return true;
  }
}

if (typeof module !== "undefined") module.exports = {CortiSession};
if (typeof document !== "undefined") {
  const connect = document.getElementById("connect"), stop = document.getElementById("stop");
  const status = document.getElementById("status"), readings = document.getElementById("readings");
  const canvas = document.getElementById("wave"), ctx = canvas.getContext("2d");
  let port, reader, timer, session, sessionId, revision = 0, pending = "", lastSample = 0;
  let active = false, closing = false, lastPublished = 0, lastParent = performance.now();
  const send = (type, data) => window.parent.postMessage({isStreamlitMessage: true, type, ...data}, "*");
  function publish(state, message) {
    send("streamlit:setComponentValue", {dataType: "json", value: {
      ...session?.header, session: sessionId, revision: ++revision, state, message,
      rows: state === "streaming" ? session.rows : []
    }});
    lastPublished = performance.now();
  }
  async function command(text) {
    if (!port?.writable) return;
    const writer = port.writable.getWriter();
    try { await writer.write(new TextEncoder().encode(text)); }
    finally { writer.releaseLock(); }
  }
  async function disconnect(message = "Disconnected. Connect again for a fresh session.") {
    if (closing) return;
    closing = true; active = false; clearInterval(timer);
    status.textContent = message;
    publish("disconnected", message); // Clear any previous prediction immediately.
    stop.disabled = true;
    try { await command("X"); } catch (_) { /* unplugged */ }
    try { await reader?.cancel(); } catch (_) { /* unplugged */ }
    // The read loop releases its lock before closing the port in its finally block.
  }
  function paint() {
    const rows = session.rows, latest = rows.at(-1);
    if (!latest) return;
    const fs = session.header.bvp_fs;
    const eda = [...rows.slice(-fs / 4)].reverse().find(row => row[4] !== null);
    status.textContent = `Receiving · ${((latest[0] + 1) / fs).toFixed(0)} seconds`;
    readings.textContent = `Pulse: ${latest[2] ?? "—"} ADC · GSR: ${eda?.[4] ?? "—"} ADC` +
      (session.header.calibrated ? ` · ${eda?.[3]?.toFixed(2) ?? "—"} µS` : " · calibration needed");
    const points = rows.slice(-5 * fs).map(row => row[2]);
    const finite = points.filter(x => x !== null);
    const lo = Math.min(...finite), span = Math.max(1, Math.max(...finite) - lo);
    ctx.clearRect(0, 0, 600, 70); ctx.beginPath(); ctx.strokeStyle = "#23796a"; ctx.lineWidth = 2;
    points.forEach((v, i) => { if (v !== null) ctx.lineTo(i * 600 / (5 * fs), 62 - (v - lo) / span * 54); });
    ctx.stroke();
  }
  connect.onclick = async () => {
    connect.disabled = true; closing = false;
    session = new CortiSession(); sessionId = crypto.randomUUID(); revision = 0; pending = "";
    ctx.clearRect(0, 0, 600, 70); readings.textContent = "Pulse + skin response";
    publish("connecting", "Connecting Arduino…");
    try {
      port = await navigator.serial.requestPort();
      await port.open({baudRate: 115200, bufferSize: 65536});
      status.textContent = "Starting Arduino…";
      await new Promise(resolve => setTimeout(resolve, 2000)); // USB open can reset an Uno.
      await command("S");
      reader = port.readable.getReader(); active = true; stop.disabled = false;
      lastSample = performance.now(); lastPublished = lastSample;
      const decoder = new TextDecoder();
      timer = setInterval(() => {
        if (!active) return;
        const now = performance.now();
        if (now - lastParent > 15000) { disconnect("Website connection lost. Reconnect to start again."); return; }
        if (now - lastSample > 5000) { disconnect("No sensor data. Check USB and upload the Corti sketch."); return; }
        paint();
        if (session.rows.length && now - lastPublished >= 5000) publish("streaming", "Receiving signals");
      }, 250);
      while (active) {
        const {value, done} = await reader.read();
        if (done) break;
        pending += decoder.decode(value, {stream: true});
        const lines = pending.split("\n"); pending = lines.pop();
        if (pending.length > 256) throw Error("Invalid serial data. Check the sketch and baud rate.");
        for (const line of lines) {
          if (session.accept(line)) lastSample = performance.now();
          if (session.rows.at(-1)?.[0] >= 600 * session.header?.bvp_fs - 1) { await disconnect("Ten-minute session complete. Reconnect to start another."); break; }
        }
      }
      if (active) await disconnect("Device disconnected. Reconnect to start again.");
    } catch (error) {
      await disconnect(error.name === "NotFoundError" ? "Connection cancelled." : `Connection stopped: ${error.message}`);
    } finally {
      active = false; clearInterval(timer);
      try { reader?.releaseLock(); } catch (_) { /* already released */ }
      reader = null;
      try { await port?.close(); } catch (_) { /* unplugged or not opened */ }
      port = null; connect.disabled = false; stop.disabled = true; closing = false;
    }
  };
  stop.onclick = () => disconnect();
  window.addEventListener("message", event => {
    if (event.source === window.parent && event.data?.type === "streamlit:render") {
      lastParent = performance.now();
      send("streamlit:setFrameHeight", {height: 190});
    }
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && active) disconnect("Capture paused because the tab was hidden. Reconnect to start again.");
  });
  window.addEventListener("pagehide", () => { if (active) disconnect(); });
  if (!window.isSecureContext || !("serial" in navigator) ||
      (document.featurePolicy && !document.featurePolicy.allowsFeature("serial"))) {
    connect.disabled = true;
    status.textContent = "Open this app directly in desktop Chrome or Edge over HTTPS (or localhost) to connect USB.";
  }
  send("streamlit:componentReady", {apiVersion: 1});
  send("streamlit:setFrameHeight", {height: 190});
}
