// Live landing-form feedback. Camera -> BlazePose -> landing detector -> classifier -> cue -> log.
// Everything runs on this device; no video or image leaves the browser.
import { PoseLandmarker, FilesetResolver, DrawingUtils } from
  "./vendor/mediapipe/vision_bundle.mjs";
import { LandingDetector, frameAngles } from "./detector.js?v=4";
import { Classifier, personBox, cropCanvas } from "./classifier.js?v=3";

// local copies (vendor/) so the page can be cross-origin isolated and use several CPU threads
ort.env.wasm.wasmPaths = new URL("vendor/onnxruntime-web/", location.href).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const T = 8, BUFFER = 60, MAX_W = 640, TRUNK_HINT = 6.0, FLASH_MS = 2600, SIDE_YAW = 45;
// say = short spoken cue, so it ends before the next jump
const CUES = {
  correct_form: { title: "Good landing", text: "Knees over toes, trunk upright. Keep it up!", kind: "good", say: "Good landing" },
  knee_valgus: { title: "Knees caving in", text: "Push your knees out so they stay over your toes as you land.", kind: "bad", say: "Knees out" },
  trunk_lean: { title: "Leaning to the side", text: "Keep your chest centred over your hips when you land.", kind: "bad", say: "Chest centred" },
};
const BADGE = { correct_form: "Good", knee_valgus: "Knees in", trunk_lean: "Leaning" };
const ICONS = {
  good: '<svg width="24" height="24" viewBox="0 0 24 24"><path d="M20 6 9 17l-5-5"/></svg>',
  bad: '<svg width="24" height="24" viewBox="0 0 24 24"><path d="M12 9v4m0 4h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/></svg>',
  idle: '<svg width="24" height="24" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 8v4l3 2"/></svg>',
};

const $ = id => document.getElementById(id);
const video = $("video"), view = $("view"), ctx = view.getContext("2d");
let pose, classifier, detector, running = false, busy = false, downloaded = true;
let frameNo = 0, lastVideoTime = -1, buffer = [], log = [], fpsTimes = [];
let stats = newStats(), flashTimer = null, yawAvg = null, lastReady = "";
let lastLanding = null;   // {contactTime, cueAt, takeoff, entry} for next-take-off timing (FR7)

function newStats() { return { total: 0, correct_form: 0, knee_valgus: 0, trunk_lean: 0 }; }
function status(msg) { $("status").textContent = msg; }

// ---------------------------------------------------------------- preferences (this browser only)
function loadPrefs() {
  try { return JSON.parse(localStorage.getItem("lfc-prefs")) || {}; } catch { return {}; }
}
function savePrefs() {
  try {
    localStorage.setItem("lfc-prefs", JSON.stringify({
      voice: $("voice").checked, details: $("details").checked, model: $("model").value, classes: $("classes").value,
    }));
  } catch { /* storage blocked: settings just aren't remembered */ }
}
const prefs = loadPrefs();
if (prefs.voice === false) $("voice").checked = false;
if (prefs.details) $("details").checked = true;
if (prefs.model) $("model").value = prefs.model;
if (prefs.classes) $("classes").value = prefs.classes;
document.body.classList.toggle("show-details", $("details").checked);
$("cue-icon").innerHTML = ICONS.idle;

// ---------------------------------------------------------------- voice cues
// browser speech synthesis with an on-device voice only, so no text leaves the device
let voice = null;
function pickVoice() {
  const voices = speechSynthesis.getVoices().filter(v => v.localService);
  voice = voices.find(v => v.lang.startsWith("en") && v.default) || voices.find(v => v.lang.startsWith("en")) ||
          voices[0] || null;
  $("voice-note").textContent = voice ? voice.name : "no on-device voice found";
}
if ("speechSynthesis" in window) {
  pickVoice();
  speechSynthesis.onvoiceschanged = pickVoice;
} else {
  $("voice").disabled = true;
  $("voice-note").textContent = "not supported in this browser";
}
function speak(text) {
  if (!$("voice").checked || !voice || !text) return;
  speechSynthesis.cancel();                 // never queue up old cues
  const u = new SpeechSynthesisUtterance(text);
  u.voice = voice;
  u.rate = 1.15;
  speechSynthesis.speak(u);
}

// ---------------------------------------------------------------- stage states
function setStage(state) {
  $("stage").dataset.state = state;
  for (const s of ["idle", "loading", "error"]) $(`overlay-${s}`).hidden = state !== s;
  $("ready").hidden = state !== "running";
}
function loading(step, total, text) {
  $("loading-text").textContent = text;
  $("loading-bar").style.width = `${Math.round(step / total * 100)}%`;
}

async function loadPose() {
  const files = await FilesetResolver.forVisionTasks(
    new URL("vendor/mediapipe/wasm", location.href).href);
  return PoseLandmarker.createFromOptions(files, {
    baseOptions: { modelAssetPath: "models/pose_landmarker_full.task", delegate: "GPU" },
    runningMode: "VIDEO", numPoses: 1,
  });
}

async function loadClassifier() {
  classifier = await Classifier.load($("model").value, Number($("classes").value));
}

async function start() {
  if (log.length && !downloaded &&
      !confirm("Start a new session? The current history will be cleared. Download it first if you need it.")) return;
  resetSession();
  setStage("loading");
  try {
    if (!pose) { loading(0, 3, "Loading the pose model…"); pose = await loadPose(); }
    loading(1, 3, "Loading the landing model…");
    await loadClassifier();
    loading(2, 3, "Starting the camera…");
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    video.srcObject = stream;
    await video.play();
    loading(3, 3, "Ready");
    detector = new LandingDetector(onLanding, onTakeoff);
    running = true;
    lastReady = "";
    setStage("running");
    $("stop").disabled = false;
    status("Camera on. Stand where the green dot shows Ready, then jump.");
    requestAnimationFrame(loop);
  } catch (e) {
    showError(e);
  }
}

function showError(e) {
  const msg = {
    NotAllowedError: "Camera access was blocked. Click the camera icon in the address bar, allow the camera for this page, then press Try again.",
    NotFoundError: "No camera was found. Connect a camera, then press Try again.",
    OverconstrainedError: "No suitable camera was found. Connect a camera, then press Try again.",
    NotReadableError: "The camera is being used by another app. Close that app, then press Try again.",
  }[e.name] || `Something went wrong while starting (${e.message}). Check your internet connection for the first load, then press Try again.`;
  $("error-text").textContent = msg;
  setStage("error");
  status("Could not start.");
}

function endSession() {
  running = false;
  video.srcObject?.getTracks().forEach(t => t.stop());
  speechSynthesis?.cancel?.();
  $("stop").disabled = true;
  $("start-label").textContent = "Start new session";
  setStage("idle");
  status(log.length ? "Session ended." : "Session ended with no landings.");
  if (stats.total) showSummary();
}

function resetSession() {
  log = []; stats = newStats(); downloaded = true; buffer = [];
  $("log").innerHTML = '<li class="empty">No landings yet</li>';
  $("count").textContent = "";
  $("download").disabled = true;
  updateStats();
  const cue = $("cue");
  cue.className = "card cue idle";
  $("cue-icon").innerHTML = ICONS.idle;
  $("cue-title").textContent = "Waiting for your first landing";
  $("cue-text").textContent = "Face the camera, then jump and land on both feet.";
  $("cue-conf-wrap").hidden = true;
}

// ---------------------------------------------------------------- per-frame loop
function readiness(lm) {
  if (!lm) return ["warn", "No one in view: stand in front of the camera"];
  const vis = i => (lm[i].visibility ?? 1) >= 0.5;
  const inside = i => lm[i].x > 0.01 && lm[i].x < 0.99 && lm[i].y > 0.01 && lm[i].y < 0.99;
  if (![27, 28].every(i => vis(i) && inside(i))) return ["warn", "Step back so your feet are in view"];
  if (!inside(0)) return ["warn", "Step back so your head is in view"];
  if (![23, 24, 25, 26].every(vis)) return ["warn", "Make sure your legs are clearly visible"];
  // body turned away from the camera (same shoulder-depth check as the dataset screening)
  const yaw = Math.atan2(Math.abs(lm[11].z - lm[12].z), Math.abs(lm[11].x - lm[12].x)) * 180 / Math.PI;
  yawAvg = yawAvg === null ? yaw : 0.85 * yawAvg + 0.15 * yaw;
  if (yawAvg > SIDE_YAW) return ["warn", "Turn to face the camera"];
  return ["ok", "Ready: jump when you like"];
}

function loop() {
  if (!running) return;
  if (video.currentTime !== lastVideoTime && video.videoWidth) {
    lastVideoTime = video.currentTime;
    const now = performance.now();
    const res = pose.detectForVideo(video, now);
    const scale = Math.min(1, MAX_W / video.videoWidth);
    const w = Math.round(video.videoWidth * scale), h = Math.round(video.videoHeight * scale);
    if (view.width !== w) { view.width = w; view.height = h; }

    // keep a copy of the frame for the classifier (ring buffer)
    const slot = buffer.length < BUFFER ? document.createElement("canvas") : buffer.shift().canvas;
    slot.width = w; slot.height = h;
    slot.getContext("2d").drawImage(video, 0, 0, w, h);
    const lm = res.landmarks?.[0] || null;
    buffer.push({ frame: frameNo, time: now / 1000, lm, canvas: slot });

    ctx.drawImage(slot, 0, 0);
    const [kind, text] = readiness(lm);
    if (kind + text !== lastReady) {
      lastReady = kind + text;
      $("ready").className = "ready-chip " + kind;
      $("ready-text").textContent = text;
    }
    if (lm) {
      const du = new DrawingUtils(ctx);
      du.drawConnectors(lm, PoseLandmarker.POSE_CONNECTIONS, { color: "rgba(109,155,255,.9)", lineWidth: 3 });
      du.drawLandmarks(lm, { radius: 2, color: "#ffffff" });
      detector.push(frameNo, now / 1000, lm, w, h);
      const a = frameAngles(lm, w, h);
      $("live").textContent = a
        ? `FPPA L ${a.left.toFixed(0)}° R ${a.right.toFixed(0)}° · trunk ${a.trunk.toFixed(0)}°`
        : "legs not clearly visible";
    } else {
      $("live").textContent = "no person";
    }
    fpsTimes.push(now);
    while (fpsTimes.length && now - fpsTimes[0] > 1000) fpsTimes.shift();
    $("fps").textContent = `${fpsTimes.length} fps`;
    frameNo++;
  }
  requestAnimationFrame(loop);
}

// ---------------------------------------------------------------- one decision per landing
async function onLanding({ contact, flex, contactTime }) {
  if (busy) { addLog({ result: "skipped (busy)" }); return; }
  const win = buffer.filter(b => b.frame >= contact && b.frame <= flex && b.lm);
  if (win.length < 2) return;
  const landing = { contactTime, cueAt: null, takeoff: null, entry: null };
  lastLanding = landing;
  busy = true;
  try {
    // copy everything we need now, before the buffer moves on
    const idx = Array.from({ length: T }, (_, i) => Math.round(i * (win.length - 1) / (T - 1)));
    const w = win[0].canvas.width, h = win[0].canvas.height;
    const box = personBox(win.map(b => b.lm), w, h);
    const crops = idx.map(i => cropCanvas(win[i].canvas, box));
    const angles = win.map(b => frameAngles(b.lm, w, h)).filter(Boolean);
    const peakL = Math.max(...angles.map(a => a.left)), peakR = Math.max(...angles.map(a => a.right));
    const peakT = Math.max(...angles.map(a => a.trunk));
    const meanT = angles.reduce((s, a) => s + a.trunk, 0) / Math.max(1, angles.length);
    const summary = angles.length
      ? [peakL, peakR, Math.max(peakL, peakR), peakT, meanT, win[win.length - 1].time - win[0].time]
      : [0, 0, 0, 0, 0, 0];

    const t0 = performance.now();
    const probs = await classifier.predict(crops, summary);
    const modelMs = performance.now() - t0;
    const k = probs.indexOf(Math.max(...probs));
    const label = classifier.meta.classes[k];
    // 2-class model says only "fault": the angles decide which cue to give
    const shown = label === "fault" ? (summary[3] >= TRUNK_HINT ? "trunk_lean" : "knee_valgus") : label;
    const cue = label === "fault" ? { ...CUES[shown], title: "Landing fault" } : CUES[shown];
    const toCue = performance.now() - contactTime * 1000;

    showCue(cue, probs[k]);
    landing.cueAt = performance.now() / 1000;
    stats.total++;
    stats[shown]++;
    updateStats();
    landing.entry = addLog({ result: label, shown, kind: cue.kind, confidence: probs[k], fppa: summary[2],
                             trunk: summary[3], windowFrames: win.length, modelMs, toCueMs: toCue });
    if (landing.takeoff !== null) setTakeoff(landing);   // athlete took off before the cue appeared
    $("crop").getContext("2d").drawImage(crops[crops.length - 1], 0, 0, 112, 112);
  } catch (e) {
    addLog({ result: "error: " + e.message });
  } finally {
    busy = false;
  }
}

// next take-off after a landing: was the cue already on screen?
function onTakeoff(time) {
  if (!lastLanding || lastLanding.takeoff !== null) return;
  lastLanding.takeoff = time;
  if (lastLanding.entry) setTakeoff(lastLanding);
}
function setTakeoff(landing) {
  const e = landing.entry;
  e.nextTakeoffMs = (landing.takeoff - landing.contactTime) * 1000;
  e.cueBeforeTakeoff = landing.cueAt !== null && landing.cueAt < landing.takeoff;
  if (e.extraEl) e.extraEl.textContent = extraText(e);
}
function extraText(e) {
  let s = `FPPA ${e.fppa.toFixed(0)}° · trunk ${e.trunk.toFixed(0)}° · ${e.windowFrames} frames · ` +
    `model ${e.modelMs.toFixed(0)} ms · landing→cue ${e.toCueMs.toFixed(0)} ms`;
  if (e.nextTakeoffMs !== undefined) {
    s += ` · next take-off ${e.nextTakeoffMs.toFixed(0)} ms (cue ${e.cueBeforeTakeoff ? "before" : "after"})`;
  }
  return s;
}

function showCue(cue, conf) {
  speak(cue.say);
  $("cue").className = "card cue " + cue.kind;
  $("cue-icon").innerHTML = ICONS[cue.kind];
  $("cue-title").textContent = cue.title;
  $("cue-text").textContent = cue.text;
  $("cue-conf-wrap").hidden = false;
  $("cue-conf-bar").style.width = `${Math.round(conf * 100)}%`;
  $("cue-conf").textContent = `${Math.round(conf * 100)}% confident`;

  const f = $("flash");
  f.className = "flash " + cue.kind;
  $("flash-icon").innerHTML = ICONS[cue.kind];
  $("flash-title").textContent = cue.title;
  $("flash-text").textContent = cue.text;
  f.hidden = false;
  void f.offsetWidth;                 // restart the entry animation
  f.classList.add("show");
  clearTimeout(flashTimer);
  flashTimer = setTimeout(() => { f.hidden = true; }, FLASH_MS);

  const stage = $("stage");
  stage.classList.remove("glow-good", "glow-bad");
  void stage.offsetWidth;
  stage.classList.add("glow-" + cue.kind);
}

function updateStats() {
  $("s-total").textContent = stats.total;
  $("s-good").textContent = stats.correct_form;
  $("s-valgus").textContent = stats.knee_valgus;
  $("s-trunk").textContent = stats.trunk_lean;
}

function addLog(entry) {
  entry.n = log.length + 1;
  entry.time = new Date().toLocaleTimeString();
  entry.model = $("model").value + "/" + $("classes").value;
  log.push(entry);
  downloaded = false;
  $("download").disabled = false;

  const list = $("log");
  list.querySelector(".empty")?.remove();
  const li = document.createElement("li");
  const n = document.createElement("span");
  n.className = "n";
  n.textContent = entry.n;
  const badge = document.createElement("span");
  badge.className = "badge " + (entry.kind || "neutral");
  badge.textContent = entry.shown ? BADGE[entry.shown] + (entry.result === "fault" ? " (fault)" : "") : entry.result;
  const time = document.createElement("span");
  time.className = "time";
  time.textContent = entry.confidence !== undefined
    ? `${Math.round(entry.confidence * 100)}% · ${entry.time}` : entry.time;
  li.append(n, badge, time);
  if (entry.modelMs !== undefined) {
    const extra = document.createElement("span");
    extra.className = "extra details-only";
    extra.textContent = extraText(entry);
    entry.extraEl = extra;
    li.append(extra);
  }
  list.prepend(li);
  $("count").textContent = `${stats.total} landing${stats.total === 1 ? "" : "s"}`;
  return entry;
}

// ---------------------------------------------------------------- end of session
function showSummary() {
  const landings = log.filter(e => e.shown);
  const pct = Math.round(100 * stats.correct_form / stats.total);
  const cues = landings.map(e => e.toCueMs).sort((a, b) => a - b);
  const median = cues.length ? cues[Math.floor(cues.length / 2)] : null;
  const [topFault, topCount] = [["knee_valgus", stats.knee_valgus], ["trunk_lean", stats.trunk_lean]]
    .sort((a, b) => b[1] - a[1])[0];
  const tip = topCount
    ? `<strong>Focus for next time: ${CUES[topFault].title.toLowerCase()}</strong>${CUES[topFault].text}`
    : "<strong>Great session</strong>No faults detected. Keep the same technique.";
  $("summary-body").innerHTML = `
    <div class="score">
      <div class="ring" style="--p:${pct}"><span>${pct}%</span></div>
      <div><strong>${stats.correct_form} of ${stats.total}</strong> landings were rated good.</div>
    </div>
    <ul class="summary-list">
      <li><span>Landings</span><strong>${stats.total}</strong></li>
      <li><span>Good</span><strong>${stats.correct_form}</strong></li>
      <li><span>Knees caving in</span><strong>${stats.knee_valgus}</strong></li>
      <li><span>Leaning to the side</span><strong>${stats.trunk_lean}</strong></li>
      <li><span>Cues shown before the next take-off</span><strong>${(() => {
        const t = landings.filter(e => e.cueBeforeTakeoff !== undefined);
        return t.length ? `${t.filter(e => e.cueBeforeTakeoff).length} of ${t.length}` : "–";
      })()}</strong></li>
      <li><span>Median time from landing to cue</span><strong>${median === null ? "–" : (median / 1000).toFixed(2) + " s"}</strong></li>
    </ul>
    <div class="tip">${tip}</div>`;
  const dlg = $("summary");
  if (typeof dlg.showModal === "function") dlg.showModal();
}

function downloadLog() {
  const cols = ["n", "time", "model", "result", "shown", "confidence", "fppa", "trunk", "windowFrames", "modelMs", "toCueMs", "nextTakeoffMs", "cueBeforeTakeoff"];
  const csv = [cols.join(",")].concat(log.map(e => cols.map(c => e[c] ?? "").join(","))).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = `landing_log_${new Date().toISOString().slice(0, 19).replace(/:/g, "-")}.csv`;
  a.click();
  downloaded = true;
}

// ---------------------------------------------------------------- wiring
$("start").onclick = start;
$("retry").onclick = start;
$("stop").onclick = endSession;
$("download").onclick = downloadLog;
$("summary-download").onclick = downloadLog;
$("summary-close").onclick = () => $("summary").close();

const settingsBtn = $("settings-btn"), settings = $("settings");
function toggleSettings(open = settings.hidden) {
  settings.hidden = !open;
  settingsBtn.setAttribute("aria-expanded", String(open));
}
settingsBtn.onclick = e => { e.stopPropagation(); toggleSettings(); };
document.addEventListener("click", e => { if (!settings.hidden && !settings.contains(e.target)) toggleSettings(false); });
document.addEventListener("keydown", e => { if (e.key === "Escape" && !settings.hidden) toggleSettings(false); });

$("voice").onchange = savePrefs;
$("details").onchange = () => { document.body.classList.toggle("show-details", $("details").checked); savePrefs(); };
$("model").onchange = $("classes").onchange = async () => {
  savePrefs();
  if (!running) return;
  status("Switching model…");
  await loadClassifier();
  status("Model switched. Keep jumping.");
};
