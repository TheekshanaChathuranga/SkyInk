// AirScript Browser Data Collector
const videoElement = document.getElementById("webcam-video");
const canvasElement = document.getElementById("drawing-canvas");
const canvasCtx = canvasElement.getContext("2d");

const penStatusPill = document.getElementById("pen-status-pill");
const penDot = document.getElementById("pen-dot");
const penStatusText = document.getElementById("pen-status-text");
const fpsDisplay = document.getElementById("fps-display");

const targetCharElem = document.getElementById("target-char");
const userInput = document.getElementById("user-input");
const statTotal = document.getElementById("stat-total");
const statTargetCount = document.getElementById("stat-target-count");
const statStrokes = document.getElementById("stat-strokes");
const statPoints = document.getElementById("stat-points");

// Characters alphabet
const ALPHABET = [
  ..."ABCDEFGHIJKLMNOPQRSTUVWXYZ",
  ..."abcdefghijklmnopqrstuvwxyz",
  ..."0123456789"
];
let targetIndex = 0;

// Trajectory state
let currentStroke = [];
let completedStrokes = [];
let isPenDown = false;
let lastPenUpTime = 0;
const DEBOUNCE_MS = 900;
let lastFrameTime = performance.now();
let fps = 0;

function updateTargetDisplay() {
  targetCharElem.textContent = ALPHABET[targetIndex];
  fetchStats();
}

function clearTrajectory() {
  currentStroke = [];
  completedStrokes = [];
  isPenDown = false;
  statStrokes.textContent = "0";
  statPoints.textContent = "0";
  renderCanvas();
}

function undoLastStroke() {
  if (completedStrokes.length > 0) {
    completedStrokes.pop();
    statStrokes.textContent = completedStrokes.length;
    renderCanvas();
  }
}

async function saveCurrentSample() {
  const allStrokes = [...completedStrokes];
  if (currentStroke.length > 0) allStrokes.push(currentStroke);

  const totalPts = allStrokes.reduce((acc, s) => acc + s.length, 0);
  if (totalPts < 6) return;

  const payload = {
    label: ALPHABET[targetIndex],
    user_id: userInput.value.trim() || "anonymous",
    strokes: allStrokes.map(stroke =>
      stroke.map(p => ({
        x: p.x,
        y: p.y,
        z: p.z || 0.0,
        t: p.t,
        p: 1
      }))
    ),
    metadata: { collected_at: new Date().toISOString() }
  };

  try {
    const res = await fetch("/api/collect/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (res.ok) {
      clearTrajectory();
      targetIndex = (targetIndex + 1) % ALPHABET.length;
      updateTargetDisplay();
    }
  } catch (err) {
    console.error("Save error:", err);
  }
}

async function fetchStats() {
  try {
    const res = await fetch("/api/collect/stats");
    if (res.ok) {
      const data = await res.json();
      statTotal.textContent = data.total_samples || 0;
      const target = ALPHABET[targetIndex];
      statTargetCount.textContent = (data.samples_by_class && data.samples_by_class[target]) || 0;
    }
  } catch (e) {}
}

function renderCanvas(fingertip = null) {
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  const w = canvasElement.width;
  const h = canvasElement.height;

  // Draw completed strokes
  const all = [...completedStrokes];
  if (currentStroke.length > 0) all.push(currentStroke);

  for (const stroke of all) {
    if (stroke.length < 2) continue;
    canvasCtx.beginPath();
    canvasCtx.strokeStyle = "rgba(6, 182, 212, 0.4)";
    canvasCtx.lineWidth = 10;
    canvasCtx.lineCap = "round";
    canvasCtx.lineJoin = "round";
    canvasCtx.moveTo(stroke[0].x * w, stroke[0].y * h);
    for (let i = 1; i < stroke.length; i++) {
      canvasCtx.lineTo(stroke[i].x * w, stroke[i].y * h);
    }
    canvasCtx.stroke();

    // Core sharp line
    canvasCtx.beginPath();
    canvasCtx.strokeStyle = "#38bdf8";
    canvasCtx.lineWidth = 4;
    canvasCtx.moveTo(stroke[0].x * w, stroke[0].y * h);
    for (let i = 1; i < stroke.length; i++) {
      canvasCtx.lineTo(stroke[i].x * w, stroke[i].y * h);
    }
    canvasCtx.stroke();
  }

  // Draw fingertip target dot
  if (fingertip) {
    const cx = fingertip.x * w;
    const cy = fingertip.y * h;
    canvasCtx.beginPath();
    canvasCtx.arc(cx, cy, isPenDown ? 12 : 7, 0, 2 * Math.PI);
    canvasCtx.fillStyle = isPenDown ? "#10b981" : "#38bdf8";
    canvasCtx.fill();
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = "#ffffff";
    canvasCtx.stroke();
  }
}

// MediaPipe results handler
function onResults(results) {
  const now = performance.now();
  const dt = now - lastFrameTime;
  lastFrameTime = now;
  if (dt > 0) {
    fps = 0.9 * fps + 0.1 * (1000 / dt);
    fpsDisplay.textContent = `FPS: ${fps.toFixed(0)}`;
  }

  // Ensure canvas matches container dimensions
  if (canvasElement.width !== videoElement.videoWidth && videoElement.videoWidth > 0) {
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
  }

  let fingertip = null;

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const lm = results.multiHandLandmarks[0];

    // Compute hand scale (wrist 0 to middle MCP 9)
    const wrist = lm[0];
    const middleMcp = lm[9];
    const handScale = Math.hypot(middleMcp.x - wrist.x, middleMcp.y - wrist.y);

    // Index tip 8, Thumb tip 4
    const indexTip = lm[8];
    const thumbTip = lm[4];
    const pinchDist = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y);
    const normalizedPinch = pinchDist / Math.max(handScale, 0.001);

    const pinchThreshold = 0.15;
    const penDownDetected = normalizedPinch < pinchThreshold;

    // Mirrored fingertip (natural writing)
    fingertip = {
      x: 1.0 - indexTip.x,
      y: indexTip.y,
      z: indexTip.z || 0.0
    };

    if (penDownDetected) {
      if (!isPenDown) {
        isPenDown = true;
        currentStroke = [];
      }
      currentStroke.push({ ...fingertip, t: now / 1000 });
      lastPenUpTime = 0;
    } else {
      if (isPenDown) {
        isPenDown = false;
        if (currentStroke.length >= 4) {
          completedStrokes.push(currentStroke);
        }
        currentStroke = [];
        lastPenUpTime = now;
      }
    }

    penDot.className = isPenDown ? "status-dot active" : "status-dot";
    penStatusText.textContent = isPenDown ? "PEN DOWN (WRITING)" : "PEN UP (HOVER)";
  } else {
    if (isPenDown) {
      isPenDown = false;
      if (currentStroke.length >= 4) completedStrokes.push(currentStroke);
      currentStroke = [];
      lastPenUpTime = now;
    }
  }

  // Update stroke and point counts
  const totalPoints = completedStrokes.reduce((acc, s) => acc + s.length, 0) + currentStroke.length;
  statStrokes.textContent = completedStrokes.length + (currentStroke.length > 0 ? 1 : 0);
  statPoints.textContent = totalPoints;

  renderCanvas(fingertip);

  // Auto-save on pen-up debounce
  if (lastPenUpTime > 0 && !isPenDown && completedStrokes.length > 0) {
    if (now - lastPenUpTime >= DEBOUNCE_MS) {
      lastPenUpTime = 0;
      saveCurrentSample();
    }
  }
}

// Controls & listeners
document.getElementById("btn-save").addEventListener("click", saveCurrentSample);
document.getElementById("btn-clear").addEventListener("click", clearTrajectory);
document.getElementById("btn-undo").addEventListener("click", undoLastStroke);
document.getElementById("btn-next-target").addEventListener("click", () => {
  targetIndex = (targetIndex + 1) % ALPHABET.length;
  clearTrajectory();
  updateTargetDisplay();
});

window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  if (e.code === "Space") {
    e.preventDefault();
    saveCurrentSample();
  } else if (e.key === "c" || e.key === "C") {
    clearTrajectory();
  } else if (e.key === "u" || e.key === "U") {
    undoLastStroke();
  } else if (e.key === "s" || e.key === "S") {
    targetIndex = (targetIndex + 1) % ALPHABET.length;
    clearTrajectory();
    updateTargetDisplay();
  }
});

// Initialize Camera and MediaPipe Hands
async function init() {
  updateTargetDisplay();

  const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });

  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7
  });

  hands.onResults(onResults);

  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    const camera = new Camera(videoElement, {
      onFrame: async () => {
        await hands.send({ image: videoElement });
      },
      width: 640,
      height: 480
    });
    camera.start();
  }
}

window.addEventListener("DOMContentLoaded", init);
