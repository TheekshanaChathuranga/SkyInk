// SkyInk Live Air-Writing Recognition Client
const videoElement = document.getElementById("webcam-video");
const canvasElement = document.getElementById("drawing-canvas");
const canvasCtx = canvasElement.getContext("2d");

const penStatusPill = document.getElementById("pen-status-pill");
const penDot = document.getElementById("pen-dot");
const penStatusText = document.getElementById("pen-status-text");
const fpsDisplay = document.getElementById("fps-display");
const wsStatus = document.getElementById("ws-status");

const liveChar = document.getElementById("live-char");
const confidenceVal = document.getElementById("confidence-val");
const latencyVal = document.getElementById("latency-val");
const textBuffer = document.getElementById("text-buffer");

let ws = null;
let lastFrameTime = performance.now();
let fps = 0;
let currentStrokes = [];
let isPenDown = false;

// Initialize WebSocket connection
function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/recognize`;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    wsStatus.textContent = "● Connected";
    wsStatus.style.color = "var(--accent-green)";
  };

  ws.onclose = () => {
    wsStatus.textContent = "○ Reconnecting...";
    wsStatus.style.color = "var(--accent-red)";
    setTimeout(connectWebSocket, 1500);
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "frame_update") {
        isPenDown = data.is_pen_down;
        penDot.className = isPenDown ? "status-dot active" : "status-dot";
        penStatusText.textContent = isPenDown ? "PEN DOWN (WRITING)" : "PEN UP (HOVER)";

        if (data.action_trigger === "clear") {
          currentStrokes = [];
          renderCanvas();
        } else if (data.all_strokes) {
          currentStrokes = data.all_strokes;
          renderCanvas(data.fingertip);
        }

        if (data.recognition) {
          const rec = data.recognition;
          liveChar.textContent = rec.text;
          confidenceVal.textContent = `${Math.round(rec.confidence * 100)}%`;
          textBuffer.value += rec.text;
          textBuffer.scrollTop = textBuffer.scrollHeight;
        }
      }
    } catch (err) {
      console.error("WS parse error:", err);
    }
  };
}

function renderCanvas(fingertip = null) {
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  const w = canvasElement.width;
  const h = canvasElement.height;

  // Render all active strokes with neon glow
  for (const stroke of currentStrokes) {
    if (stroke.length < 2) continue;

    // Glowing outer halo
    canvasCtx.beginPath();
    canvasCtx.strokeStyle = "rgba(6, 182, 212, 0.35)";
    canvasCtx.lineWidth = 10;
    canvasCtx.lineCap = "round";
    canvasCtx.lineJoin = "round";
    canvasCtx.moveTo(stroke[0][0] * w, stroke[0][1] * h);
    for (let i = 1; i < stroke.length; i++) {
      canvasCtx.lineTo(stroke[i][0] * w, stroke[i][1] * h);
    }
    canvasCtx.stroke();

    // Vibrant core stroke
    canvasCtx.beginPath();
    canvasCtx.strokeStyle = "#38bdf8";
    canvasCtx.lineWidth = 4;
    canvasCtx.moveTo(stroke[0][0] * w, stroke[0][1] * h);
    for (let i = 1; i < stroke.length; i++) {
      canvasCtx.lineTo(stroke[i][0] * w, stroke[i][1] * h);
    }
    canvasCtx.stroke();
  }

  // Fingertip cursor indicator
  if (fingertip) {
    const cx = fingertip[0] * w;
    const cy = fingertip[1] * h;
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

  if (canvasElement.width !== videoElement.videoWidth && videoElement.videoWidth > 0) {
    canvasElement.width = videoElement.videoWidth;
    canvasElement.height = videoElement.videoHeight;
  }

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const rawLm = results.multiHandLandmarks[0];
    // Mirror X coordinates for natural interaction
    const mirroredLm = rawLm.map((lm) => ({
      x: 1.0 - lm.x,
      y: lm.y,
      z: lm.z || 0.0,
    }));

    if (ws && ws.readyState === WebSocket.OPEN) {
      const sendStart = performance.now();
      ws.send(
        JSON.stringify({
          type: "landmarks",
          landmarks: mirroredLm,
          timestamp: sendStart / 1000,
        })
      );
      latencyVal.textContent = `~${Math.round(performance.now() - sendStart)}ms`;
    }
  } else {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "landmarks", landmarks: [], timestamp: now / 1000 }));
    }
  }
}

// UI Button Handlers
document.getElementById("btn-add-space").addEventListener("click", () => {
  textBuffer.value += " ";
});

document.getElementById("btn-backspace").addEventListener("click", () => {
  textBuffer.value = textBuffer.value.slice(0, -1);
});

document.getElementById("btn-clear-buffer").addEventListener("click", () => {
  textBuffer.value = "";
  liveChar.textContent = "—";
});

document.getElementById("btn-clear-canvas").addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "clear" }));
  }
  currentStrokes = [];
  renderCanvas();
});

document.getElementById("btn-undo-stroke").addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "undo" }));
  }
});

// Keyboard shortcuts
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (e.key === "c" || e.key === "C") {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "clear" }));
    currentStrokes = [];
    renderCanvas();
  } else if (e.key === "u" || e.key === "U") {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "undo" }));
  } else if (e.code === "Space") {
    e.preventDefault();
    textBuffer.value += " ";
  } else if (e.code === "Backspace") {
    textBuffer.value = textBuffer.value.slice(0, -1);
  }
});

// Initialize Camera and MediaPipe Hands
async function init() {
  connectWebSocket();

  const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
  });

  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7,
  });

  hands.onResults(onResults);

  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    const camera = new Camera(videoElement, {
      onFrame: async () => {
        await hands.send({ image: videoElement });
      },
      width: 640,
      height: 480,
    });
    camera.start();
  }
}

window.addEventListener("DOMContentLoaded", init);
