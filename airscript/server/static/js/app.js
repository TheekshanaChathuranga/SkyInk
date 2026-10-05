// SkyInk Live Air-Writing Recognition & Hand Pose Engine
const videoElement = document.getElementById("webcam-video");
const canvasElement = document.getElementById("drawing-canvas");
const canvasCtx = canvasElement.getContext("2d");

const penStatusPill = document.getElementById("pen-status-pill");
const penDot = document.getElementById("pen-dot");
const penStatusText = document.getElementById("pen-status-text");
const fpsDisplay = document.getElementById("fps-display");
const wsStatus = document.getElementById("ws-status");
const handIndicator = document.getElementById("hand-indicator");
const camStatusMsg = document.getElementById("cam-status-msg");
const activePoseBadge = document.getElementById("active-pose-badge");

const liveChar = document.getElementById("live-char");
const confidenceVal = document.getElementById("confidence-val");
const latencyVal = document.getElementById("latency-val");
const textBuffer = document.getElementById("text-buffer");
const gestureModeSelect = document.getElementById("gesture-mode");
const showSkeletonCheckbox = document.getElementById("show-skeleton");

let ws = null;
let lastFrameTime = performance.now();
let fps = 0;
let currentStrokes = [];
let isPenDown = false;
let latestLandmarks = null;
let activePoseName = "None";
let openPalmStartTime = null;

// MediaPipe 21 Hand Skeleton Connections
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // Index
  [5, 9], [9, 10], [10, 11], [11, 12],  // Middle
  [9, 13], [13, 14], [14, 15], [15, 16],// Ring
  [13, 17], [17, 18], [18, 19], [19, 20],// Pinky
  [0, 17]                               // Palm base
];

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
      console.error("WS error:", err);
    }
  };
}

function renderHandSkeleton(landmarks, w, h) {
  if (!landmarks || landmarks.length < 21 || !showSkeletonCheckbox.checked) return;

  // Draw bone lines
  canvasCtx.lineWidth = 2.5;
  canvasCtx.strokeStyle = isPenDown ? "rgba(16, 185, 129, 0.45)" : "rgba(6, 182, 212, 0.35)";
  canvasCtx.lineCap = "round";

  for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
    const p1 = landmarks[startIdx];
    const p2 = landmarks[endIdx];
    canvasCtx.beginPath();
    canvasCtx.moveTo(p1.x * w, p1.y * h);
    canvasCtx.lineTo(p2.x * w, p2.y * h);
    canvasCtx.stroke();
  }

  // Draw joint landmarks
  for (let i = 0; i < landmarks.length; i++) {
    const pt = landmarks[i];
    const cx = pt.x * w;
    const cy = pt.y * h;
    const isTip = [4, 8, 12, 16, 20].includes(i);

    canvasCtx.beginPath();
    canvasCtx.arc(cx, cy, isTip ? 5 : 3, 0, 2 * Math.PI);
    canvasCtx.fillStyle = isTip ? (i === 8 && isPenDown ? "#10b981" : "#38bdf8") : "rgba(255, 255, 255, 0.7)";
    canvasCtx.fill();
  }
}

function renderCanvas(fingertip = null) {
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  const w = canvasElement.width;
  const h = canvasElement.height;

  // 1. Draw Hand Skeleton
  if (latestLandmarks) {
    renderHandSkeleton(latestLandmarks, w, h);
  }

  // 2. Draw Trajectory Strokes with glowing neon effect
  for (const stroke of currentStrokes) {
    if (stroke.length < 2) continue;

    // Glowing halo
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

  // 3. Fingertip indicator and Floating Status Label
  const tip = fingertip || (latestLandmarks ? latestLandmarks[8] : null);
  if (tip) {
    const cx = (tip.x || tip[0]) * w;
    const cy = (tip.y || tip[1]) * h;

    // Target circle
    canvasCtx.beginPath();
    canvasCtx.arc(cx, cy, isPenDown ? 14 : 9, 0, 2 * Math.PI);
    canvasCtx.fillStyle = isPenDown ? "#10b981" : "#0284c7";
    canvasCtx.fill();
    canvasCtx.lineWidth = 2.5;
    canvasCtx.strokeStyle = "#ffffff";
    canvasCtx.stroke();

    if (isPenDown) {
      canvasCtx.beginPath();
      canvasCtx.arc(cx, cy, 22, 0, 2 * Math.PI);
      canvasCtx.strokeStyle = "rgba(16, 185, 129, 0.6)";
      canvasCtx.lineWidth = 2.5;
      canvasCtx.stroke();
    }

    // Floating Pose Text Badge above fingertip
    canvasCtx.font = "bold 13px system-ui, sans-serif";
    const label = isPenDown ? "✍️ DRAWING" : (activePoseName === "Two Fingers (✌️)" ? "✌️ PEN UP" : "HOVER");
    const textMetrics = canvasCtx.measureText(label);
    const boxW = textMetrics.width + 16;
    const boxH = 22;
    const boxX = cx - boxW / 2;
    const boxY = cy - 36;

    canvasCtx.fillStyle = isPenDown ? "rgba(16, 185, 129, 0.9)" : "rgba(15, 23, 42, 0.85)";
    canvasCtx.beginPath();
    canvasCtx.roundRect(boxX, boxY, boxW, boxH, 6);
    canvasCtx.fill();
    canvasCtx.strokeStyle = isPenDown ? "#10b981" : "rgba(255, 255, 255, 0.2)";
    canvasCtx.lineWidth = 1;
    canvasCtx.stroke();

    canvasCtx.fillStyle = "#ffffff";
    canvasCtx.fillText(label, boxX + 8, boxY + 16);
  }
}

// MediaPipe Results Handler
function onResults(results) {
  const now = performance.now();
  const dt = now - lastFrameTime;
  lastFrameTime = now;
  if (dt > 0) {
    fps = 0.9 * fps + 0.1 * (1000 / dt);
    fpsDisplay.textContent = `FPS: ${fps.toFixed(0)}`;
  }

  if (videoElement.videoWidth > 0 && canvasElement.width !== videoElement.videoWidth) {
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
    latestLandmarks = mirroredLm;

    handIndicator.textContent = "Hand: Detected";
    handIndicator.style.color = "var(--accent-green)";

    // Wrist
    const wrist = mirroredLm[0];

    // Rotation-invariant finger extension checker
    function isExtended(tipIdx, pipIdx, mcpIdx) {
      const dTip = Math.hypot(mirroredLm[tipIdx].x - wrist.x, mirroredLm[tipIdx].y - wrist.y);
      const dPip = Math.hypot(mirroredLm[pipIdx].x - wrist.x, mirroredLm[pipIdx].y - wrist.y);
      const dMcp = Math.hypot(mirroredLm[mcpIdx].x - wrist.x, mirroredLm[mcpIdx].y - wrist.y);
      return (dTip > dPip * 1.04) && (dTip > dMcp * 1.12);
    }

    const indexUp = isExtended(8, 6, 5);
    const middleUp = isExtended(12, 10, 9);
    const ringUp = isExtended(16, 14, 13);
    const pinkyUp = isExtended(20, 18, 17);

    // Compute pinch distance (Thumb 4 to Index 8)
    const middleMcp = mirroredLm[9];
    const handScale = Math.max(Math.hypot(middleMcp.x - wrist.x, middleMcp.y - wrist.y), 0.01);
    const thumb = mirroredLm[4];
    const index = mirroredLm[8];
    const pinchDist = Math.hypot(thumb.x - index.x, thumb.y - index.y);
    const normalizedPinch = pinchDist / handScale;

    // Gesture classifications:
    // 1-Finger: Index only extended -> WRITING (PEN DOWN)
    const isOneFingerWrite = indexUp && !middleUp && !ringUp;

    // 2-Fingers: Index + Middle extended -> PEN UP (HOVER / MOVE)
    const isTwoFingerHover = indexUp && middleUp && !ringUp;

    // Open Palm: 4 or 5 fingers extended -> CLEAR
    const isOpenPalm = indexUp && middleUp && ringUp && pinkyUp;

    // Pinch: Thumb + Index close together
    const isPinch = normalizedPinch < 0.18;

    // Handle Open Palm Clear
    if (isOpenPalm) {
      activePoseName = "Open Palm (✋)";
      activePoseBadge.textContent = "Pose: ✋ Open Palm";
      activePoseBadge.style.color = "#f59e0b";
      if (!openPalmStartTime) openPalmStartTime = now;
      const elapsed = Math.round((now - openPalmStartTime) / 100) / 10;
      camStatusMsg.textContent = `✋ Open Palm detected! Hold to clear canvas (${elapsed}s / 1.0s)`;

      if (now - openPalmStartTime >= 1000) {
        currentStrokes = [];
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "clear" }));
        camStatusMsg.textContent = "🧹 Canvas Cleared!";
        openPalmStartTime = null;
      }
    } else {
      openPalmStartTime = null;
    }

    // Determine writing state based on user's selected mode
    const mode = gestureModeSelect.value;
    let penDown = false;

    if (mode === "one_finger") {
      penDown = isOneFingerWrite;
      if (isOneFingerWrite) {
        activePoseName = "1 Finger (☝️)";
        activePoseBadge.textContent = "Pose: ☝️ 1-Finger (Drawing)";
        activePoseBadge.style.color = "var(--accent-green)";
      } else if (isTwoFingerHover) {
        activePoseName = "Two Fingers (✌️)";
        activePoseBadge.textContent = "Pose: ✌️ 2-Fingers (Pen Up)";
        activePoseBadge.style.color = "var(--accent-cyan)";
      } else {
        activePoseName = "Idle";
        activePoseBadge.textContent = "Pose: Idle / Moving";
        activePoseBadge.style.color = "#94a3b8";
      }
    } else if (mode === "pinch") {
      penDown = isPinch;
      activePoseName = isPinch ? "Pinch (🤏)" : "Hover";
      activePoseBadge.textContent = isPinch ? "Pose: 🤏 Pinch (Drawing)" : `Pinch: ${normalizedPinch.toFixed(2)}`;
      activePoseBadge.style.color = isPinch ? "var(--accent-green)" : "#38bdf8";
    } else { // auto: 1-finger write OR pinch
      penDown = isOneFingerWrite || isPinch;
      activePoseName = penDown ? "Drawing" : (isTwoFingerHover ? "2-Fingers (✌️)" : "Hover");
      activePoseBadge.textContent = penDown ? "Pose: ✍️ Drawing" : (isTwoFingerHover ? "Pose: ✌️ Pen Up" : "Pose: Hover");
      activePoseBadge.style.color = penDown ? "var(--accent-green)" : "var(--accent-cyan)";
    }

    isPenDown = penDown;
    penDot.className = isPenDown ? "status-dot active" : "status-dot";
    penStatusText.textContent = isPenDown ? "PEN DOWN (DRAWING)" : "PEN UP (HOVER)";

    if (!isOpenPalm) {
      if (isPenDown) {
        camStatusMsg.textContent = "✍️ Drawing stroke... Lift 2nd finger (✌️) to pause or lift pen";
      } else if (isTwoFingerHover) {
        camStatusMsg.textContent = "✌️ Pen Up! Move hand anywhere. Curl middle finger to draw again";
      } else {
        camStatusMsg.textContent = "💡 Point 1 finger (☝️) to draw. Raise 2 fingers (✌️) to hover.";
      }
    }

    // Stream landmarks & active mode to backend
    if (ws && ws.readyState === WebSocket.OPEN) {
      const sendStart = performance.now();
      ws.send(
        JSON.stringify({
          type: "landmarks",
          gesture_mode: mode,
          landmarks: mirroredLm,
          timestamp: sendStart / 1000,
        })
      );
      latencyVal.textContent = `~${Math.round(performance.now() - sendStart)}ms`;
    }

    renderCanvas(mirroredLm[8]);
  } else {
    latestLandmarks = null;
    handIndicator.textContent = "Hand: Searching...";
    handIndicator.style.color = "var(--accent-red)";
    penDot.className = "status-dot";
    penStatusText.textContent = "NO HAND DETECTED";
    camStatusMsg.textContent = "💡 Bring your hand in front of the camera with fingers visible.";
    activePoseBadge.textContent = "Pose: —";

    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "landmarks", landmarks: [], timestamp: now / 1000 }));
    }
    renderCanvas();
  }
}

// UI Controls
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
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "clear" }));
  currentStrokes = [];
  renderCanvas();
});

document.getElementById("btn-undo-stroke").addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "undo" }));
});

// Keyboard hotkeys
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

// Camera and MediaPipe Hands Initializer
async function init() {
  connectWebSocket();

  try {
    camStatusMsg.textContent = "📷 Requesting webcam access...";
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user"
      },
      audio: false
    });

    videoElement.srcObject = stream;
    await videoElement.play();
    camStatusMsg.textContent = "✅ Camera ready. Initializing MediaPipe Hands...";

    const hands = new Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    hands.onResults(onResults);

    let isProcessing = false;
    async function processLoop() {
      if (videoElement.readyState >= 2 && !isProcessing) {
        isProcessing = true;
        try {
          await hands.send({ image: videoElement });
        } catch (err) {
          console.warn("Hands.send error:", err);
        }
        isProcessing = false;
      }
      requestAnimationFrame(processLoop);
    }

    processLoop();
    camStatusMsg.textContent = "✨ SkyInk ready! Point 1 finger (☝️) to draw, 2 fingers (✌️) to lift pen.";
  } catch (err) {
    console.error("Camera access error:", err);
    camStatusMsg.textContent = `❌ Camera error: ${err.message}. Please allow camera permissions in browser.`;
    penStatusText.textContent = "CAMERA ERROR";
    penDot.className = "status-dot";
  }
}

window.addEventListener("DOMContentLoaded", init);
