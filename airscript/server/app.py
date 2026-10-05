"""FastAPI application for AirScript real-time recognition and data collection."""
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from airscript.core.tracker import HandFingertipTracker
from airscript.core.trajectory import Trajectory
from airscript.core.preprocessor import TrajectoryPreprocessor
from airscript.dataset.dataset import save_dataset_npz
from airscript.utils.config import load_config
from airscript.utils.logger import setup_logger

logger = setup_logger("airscript_server")

app = FastAPI(title="AirScript API", version="0.1.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
config = load_config()

# Directories
STATIC_DIR = Path(__file__).parent / "static"
RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class CollectSamplePayload(BaseModel):
    label: str
    user_id: str
    strokes: List[List[Dict[str, Any]]]
    metadata: Optional[Dict[str, Any]] = None


@app.get("/")
def read_root():
    """Serves the main air-writing interface."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "AirScript Server Running"}


@app.get("/collect")
def read_collector():
    """Serves the data collection interface."""
    collect_file = STATIC_DIR / "collect.html"
    if collect_file.exists():
        return FileResponse(collect_file)
    return {"message": "Data collection page"}


@app.post("/api/collect/save")
def save_collected_sample(payload: CollectSamplePayload):
    """Saves a single air-writing sample to server storage."""
    traj = Trajectory.from_dict({
        "label": payload.label,
        "user_id": payload.user_id,
        "metadata": payload.metadata or {},
        "strokes": payload.strokes,
    })

    if traj.total_points() < 4:
        raise HTTPException(status_code=400, detail="Trajectory contains insufficient points.")

    timestamp = int(time.time() * 1000)
    filename = f"sample_{payload.user_id}_{payload.label}_{timestamp}.json"
    file_path = RAW_DATA_DIR / filename

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(traj.to_dict(), f, indent=2)

    logger.info(f"Saved sample: {filename} ({traj.total_points()} points)")
    return {"status": "success", "file": filename, "points": traj.total_points()}


@app.get("/api/collect/stats")
def get_collection_stats():
    """Returns count of collected samples grouped by label and user."""
    stats: Dict[str, int] = {}
    users: Dict[str, int] = {}
    total = 0

    for p in RAW_DATA_DIR.glob("*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            lbl = data.get("label", "unknown")
            usr = data.get("user_id", "anonymous")
            stats[lbl] = stats.get(lbl, 0) + 1
            users[usr] = users.get(usr, 0) + 1
            total += 1
        except Exception:
            continue

    return {
        "total_samples": total,
        "samples_by_class": stats,
        "samples_by_user": users,
    }


# Active model runner reference (populated in Phase 3/4/5)
MODEL_RUNNER = None


@app.websocket("/ws/recognize")
async def websocket_recognize(websocket: WebSocket):
    """WebSocket endpoint for real-time landmark streaming, gesture tracking, and decoding."""
    await websocket.accept()
    tracker = HandFingertipTracker(
        min_detection_confidence=config.tracker.min_detection_confidence,
        min_tracking_confidence=config.tracker.min_tracking_confidence,
        pinch_threshold=config.tracker.pinch_threshold,
        debounce_timeout=config.tracker.debounce_timeout_ms / 1000.0,
    )
    preprocessor = TrajectoryPreprocessor()
    logger.info("WebSocket client connected to /ws/recognize")

    try:
        while True:
            data = await websocket.receive_json()
            # Expected msg: {"type": "landmarks", "landmarks": [...], "timestamp": ...}
            msg_type = data.get("type", "landmarks")

            if msg_type == "clear":
                tracker.reset_trajectory()
                await websocket.send_json({"type": "cleared"})
                continue
            elif msg_type == "undo":
                tracker.undo_last_stroke()
                await websocket.send_json({"type": "undone"})
                continue

            landmarks = data.get("landmarks", [])
            ts = data.get("timestamp", time.perf_counter())

            result = tracker.process_landmarks(landmarks, timestamp=ts)

            resp: Dict[str, Any] = {
                "type": "frame_update",
                "hand_detected": result.hand_detected,
                "is_pen_down": result.is_pen_down,
                "gesture": result.gesture,
                "fingertip": result.fingertip,
                "all_strokes": result.all_strokes,
                "action_trigger": result.action_trigger,
                "fps": round(result.fps, 1),
            }

            if result.should_recognize:
                traj = tracker.get_current_trajectory()
                if not traj.is_empty():
                    # Preprocess features
                    features = preprocessor.process(traj, target_points=64)

                    recognized_text = ""
                    confidence = 0.0

                    if MODEL_RUNNER is not None:
                        recognized_text, confidence = MODEL_RUNNER.predict(features)
                    else:
                        recognized_text = f"Sample_{len(traj.strokes)}s"
                        confidence = 0.95

                    resp["recognition"] = {
                        "text": recognized_text,
                        "confidence": float(confidence),
                        "num_strokes": len(traj.strokes),
                        "points": traj.total_points(),
                    }

            await websocket.send_json(resp)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
