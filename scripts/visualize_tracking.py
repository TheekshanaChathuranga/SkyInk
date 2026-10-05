"""Live fingertip tracking and trajectory visualization.

Can run with a live webcam or in demo simulation mode if webcam is not accessible.
Usage:
    python scripts/visualize_tracking.py --source 0
    python scripts/visualize_tracking.py --demo
"""
import argparse
from pathlib import Path
import sys
import time
from typing import Optional

# Ensure airscript is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from airscript.core.tracker import HandFingertipTracker
from airscript.core.preprocessor import TrajectoryPreprocessor
from airscript.utils.logger import setup_logger

logger = setup_logger("visualizer")


def draw_hud(
    frame: np.ndarray,
    fps: float,
    is_pen_down: bool,
    gesture: str,
    stroke_count: int,
    total_points: int,
    tip_coord: Optional[tuple] = None,
):
    """Draws rich information HUD on top of the video frame."""
    h, w = frame.shape[:2]

    # Glassmorphism dark HUD bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Title & FPS
    cv2.putText(
        frame,
        f"AirScript Tracking | FPS: {fps:.1f}",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Pen state pill
    pen_color = (0, 230, 115) if is_pen_down else (70, 70, 255)
    pen_text = "PEN DOWN (WRITING)" if is_pen_down else "PEN UP (HOVER)"
    cv2.putText(
        frame,
        pen_text,
        (15, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        pen_color,
        2,
        cv2.LINE_AA,
    )

    # Stroke stats
    stats_text = f"Strokes: {stroke_count} | Points: {total_points} | Gesture: {gesture}"
    text_size = cv2.getTextSize(stats_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    cv2.putText(
        frame,
        stats_text,
        (max(w - text_size[0] - 15, 20), 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 220, 240),
        1,
        cv2.LINE_AA,
    )

    # Draw fingertip target indicator
    if tip_coord is not None:
        cx, cy = int(tip_coord[0] * w), int(tip_coord[1] * h)
        radius = 12 if is_pen_down else 7
        color = (0, 230, 115) if is_pen_down else (0, 190, 255)
        cv2.circle(frame, (cx, cy), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), radius + 4, (255, 255, 255), 2, cv2.LINE_AA)


def draw_strokes(frame: np.ndarray, strokes: list, is_active: bool = False):
    """Draws glowing smooth trajectory strokes onto the frame."""
    h, w = frame.shape[:2]
    for stroke_idx, stroke in enumerate(strokes):
        if len(stroke) < 2:
            continue
        pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in stroke], dtype=np.int32)

        # Glow effect
        cv2.polylines(frame, [pts], isClosed=False, color=(50, 100, 255), thickness=6, lineType=cv2.LINE_AA)
        # Core line
        cv2.polylines(frame, [pts], isClosed=False, color=(0, 240, 255), thickness=3, lineType=cv2.LINE_AA)


def run_demo_simulation(output_path: str = "demo_trajectory.png"):
    """Generates synthetic letter 'S' trajectory in simulation mode for validation."""
    logger.info("Running AirScript tracker simulation mode...")
    tracker = HandFingertipTracker(gesture_mode="pinch")
    preprocessor = TrajectoryPreprocessor(target_points=64)

    # Generate synthetic letter 'S' curve
    t = np.linspace(0, 1.0, 50)
    # Parametric 'S'
    xs = 0.5 + 0.15 * np.sin(2 * np.pi * t)
    ys = 0.3 + 0.4 * t

    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas[:] = (30, 32, 40)

    strokes = []
    current = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        # Inject realistic tremor/jitter
        noisy_x = x + np.random.normal(0, 0.003)
        noisy_y = y + np.random.normal(0, 0.003)

        fx, fy, _ = tracker.point_filter.filter(noisy_x, noisy_y, 0.0, timestamp=i * 0.033)
        current.append((fx, fy))

    strokes.append(current)
    draw_strokes(canvas, strokes)
    draw_hud(
        canvas,
        fps=30.0,
        is_pen_down=True,
        gesture="writing",
        stroke_count=1,
        total_points=len(current),
        tip_coord=current[-1],
    )

    # Preprocess
    features = preprocessor.process(strokes, target_points=64)
    logger.info(f"Preprocessed features shape: {features.shape} (target_points x features)")

    cv2.imwrite(output_path, canvas)
    logger.info(f"Saved demo visualization to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="AirScript Live Tracker & Visualizer")
    parser.add_argument("--source", type=int, default=0, help="Camera device index")
    parser.add_argument("--demo", action="store_true", help="Run simulation test without camera")
    parser.add_argument("--gesture", type=str, default="pinch", choices=["pinch", "point", "auto"])
    args = parser.parse_args()

    if args.demo:
        run_demo_simulation()
        return

    logger.info(f"Attempting to open camera device {args.source}...")
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        logger.warning(f"Could not open camera {args.source}. Falling back to simulation demo.")
        run_demo_simulation()
        return

    tracker = HandFingertipTracker(gesture_mode=args.gesture)
    logger.info("Camera successfully opened! Controls: [C] Clear, [U] Undo, [Q] Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror frame horizontally for natural air-writing experience
        frame = cv2.flip(frame, 1)

        result = tracker.process_frame(frame)

        # Draw all trajectories
        draw_strokes(frame, result.all_strokes)

        # Draw HUD
        tip = result.fingertip[:2] if result.fingertip else None
        draw_hud(
            frame,
            fps=result.fps,
            is_pen_down=result.is_pen_down,
            gesture=result.gesture,
            stroke_count=len(result.all_strokes),
            total_points=tracker.get_current_trajectory().total_points(),
            tip_coord=tip,
        )

        cv2.imshow("AirScript - Live Tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            tracker.reset_trajectory()
        elif key == ord("u"):
            tracker.undo_last_stroke()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
