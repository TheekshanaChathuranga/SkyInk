"""Interactive CLI tool to collect labeled air-writing trajectories.

Usage:
    python scripts/collect_cli.py --user my_name --chars A-Z
    python scripts/collect_cli.py --user tester --demo  (generates sample dataset)
"""
import argparse
import json
from pathlib import Path
import sys
import time
from typing import List

# Ensure airscript is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from airscript.core.tracker import HandFingertipTracker
from airscript.core.trajectory import Trajectory
from airscript.dataset.dataset import save_dataset_npz
from airscript.dataset.synthetic import SyntheticStrokeGenerator
from airscript.utils.logger import setup_logger

logger = setup_logger("collect_cli")


def collect_with_webcam(user_id: str, targets: List[str], output_dir: Path, gesture: str = "pinch"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam.")
        return

    tracker = HandFingertipTracker(gesture_mode=gesture, debounce_timeout=0.9)
    collected: List[Trajectory] = []
    target_idx = 0

    print("\n" + "=" * 60)
    print(f"AirScript Data Collection - User: {user_id}")
    print("Controls:")
    print("  [SPACE] Save current stroke trajectory")
    print("  [C]     Clear current trajectory")
    print("  [U]     Undo last stroke")
    print("  [S]     Skip to next character")
    print("  [Q]     Save and quit")
    print("=" * 60 + "\n")

    while target_idx < len(targets):
        current_target = targets[target_idx]
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        result = tracker.process_frame(frame)

        # Draw HUD
        hud = frame.copy()
        cv2.rectangle(hud, (0, 0), (w, 75), (25, 25, 30), -1)
        cv2.addWeighted(hud, 0.75, frame, 0.25, 0, frame)

        prompt_str = f"PROMPT: Write '{current_target}' ({target_idx + 1}/{len(targets)})"
        cv2.putText(frame, prompt_str, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2, cv2.LINE_AA)

        pen_color = (0, 230, 115) if result.is_pen_down else (80, 80, 255)
        pen_text = "PEN DOWN (PINCH)" if result.is_pen_down else "PEN UP (HOVER)"
        cv2.putText(frame, pen_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, pen_color, 2, cv2.LINE_AA)

        info_text = f"Collected: {len(collected)} | Points: {tracker.get_current_trajectory().total_points()}"
        cv2.putText(frame, info_text, (w - 360, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        # Draw strokes
        for stroke in result.all_strokes:
            if len(stroke) >= 2:
                pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in stroke], dtype=np.int32)
                cv2.polylines(frame, [pts], False, (0, 220, 255), 3, cv2.LINE_AA)

        # Fingertip indicator
        if result.fingertip:
            cx, cy = int(result.fingertip[0] * w), int(result.fingertip[1] * h)
            c = (0, 230, 115) if result.is_pen_down else (0, 190, 255)
            cv2.circle(frame, (cx, cy), 10, c, -1, cv2.LINE_AA)

        cv2.imshow("AirScript Data Collector", frame)
        key = cv2.waitKey(1) & 0xFF

        # Auto-trigger on pen-up debounce timeout or manual [SPACE]
        auto_save = result.should_recognize and tracker.get_current_trajectory().total_points() >= 12
        if key == 32 or auto_save:  # SPACE or auto debounce
            traj = tracker.get_current_trajectory()
            if traj.total_points() >= 8:
                traj.label = current_target
                traj.user_id = user_id
                collected.append(traj)
                logger.info(f"Saved sample for '{current_target}' ({traj.total_points()} points)")
                target_idx += 1
                tracker.reset_trajectory()
            else:
                logger.warning("Trajectory too short. Try again.")
        elif key == ord("c"):
            tracker.reset_trajectory()
        elif key == ord("u"):
            tracker.undo_last_stroke()
        elif key == ord("s"):
            logger.info(f"Skipped '{current_target}'")
            target_idx += 1
            tracker.reset_trajectory()
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if collected:
        out_file = output_dir / f"airwriting_{user_id}_{int(time.time())}.npz"
        save_dataset_npz(collected, out_file)
        logger.info(f"Saved {len(collected)} samples to: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="AirScript Data Collection")
    parser.add_argument("--user", type=str, default="user_pilot", help="User / Subject ID")
    parser.add_argument("--chars", type=str, default="0-9,A-Z,a-z", help="Characters to collect")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic baseline dataset")
    parser.add_argument("--samples_per_class", type=int, default=20, help="Samples per class in demo mode")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Target output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        logger.info(f"Generating synthetic baseline dataset ({args.samples_per_class} per class)...")
        gen = SyntheticStrokeGenerator()
        dataset = gen.generate_balanced_dataset(samples_per_class=args.samples_per_class)
        out_file = out_dir / "synthetic_dataset.npz"
        save_dataset_npz(dataset, out_file)
        logger.info(f"Generated {len(dataset)} trajectories saved to: {out_file}")
        return

    # Parse character targets
    targets = []
    for part in args.chars.split(","):
        part = part.strip()
        if "-" in part and len(part) == 3:
            start_c, end_c = part[0], part[2]
            targets.extend([chr(c) for c in range(ord(start_c), ord(end_c) + 1)])
        else:
            targets.extend(list(part))

    collect_with_webcam(args.user, targets, out_dir)


if __name__ == "__main__":
    main()
