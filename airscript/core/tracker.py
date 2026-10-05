"""Fingertip tracking and gesture recognition engine using MediaPipe Hands."""
from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .one_euro_filter import PointFilter
from .trajectory import Point, Stroke, Trajectory


@dataclass
class FrameResult:
    """Detection and tracking results for a single video frame."""
    hand_detected: bool = False
    fingertip: Optional[Tuple[float, float, float]] = None  # Filtered (x, y, z) in [0, 1]
    raw_fingertip: Optional[Tuple[float, float, float]] = None
    is_pen_down: bool = False
    gesture: str = "none"  # "writing", "hover", "open_palm_clear", "none"
    current_stroke: List[Tuple[float, float]] = field(default_factory=list)
    all_strokes: List[List[Tuple[float, float]]] = field(default_factory=list)
    should_recognize: bool = False
    action_trigger: Optional[str] = None  # "clear", "undo", "submit"
    landmarks_2d: Optional[List[Tuple[float, float]]] = None
    fps: float = 0.0


class HandFingertipTracker:
    """Tracks index fingertip and detects writing / control gestures in real time."""

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        pinch_threshold: float = 0.12,  # normalized by hand scale
        gesture_mode: str = "pinch",  # "pinch", "point", or "auto"
        debounce_timeout: float = 0.8,  # seconds after pen-up to trigger recognition
        filter_min_cutoff: float = 1.0,
        filter_beta: float = 0.05,
    ):
        self.pinch_threshold = pinch_threshold
        self.gesture_mode = gesture_mode
        self.debounce_timeout = debounce_timeout

        self.point_filter = PointFilter(min_cutoff=filter_min_cutoff, beta=filter_beta)

        # State tracking
        self.current_stroke = Stroke()
        self.completed_strokes: List[Stroke] = []
        self.last_pen_down_time: Optional[float] = None
        self.last_pen_up_time: Optional[float] = None
        self.is_currently_pen_down = False
        self.open_palm_start_time: Optional[float] = None

        # FPS calculation
        self._prev_frame_time = time.perf_counter()
        self._fps = 0.0

        # Lazy init mediapipe
        self.mp_hands = None
        self.hands_detector = None
        self._init_mediapipe(min_detection_confidence, min_tracking_confidence)

    def _init_mediapipe(self, min_det: float, min_track: float):
        try:
            import mediapipe as mp
            # Handle both mediapipe solutions API and newer mediapipe versions
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                self.mp_hands = mp.solutions.hands
                self.hands_detector = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=min_det,
                    min_tracking_confidence=min_track,
                )
            else:
                # Tasks API or fallback
                self.mp_hands = None
                self.hands_detector = None
        except Exception:
            self.mp_hands = None
            self.hands_detector = None

    def reset_trajectory(self):
        """Clears all accumulated strokes and active stroke."""
        self.current_stroke = Stroke()
        self.completed_strokes = []
        self.last_pen_down_time = None
        self.last_pen_up_time = None
        self.is_currently_pen_down = False
        self.open_palm_start_time = None
        self.point_filter.reset()

    def undo_last_stroke(self) -> bool:
        """Removes the most recent stroke."""
        if self.completed_strokes:
            self.completed_strokes.pop()
            return True
        return False

    def get_current_trajectory(self) -> Trajectory:
        """Returns the full active Trajectory object."""
        traj = Trajectory()
        for s in self.completed_strokes:
            traj.add_stroke(s)
        if len(self.current_stroke) > 0:
            traj.add_stroke(self.current_stroke)
        return traj

    def _calculate_hand_scale(self, landmarks) -> float:
        """Computes scale of hand: Euclidean distance from wrist (0) to middle MCP (9)."""
        w = landmarks[0]
        m = landmarks[9]
        scale = math.hypot(m.x - w.x, m.y - w.y)
        return max(scale, 1e-4)

    def _is_pinch(self, landmarks, hand_scale: float) -> Tuple[bool, float]:
        """Detects pinch between thumb tip (4) and index tip (8)."""
        thumb = landmarks[4]
        index = landmarks[8]
        dist = math.hypot(thumb.x - index.x, thumb.y - index.y)
        normalized_dist = dist / hand_scale
        return normalized_dist < self.pinch_threshold, normalized_dist

    def _is_pointing(self, landmarks) -> bool:
        """Detects pointing: index extended, other 3 fingers folded."""
        index_tip = landmarks[8].y < landmarks[6].y
        middle_folded = landmarks[12].y > landmarks[10].y
        ring_folded = landmarks[16].y > landmarks[14].y
        pinky_folded = landmarks[20].y > landmarks[18].y
        return index_tip and middle_folded and ring_folded and pinky_folded

    def _is_open_palm(self, landmarks) -> bool:
        """Detects open palm gesture (all fingers extended)."""
        fingers_up = (
            landmarks[4].x < landmarks[3].x if landmarks[4].x < landmarks[0].x else landmarks[4].x > landmarks[3].x
        )
        index_up = landmarks[8].y < landmarks[6].y
        middle_up = landmarks[12].y < landmarks[10].y
        ring_up = landmarks[16].y < landmarks[14].y
        pinky_up = landmarks[20].y < landmarks[18].y
        return index_up and middle_up and ring_up and pinky_up

    def process_landmarks(
        self,
        landmarks: List[Any],
        timestamp: Optional[float] = None,
    ) -> FrameResult:
        """Processes normalized hand landmarks (e.g. from MediaPipe or browser client)."""
        now = timestamp if timestamp is not None else time.perf_counter()

        # Update FPS
        dt = now - self._prev_frame_time
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
        self._prev_frame_time = now

        result = FrameResult(hand_detected=True, fps=self._fps)

        # Convert landmarks to structured format if needed
        # Each landmark has x, y, z
        class SimpleLM:
            def __init__(self, x, y, z=0.0):
                self.x = x
                self.y = y
                self.z = z

        lm_objs = []
        lm_2d = []
        for lm in landmarks:
            if hasattr(lm, "x"):
                x, y, z = lm.x, lm.y, getattr(lm, "z", 0.0)
            elif isinstance(lm, (list, tuple)):
                x, y = lm[0], lm[1]
                z = lm[2] if len(lm) > 2 else 0.0
            elif isinstance(lm, dict):
                x, y = lm["x"], lm["y"]
                z = lm.get("z", 0.0)
            else:
                continue
            lm_objs.append(SimpleLM(x, y, z))
            lm_2d.append((x, y))

        result.landmarks_2d = lm_2d

        if len(lm_objs) < 21:
            return result

        hand_scale = self._calculate_hand_scale(lm_objs)

        # Fingertip coordinates (landmark 8)
        raw_idx = lm_objs[8]
        raw_pt = (raw_idx.x, raw_idx.y, raw_idx.z)
        result.raw_fingertip = raw_pt

        # Filtered coordinate using One Euro Filter
        fx, fy, fz = self.point_filter.filter(raw_idx.x, raw_idx.y, raw_idx.z, timestamp=now)
        result.fingertip = (fx, fy, fz)

        # Detect clear gesture (open palm held for >= 1.0 second)
        if self._is_open_palm(lm_objs):
            if self.open_palm_start_time is None:
                self.open_palm_start_time = now
            elif (now - self.open_palm_start_time) >= 1.0:
                self.reset_trajectory()
                result.action_trigger = "clear"
                result.gesture = "open_palm_clear"
                self.open_palm_start_time = None
                return result
        else:
            self.open_palm_start_time = None

        # Determine pen-down state
        is_pen_down = False
        if self.gesture_mode == "pinch":
            pinch_detected, _ = self._is_pinch(lm_objs, hand_scale)
            is_pen_down = pinch_detected
        elif self.gesture_mode == "point":
            is_pen_down = self._is_pointing(lm_objs)
        else:  # "auto": pinch or pointing
            pinch_detected, _ = self._is_pinch(lm_objs, hand_scale)
            is_pen_down = pinch_detected or self._is_pointing(lm_objs)

        result.is_pen_down = is_pen_down
        result.gesture = "writing" if is_pen_down else "hover"

        # Trajectory state machine
        pt = Point(x=fx, y=fy, z=fz, timestamp=now, is_pen_down=is_pen_down)

        if is_pen_down:
            if not self.is_currently_pen_down:
                # Transition: Pen went DOWN
                self.is_currently_pen_down = True
                self.current_stroke = Stroke()
            self.current_stroke.add_point(pt)
            self.last_pen_down_time = now
            self.last_pen_up_time = None
        else:
            if self.is_currently_pen_down:
                # Transition: Pen went UP
                self.is_currently_pen_down = False
                if len(self.current_stroke) >= 4:  # filter accidental taps
                    self.completed_strokes.append(self.current_stroke)
                self.current_stroke = Stroke()
                self.last_pen_up_time = now

        # Check debounce timeout for word completion trigger
        should_recognize = False
        if (
            self.last_pen_up_time is not None
            and not self.is_currently_pen_down
            and len(self.completed_strokes) > 0
        ):
            if (now - self.last_pen_up_time) >= self.debounce_timeout:
                should_recognize = True
                self.last_pen_up_time = None  # fire once

        result.should_recognize = should_recognize

        # Format strokes for visualization
        result.current_stroke = [(p.x, p.y) for p in self.current_stroke.points]
        result.all_strokes = [
            [(p.x, p.y) for p in s.points] for s in self.completed_strokes
        ]
        if result.current_stroke:
            result.all_strokes.append(result.current_stroke)

        return result

    def process_frame(self, frame_bgr: np.ndarray) -> FrameResult:
        """Processes a single BGR OpenCV frame from webcam."""
        now = time.perf_counter()
        if self.hands_detector is None:
            self._init_mediapipe(0.7, 0.7)

        if self.hands_detector is None:
            return FrameResult(hand_detected=False)

        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_results = self.hands_detector.process(rgb)

        if not mp_results.multi_hand_landmarks:
            # Pen up if hand lost
            if self.is_currently_pen_down:
                self.is_currently_pen_down = False
                if len(self.current_stroke) >= 4:
                    self.completed_strokes.append(self.current_stroke)
                self.current_stroke = Stroke()
                self.last_pen_up_time = now

            should_recognize = False
            if (
                self.last_pen_up_time is not None
                and len(self.completed_strokes) > 0
                and (now - self.last_pen_up_time) >= self.debounce_timeout
            ):
                should_recognize = True
                self.last_pen_up_time = None

            return FrameResult(
                hand_detected=False,
                should_recognize=should_recognize,
                all_strokes=[[(p.x, p.y) for p in s.points] for s in self.completed_strokes],
            )

        landmarks = mp_results.multi_hand_landmarks[0].landmark
        return self.process_landmarks(landmarks, timestamp=now)
