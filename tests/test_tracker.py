"""Unit tests for fingertip tracker, pinch detection, and stroke state machine."""
from types import SimpleNamespace
import pytest
from airscript.core.tracker import HandFingertipTracker


def create_mock_landmarks(pinch: bool = False, pointing: bool = False):
    """Creates a mock set of 21 MediaPipe hand landmarks."""
    # 21 landmarks default
    landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    # Wrist at (0.5, 0.8)
    landmarks[0] = SimpleNamespace(x=0.5, y=0.8, z=0.0)
    # Middle MCP at (0.5, 0.5) -> scale = 0.3
    landmarks[9] = SimpleNamespace(x=0.5, y=0.5, z=0.0)

    if pinch:
        # Index tip (8) and Thumb tip (4) very close
        landmarks[4] = SimpleNamespace(x=0.50, y=0.40, z=0.0)
        landmarks[8] = SimpleNamespace(x=0.51, y=0.41, z=0.0)
    else:
        # Thumb tip and index tip far apart
        landmarks[4] = SimpleNamespace(x=0.40, y=0.50, z=0.0)
        landmarks[8] = SimpleNamespace(x=0.60, y=0.30, z=0.0)

    if pointing:
        # Index tip extended (y < pip.y)
        landmarks[6] = SimpleNamespace(x=0.5, y=0.45, z=0.0)
        landmarks[8] = SimpleNamespace(x=0.5, y=0.30, z=0.0)
        # Other fingers folded (y > pip.y)
        landmarks[10] = SimpleNamespace(x=0.5, y=0.50, z=0.0)
        landmarks[12] = SimpleNamespace(x=0.5, y=0.60, z=0.0)
        landmarks[14] = SimpleNamespace(x=0.5, y=0.50, z=0.0)
        landmarks[16] = SimpleNamespace(x=0.5, y=0.60, z=0.0)
        landmarks[18] = SimpleNamespace(x=0.5, y=0.50, z=0.0)
        landmarks[20] = SimpleNamespace(x=0.5, y=0.60, z=0.0)

    return landmarks


def test_tracker_pinch_detection():
    tracker = HandFingertipTracker(pinch_threshold=0.15, gesture_mode="pinch")

    # Not pinched
    lm_open = create_mock_landmarks(pinch=False)
    res_open = tracker.process_landmarks(lm_open, timestamp=0.0)
    assert res_open.hand_detected
    assert not res_open.is_pen_down
    assert res_open.gesture == "hover"

    # Pinched (Pen down)
    lm_pinch = create_mock_landmarks(pinch=True)
    res_pinch = tracker.process_landmarks(lm_pinch, timestamp=0.05)
    assert res_pinch.hand_detected
    assert res_pinch.is_pen_down
    assert res_pinch.gesture == "writing"


def test_tracker_stroke_accumulation_and_debounce():
    tracker = HandFingertipTracker(debounce_timeout=0.3, gesture_mode="pinch")

    # Pen down: draw 6 points
    lm_pinch = create_mock_landmarks(pinch=True)
    for i in range(6):
        res = tracker.process_landmarks(lm_pinch, timestamp=i * 0.05)
        assert res.is_pen_down
        assert not res.should_recognize

    assert len(tracker.current_stroke) == 6

    # Pen up
    lm_open = create_mock_landmarks(pinch=False)
    res_up = tracker.process_landmarks(lm_open, timestamp=0.35)
    assert not res_up.is_pen_down
    assert len(tracker.completed_strokes) == 1
    assert not res_up.should_recognize

    # Wait until debounce timeout expires (> 0.3s)
    res_timeout = tracker.process_landmarks(lm_open, timestamp=0.70)
    assert res_timeout.should_recognize
    assert len(tracker.completed_strokes) == 1


def test_tracker_undo_and_clear():
    tracker = HandFingertipTracker()
    lm_pinch = create_mock_landmarks(pinch=True)
    for i in range(5):
        tracker.process_landmarks(lm_pinch, timestamp=i * 0.05)

    lm_open = create_mock_landmarks(pinch=False)
    tracker.process_landmarks(lm_open, timestamp=0.30)
    assert len(tracker.completed_strokes) == 1

    # Test undo
    assert tracker.undo_last_stroke() is True
    assert len(tracker.completed_strokes) == 0

    # Test reset
    tracker.reset_trajectory()
    assert tracker.get_current_trajectory().is_empty()
