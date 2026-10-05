"""Unit tests for trajectory structures, One Euro Filter, and Preprocessor."""
import numpy as np
import pytest

from airscript.core.one_euro_filter import OneEuroFilter, PointFilter
from airscript.core.trajectory import Point, Stroke, Trajectory
from airscript.core.preprocessor import TrajectoryPreprocessor


def test_one_euro_filter():
    filt = OneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.05)
    # Warm up with constant signal
    for i in range(20):
        filt(0.5, timestamp=i * (1.0 / 30.0))
    # Inject high frequency jitter around 0.5
    jittered_vals = [0.52, 0.48, 0.53, 0.47, 0.51, 0.49]
    smoothed = [filt(v, timestamp=(20 + i) * (1.0 / 30.0)) for i, v in enumerate(jittered_vals)]
    # Filtered output variance should be significantly less than raw jitter variance
    assert np.var(smoothed) < np.var(jittered_vals)
    for s in smoothed:
        assert abs(s - 0.5) < 0.03


def test_point_filter():
    pf = PointFilter(min_cutoff=1.0, beta=0.05)
    fx, fy, fz = pf.filter(0.5, 0.5, 0.1, timestamp=0.0)
    assert 0.49 <= fx <= 0.51
    assert 0.49 <= fy <= 0.51
    assert 0.09 <= fz <= 0.11


def test_trajectory_and_stroke():
    stroke = Stroke()
    stroke.add_point(Point(0.1, 0.2, 0.0, 0.0, True))
    stroke.add_point(Point(0.3, 0.4, 0.0, 0.1, True))
    assert len(stroke) == 2
    assert stroke.path_length() > 0

    traj = Trajectory(label="A")
    traj.add_stroke(stroke)
    assert not traj.is_empty()
    assert traj.total_points() == 2

    # Serialization roundtrip
    d = traj.to_dict()
    traj_loaded = Trajectory.from_dict(d)
    assert traj_loaded.label == "A"
    assert traj_loaded.total_points() == 2


def test_preprocessor_resampling():
    preprocessor = TrajectoryPreprocessor(target_points=64)
    # Synthetic circle
    t = np.linspace(0, 2 * np.pi, 20)
    circle_pts = np.column_stack([np.cos(t), np.sin(t)])
    resampled = preprocessor.resample_equidistant(circle_pts, num_points=64)
    assert resampled.shape == (64, 2)

    # Arc-length intervals should be approximately equal
    diffs = np.diff(resampled, axis=0)
    step_lens = np.sqrt(np.sum(diffs**2, axis=1))
    assert np.std(step_lens) < 0.05


def test_preprocessor_features():
    preprocessor = TrajectoryPreprocessor(target_points=64)
    # Synthetic line from (0,0) to (1,1)
    line_pts = np.column_stack([np.linspace(0, 1, 30), np.linspace(0, 1, 30)])
    features = preprocessor.process(line_pts, target_points=64)

    assert features.shape == (64, 8)
    # Check that dx, dy, speed, sin, cos are finite numbers and not NaN
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()
    # Coordinates should be centered around 0
    assert abs(np.mean(features[:, :2])) < 0.5
