"""Trajectory preprocessing, resampling, normalization, and feature extraction."""
import math
from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.signal import savgol_filter

from .trajectory import Point, Stroke, Trajectory


class TrajectoryPreprocessor:
    """Preprocesses 2D/3D air-writing trajectories for sequence neural networks."""

    def __init__(
        self,
        target_points: int = 64,
        resample_spacing: float = 0.015,
        apply_smoothing: bool = True,
        savgol_window: int = 5,
        savgol_polyorder: int = 2,
    ):
        self.target_points = target_points
        self.resample_spacing = resample_spacing
        self.apply_smoothing = apply_smoothing
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder

    def resample_equidistant(
        self,
        points: np.ndarray,
        num_points: Optional[int] = None,
    ) -> np.ndarray:
        """Resamples polyline to equidistant spatial points along path length.

        Args:
            points: (N, 2) or (N, D) array of points.
            num_points: target number of resampled points (defaults to self.target_points).
        Returns:
            (num_points, D) array of equidistant points.
        """
        N = len(points)
        num_points = num_points or self.target_points
        if N == 0:
            return np.zeros((num_points, points.shape[1] if points.ndim > 1 else 2), dtype=np.float32)
        if N == 1:
            return np.repeat(points[:1], num_points, axis=0).astype(np.float32)

        # Compute cumulative distance along the curve
        diffs = np.diff(points[:, :2], axis=0)
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        cum_dist = np.insert(np.cumsum(dists), 0, 0.0)
        total_len = cum_dist[-1]

        if total_len < 1e-6:
            return np.repeat(points[:1], num_points, axis=0).astype(np.float32)

        # Query distances
        query_dists = np.linspace(0.0, total_len, num_points)

        resampled = np.zeros((num_points, points.shape[1]), dtype=np.float32)
        for col in range(points.shape[1]):
            resampled[:, col] = np.interp(query_dists, cum_dist, points[:, col])

        return resampled

    def smooth_stroke(self, points: np.ndarray) -> np.ndarray:
        """Applies Savitzky-Golay filter to smooth jitter without losing sharp corners."""
        if not self.apply_smoothing or len(points) < self.savgol_window:
            return points

        window = min(self.savgol_window, len(points))
        if window % 2 == 0:
            window -= 1
        if window <= self.savgol_polyorder:
            return points

        smoothed = points.copy()
        for col in range(min(2, points.shape[1])):
            smoothed[:, col] = savgol_filter(points[:, col], window_length=window, polyorder=self.savgol_polyorder)
        return smoothed

    def normalize(self, points: np.ndarray) -> np.ndarray:
        """Centers trajectory at origin and normalizes height/width to [-0.5, 0.5]. Preserves aspect ratio."""
        if len(points) == 0:
            return points

        normed = points.copy()
        xy = normed[:, :2]

        min_xy = np.min(xy, axis=0)
        max_xy = np.max(xy, axis=0)
        box_size = max_xy - min_xy
        max_dim = max(float(np.max(box_size)), 1e-4)

        # Center at origin
        center = (min_xy + max_xy) / 2.0
        normed[:, :2] = (xy - center) / max_dim

        return normed

    def compute_features(self, points: np.ndarray) -> np.ndarray:
        """Extracts dynamic sequence features per point:
        Features: [x, y, dx, dy, speed, sin_theta, cos_theta, curvature].

        Args:
            points: (N, 2) or (N, D) normalized points.
        Returns:
            (N, 8) feature matrix.
        """
        N = len(points)
        if N == 0:
            return np.empty((0, 8), dtype=np.float32)

        x = points[:, 0]
        y = points[:, 1]

        # First derivatives (dx, dy)
        dx = np.zeros(N, dtype=np.float32)
        dy = np.zeros(N, dtype=np.float32)
        if N > 1:
            dx[1:] = np.diff(x)
            dy[1:] = np.diff(y)
            dx[0] = dx[1]
            dy[0] = dy[1]

        # Tangent speed
        speed = np.sqrt(dx**2 + dy**2)

        # Tangent angle
        theta = np.arctan2(dy, dx)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Curvature: rate of change of direction angle dtheta / ds
        dtheta = np.zeros(N, dtype=np.float32)
        if N > 1:
            dtheta[1:] = np.diff(theta)
            # Unwrap angle difference to [-pi, pi]
            dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
            dtheta[0] = dtheta[1]

        denom = np.maximum(speed, 1e-4)
        curvature = dtheta / denom
        # Clip extreme curvature to prevent numerical spikes
        curvature = np.clip(curvature, -10.0, 10.0)

        features = np.column_stack([
            x,
            y,
            dx,
            dy,
            speed,
            sin_theta,
            cos_theta,
            curvature,
        ]).astype(np.float32)

        return features

    def process(
        self,
        trajectory: Union[Trajectory, List[List[Tuple[float, float]]], np.ndarray],
        target_points: Optional[int] = None,
    ) -> np.ndarray:
        """Full preprocessing pipeline:
        1. Extract (x, y) coordinates from trajectory
        2. Arc-length resampling to target_points
        3. Smoothing
        4. Coordinate normalization (scale & translation invariant)
        5. Feature computation (dx, dy, speed, sin, cos, curvature)

        Returns:
            (target_points, 8) float32 feature tensor.
        """
        num_pts = target_points or self.target_points

        # Format input into (M, 2) array
        if isinstance(trajectory, Trajectory):
            if trajectory.is_empty():
                return np.zeros((num_pts, 8), dtype=np.float32)
            pts = trajectory.to_points_list()
            raw_xy = np.array([[p.x, p.y] for p in pts], dtype=np.float32)
        elif isinstance(trajectory, list):
            if not trajectory:
                return np.zeros((num_pts, 8), dtype=np.float32)
            pts = []
            for item in trajectory:
                if isinstance(item, list):
                    for pt in item:
                        pts.append([pt[0], pt[1]])
                elif isinstance(item, (tuple, np.ndarray)):
                    pts.append([item[0], item[1]])
            if not pts:
                return np.zeros((num_pts, 8), dtype=np.float32)
            raw_xy = np.array(pts, dtype=np.float32)
        elif isinstance(trajectory, np.ndarray):
            if len(trajectory) == 0:
                return np.zeros((num_pts, 8), dtype=np.float32)
            raw_xy = trajectory[:, :2].astype(np.float32)
        else:
            return np.zeros((num_pts, 8), dtype=np.float32)

        # 1. Resample to equidistant spacing
        resampled = self.resample_equidistant(raw_xy, num_points=num_pts)

        # 2. Smooth
        smoothed = self.smooth_stroke(resampled)

        # 3. Normalize (translation & scale invariance)
        normed = self.normalize(smoothed)

        # 4. Extract kinematic and geometric features
        features = self.compute_features(normed)

        return features
