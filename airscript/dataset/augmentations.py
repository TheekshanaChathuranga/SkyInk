"""Heavy trajectory augmentations for air-writing sequence modeling.

Transforms:
- Rotation: random 2D rotation in [-max_angle, max_angle]
- Scale: independent or isotropic stretching in [min_scale, max_scale]
- Shear: affine shearing along X and Y axes
- Jitter: additive Gaussian noise
- Speed Warping: non-linear temporal resampling / pacing variation
- Random Point Dropout: simulating dropped tracking frames
- Camera Tremor: low-frequency periodic wobble
"""
import math
import random
from typing import Callable, List, Optional, Tuple, Union
import numpy as np


class TrajectoryTransform:
    """Base class for trajectory transforms."""

    def __call__(self, points: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RandomRotation(TrajectoryTransform):
    """Rotates trajectory coordinates (x, y) randomly around centroid."""

    def __init__(self, max_angle_deg: float = 15.0, prob: float = 0.8):
        self.max_angle_rad = math.radians(max_angle_deg)
        self.prob = prob

    def __call__(self, points: np.ndarray) -> np.ndarray:
        if random.random() > self.prob or len(points) == 0:
            return points

        angle = random.uniform(-self.max_angle_rad, self.max_angle_rad)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

        xy = points[:, :2]
        center = np.mean(xy, axis=0, keepdims=True)
        rotated_xy = (xy - center) @ rot_matrix.T + center

        out = points.copy()
        out[:, :2] = rotated_xy
        return out


class RandomScale(TrajectoryTransform):
    """Scales trajectory independently or isotropically along X and Y."""

    def __init__(
        self,
        min_scale: float = 0.8,
        max_scale: float = 1.2,
        aspect_ratio_variance: float = 0.15,
        prob: float = 0.8,
    ):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.aspect_ratio_variance = aspect_ratio_variance
        self.prob = prob

    def __call__(self, points: np.ndarray) -> np.ndarray:
        if random.random() > self.prob or len(points) == 0:
            return points

        base_scale = random.uniform(self.min_scale, self.max_scale)
        scale_x = base_scale * random.uniform(1.0 - self.aspect_ratio_variance, 1.0 + self.aspect_ratio_variance)
        scale_y = base_scale * random.uniform(1.0 - self.aspect_ratio_variance, 1.0 + self.aspect_ratio_variance)

        xy = points[:, :2]
        center = np.mean(xy, axis=0, keepdims=True)
        scaled_xy = (xy - center) * np.array([scale_x, scale_y], dtype=np.float32) + center

        out = points.copy()
        out[:, :2] = scaled_xy
        return out


class RandomShear(TrajectoryTransform):
    """Applies affine shear along horizontal and vertical axes."""

    def __init__(self, max_shear: float = 0.2, prob: float = 0.6):
        self.max_shear = max_shear
        self.prob = prob

    def __call__(self, points: np.ndarray) -> np.ndarray:
        if random.random() > self.prob or len(points) == 0:
            return points

        shear_x = random.uniform(-self.max_shear, self.max_shear)
        shear_y = random.uniform(-self.max_shear, self.max_shear)
        shear_mat = np.array([[1.0, shear_x], [shear_y, 1.0]], dtype=np.float32)

        xy = points[:, :2]
        center = np.mean(xy, axis=0, keepdims=True)
        sheared_xy = (xy - center) @ shear_mat.T + center

        out = points.copy()
        out[:, :2] = sheared_xy
        return out


class JitterNoise(TrajectoryTransform):
    """Injects high-frequency Gaussian jitter to simulate tracking sensor noise."""

    def __init__(self, std: float = 0.008, prob: float = 0.8):
        self.std = std
        self.prob = prob

    def __call__(self, points: np.ndarray) -> np.ndarray:
        if random.random() > self.prob or len(points) == 0:
            return points

        out = points.copy()
        noise = np.random.normal(0.0, self.std, size=(len(points), 2)).astype(np.float32)
        out[:, :2] += noise
        return out


class SpeedWarping(TrajectoryTransform):
    """Warps temporal pacing non-linearly to simulate varying writing speed."""

    def __init__(self, warp_factor: float = 0.3, prob: float = 0.7):
        self.warp_factor = warp_factor
        self.prob = prob

    def __call__(self, points: np.ndarray) -> np.ndarray:
        N = len(points)
        if random.random() > self.prob or N < 8:
            return points

        orig_t = np.linspace(0.0, 1.0, N)
        # Create monotonic non-linear time warping function using cumulative positive noise
        alpha = random.uniform(1.0 - self.warp_factor, 1.0 + self.warp_factor)
        noise = np.random.uniform(0.1, 1.9, N)
        noise = np.power(noise, alpha)
        warped_t = np.cumsum(noise)
        warped_t = (warped_t - warped_t[0]) / (warped_t[-1] - warped_t[0] + 1e-6)

        out = np.zeros_like(points)
        for col in range(points.shape[1]):
            out[:, col] = np.interp(orig_t, warped_t, points[:, col])

        return out.astype(np.float32)


class PointDropout(TrajectoryTransform):
    """Simulates dropped camera frames or occlusion by randomly dropping points."""

    def __init__(self, drop_prob: float = 0.08, prob: float = 0.5):
        self.drop_prob = drop_prob
        self.prob = prob

    def __call__(self, points: np.ndarray) -> np.ndarray:
        N = len(points)
        if random.random() > self.prob or N < 12:
            return points

        keep_mask = np.random.rand(N) > self.drop_prob
        # Always keep first and last point
        keep_mask[0] = True
        keep_mask[-1] = True

        if np.sum(keep_mask) < 4:
            return points

        dropped = points[keep_mask]
        # Resample back to original length
        orig_indices = np.linspace(0, 1, len(dropped))
        target_indices = np.linspace(0, 1, N)

        out = np.zeros_like(points)
        for col in range(points.shape[1]):
            out[:, col] = np.interp(target_indices, orig_indices, dropped[:, col])

        return out.astype(np.float32)


class CameraTremor(TrajectoryTransform):
    """Simulates webcam camera shaking / body sway with low-frequency sinusoidal wobble."""

    def __init__(self, max_amplitude: float = 0.015, prob: float = 0.6):
        self.max_amplitude = max_amplitude
        self.prob = prob

    def __call__(self, points: np.ndarray) -> np.ndarray:
        N = len(points)
        if random.random() > self.prob or N == 0:
            return points

        freq_x = random.uniform(1.0, 3.0)
        freq_y = random.uniform(1.0, 3.0)
        phase_x = random.uniform(0, 2 * math.pi)
        phase_y = random.uniform(0, 2 * math.pi)
        amp = random.uniform(0.005, self.max_amplitude)

        t = np.linspace(0, 1, N)
        wobble_x = amp * np.sin(2 * math.pi * freq_x * t + phase_x)
        wobble_y = amp * np.cos(2 * math.pi * freq_y * t + phase_y)

        out = points.copy()
        out[:, 0] += wobble_x.astype(np.float32)
        out[:, 1] += wobble_y.astype(np.float32)
        return out


class TrajectoryAugmenter:
    """Composes multiple trajectory augmentations into an end-to-end pipeline."""

    def __init__(
        self,
        enable_rotation: bool = True,
        enable_scale: bool = True,
        enable_shear: bool = True,
        enable_jitter: bool = True,
        enable_speed_warp: bool = True,
        enable_dropout: bool = True,
        enable_tremor: bool = True,
    ):
        transforms: List[TrajectoryTransform] = []
        if enable_rotation:
            transforms.append(RandomRotation(max_angle_deg=18.0, prob=0.85))
        if enable_scale:
            transforms.append(RandomScale(min_scale=0.8, max_scale=1.25, prob=0.85))
        if enable_shear:
            transforms.append(RandomShear(max_shear=0.2, prob=0.6))
        if enable_jitter:
            transforms.append(JitterNoise(std=0.008, prob=0.8))
        if enable_speed_warp:
            transforms.append(SpeedWarping(warp_factor=0.35, prob=0.7))
        if enable_dropout:
            transforms.append(PointDropout(drop_prob=0.1, prob=0.5))
        if enable_tremor:
            transforms.append(CameraTremor(max_amplitude=0.015, prob=0.6))

        self.transforms = transforms

    def __call__(self, points: np.ndarray) -> np.ndarray:
        augmented = points.copy()
        for t in self.transforms:
            augmented = t(augmented)
        return augmented
