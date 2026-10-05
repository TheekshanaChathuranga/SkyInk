"""Trajectory data structures and manipulation utilities."""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Point:
    """A single tracking point."""
    x: float
    y: float
    z: float = 0.0
    timestamp: float = 0.0
    is_pen_down: bool = True

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, 1.0 if self.is_pen_down else 0.0], dtype=np.float32)


@dataclass
class Stroke:
    """A single continuous pen-down stroke."""
    points: List[Point] = field(default_factory=list)

    def add_point(self, point: Point):
        self.points.append(point)

    def __len__(self) -> int:
        return len(self.points)

    def to_numpy(self) -> np.ndarray:
        """Returns (N, 4) array: [x, y, z, pen_down]."""
        if not self.points:
            return np.empty((0, 4), dtype=np.float32)
        return np.array([p.to_array() for p in self.points], dtype=np.float32)

    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Returns (min_x, min_y, max_x, max_y)."""
        if not self.points:
            return 0.0, 0.0, 0.0, 0.0
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)

    def path_length(self) -> float:
        """Returns total Euclidean distance along the stroke."""
        if len(self.points) < 2:
            return 0.0
        coords = np.array([[p.x, p.y] for p in self.points], dtype=np.float32)
        diffs = np.diff(coords, axis=0)
        return float(np.sum(np.sqrt(np.sum(diffs**2, axis=1))))


@dataclass
class Trajectory:
    """Complete air-writing sample, composed of one or more strokes."""
    strokes: List[Stroke] = field(default_factory=list)
    label: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_stroke(self, stroke: Stroke):
        if len(stroke) > 0:
            self.strokes.append(stroke)

    def total_points(self) -> int:
        return sum(len(s) for s in self.strokes)

    def is_empty(self) -> bool:
        return self.total_points() == 0

    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Returns overall bounding box (min_x, min_y, max_x, max_y)."""
        if self.is_empty():
            return 0.0, 0.0, 0.0, 0.0
        all_x = [p.x for s in self.strokes for p in s.points]
        all_y = [p.y for s in self.strokes for p in s.points]
        return min(all_x), min(all_y), max(all_x), max(all_y)

    def to_points_list(self) -> List[Point]:
        """Flattens all points across strokes with proper pen-down state."""
        pts = []
        for s in self.strokes:
            for p in s.points:
                pts.append(p)
        return pts

    def to_numpy(self) -> np.ndarray:
        """Returns (N, 4) array: [x, y, z, pen_state].
        Strokes are concatenated. Last point of each stroke marks stroke boundary.
        """
        if self.is_empty():
            return np.empty((0, 4), dtype=np.float32)
        arrays = [s.to_numpy() for s in self.strokes if len(s) > 0]
        if not arrays:
            return np.empty((0, 4), dtype=np.float32)
        return np.concatenate(arrays, axis=0)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes trajectory to dictionary (JSON serializable)."""
        return {
            "label": self.label,
            "user_id": self.user_id,
            "metadata": self.metadata,
            "strokes": [
                [
                    {
                        "x": round(p.x, 5),
                        "y": round(p.y, 5),
                        "z": round(p.z, 5),
                        "t": round(p.timestamp, 4),
                        "p": int(p.is_pen_down),
                    }
                    for p in s.points
                ]
                for s in self.strokes
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trajectory":
        """Deserializes trajectory from dictionary."""
        traj = cls(
            label=data.get("label"),
            user_id=data.get("user_id"),
            metadata=data.get("metadata", {}),
        )
        for stroke_data in data.get("strokes", []):
            stroke = Stroke()
            for pt_data in stroke_data:
                stroke.add_point(
                    Point(
                        x=float(pt_data["x"]),
                        y=float(pt_data["y"]),
                        z=float(pt_data.get("z", 0.0)),
                        timestamp=float(pt_data.get("t", 0.0)),
                        is_pen_down=bool(pt_data.get("p", 1)),
                    )
                )
            if len(stroke) > 0:
                traj.add_stroke(stroke)
        return traj
