"""Synthetic online handwriting trajectory generator and online stroke dataset loaders."""
import math
import os
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple
import numpy as np

from airscript.core.trajectory import Point, Stroke, Trajectory


def _bezier_curve(p0: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float], num_pts: int = 25) -> List[Tuple[float, float]]:
    """Evaluates quadratic Bezier curve."""
    t = np.linspace(0.0, 1.0, num_pts)
    bx = (1 - t)**2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0]
    by = (1 - t)**2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1]
    return list(zip(bx.tolist(), by.tolist()))


def _line(p0: Tuple[float, float], p1: Tuple[float, float], num_pts: int = 15) -> List[Tuple[float, float]]:
    """Evaluates linear segment."""
    t = np.linspace(0.0, 1.0, num_pts)
    lx = p0[0] + t * (p1[0] - p0[0])
    ly = p0[1] + t * (p1[1] - p0[1])
    return list(zip(lx.tolist(), ly.tolist()))


# Canonical stroke templates in normalized [-0.5, 0.5] coordinates
# Each character is a list of strokes, where each stroke is a list of segments
CHAR_TEMPLATES: Dict[str, List[List[List[Tuple[float, float]]]]] = {
    # Digits
    "0": [[_bezier_curve((-0.3, 0.0), (-0.3, 0.45), (0.0, 0.45), 15) +
           _bezier_curve((0.0, 0.45), (0.3, 0.45), (0.3, 0.0), 15) +
           _bezier_curve((0.3, 0.0), (0.3, -0.45), (0.0, -0.45), 15) +
           _bezier_curve((0.0, -0.45), (-0.3, -0.45), (-0.3, 0.0), 15)]],
    "1": [[_line((-0.1, 0.3), (0.0, 0.45), 10) + _line((0.0, 0.45), (0.0, -0.45), 25)]],
    "2": [[_bezier_curve((-0.25, 0.25), (0.0, 0.45), (0.25, 0.25), 15) +
           _line((0.25, 0.25), (-0.25, -0.45), 20) +
           _line((-0.25, -0.45), (0.25, -0.45), 15)]],
    "3": [[_bezier_curve((-0.25, 0.35), (0.25, 0.35), (0.0, 0.0), 20) +
           _bezier_curve((0.0, 0.0), (0.3, -0.2), (-0.25, -0.45), 25)]],
    "4": [[_line((0.2, -0.45), (0.2, 0.45), 25)],
          [_line((-0.25, 0.45), (-0.25, 0.0), 15) + _line((-0.25, 0.0), (0.3, 0.0), 15)]],
    "5": [[_line((0.2, 0.45), (-0.2, 0.45), 12) +
           _line((-0.2, 0.45), (-0.2, 0.05), 12) +
           _bezier_curve((-0.2, 0.05), (0.25, 0.05), (0.2, -0.25), 15) +
           _bezier_curve((0.2, -0.25), (0.1, -0.45), (-0.2, -0.45), 15)]],
    "6": [[_bezier_curve((0.2, 0.4), (-0.25, 0.2), (-0.25, -0.2), 20) +
           _bezier_curve((-0.25, -0.2), (-0.25, -0.45), (0.0, -0.45), 15) +
           _bezier_curve((0.0, -0.45), (0.25, -0.45), (0.25, -0.15), 15) +
           _bezier_curve((0.25, -0.15), (0.1, 0.0), (-0.25, -0.15), 15)]],
    "7": [[_line((-0.25, 0.45), (0.25, 0.45), 15) + _line((0.25, 0.45), (-0.1, -0.45), 25)]],
    "8": [[_bezier_curve((0.0, 0.45), (0.25, 0.25), (0.0, 0.0), 15) +
           _bezier_curve((0.0, 0.0), (-0.25, -0.25), (0.0, -0.45), 15) +
           _bezier_curve((0.0, -0.45), (0.25, -0.25), (0.0, 0.0), 15) +
           _bezier_curve((0.0, 0.0), (-0.25, 0.25), (0.0, 0.45), 15)]],
    "9": [[_bezier_curve((0.2, 0.1), (0.0, 0.45), (-0.2, 0.2), 15) +
           _bezier_curve((-0.2, 0.2), (-0.2, 0.0), (0.2, 0.0), 15) +
           _line((0.2, 0.1), (0.2, -0.45), 25)]],

    # Uppercase Letters (A-Z)
    "A": [[_line((-0.25, -0.45), (0.0, 0.45), 20) + _line((0.0, 0.45), (0.25, -0.45), 20)],
          [_line((-0.15, -0.05), (0.15, -0.05), 12)]],
    "B": [[_line((-0.25, -0.45), (-0.25, 0.45), 25)],
          [_bezier_curve((-0.25, 0.45), (0.2, 0.35), (-0.25, 0.0), 20) +
           _bezier_curve((-0.25, 0.0), (0.25, -0.15), (-0.25, -0.45), 20)]],
    "C": [[_bezier_curve((0.25, 0.35), (-0.25, 0.45), (-0.25, 0.0), 20) +
           _bezier_curve((-0.25, 0.0), (-0.25, -0.45), (0.25, -0.35), 20)]],
    "D": [[_line((-0.25, -0.45), (-0.25, 0.45), 25)],
          [_bezier_curve((-0.25, 0.45), (0.35, 0.0), (-0.25, -0.45), 35)]],
    "E": [[_line((0.25, 0.45), (-0.25, 0.45), 15) +
           _line((-0.25, 0.45), (-0.25, -0.45), 25) +
           _line((-0.25, -0.45), (0.25, -0.45), 15)],
          [_line((-0.25, 0.0), (0.1, 0.0), 12)]],
    "F": [[_line((0.25, 0.45), (-0.25, 0.45), 15) +
           _line((-0.25, 0.45), (-0.25, -0.45), 25)],
          [_line((-0.25, 0.05), (0.1, 0.05), 12)]],
    "G": [[_bezier_curve((0.25, 0.35), (-0.25, 0.45), (-0.25, 0.0), 20) +
           _bezier_curve((-0.25, 0.0), (-0.25, -0.45), (0.2, -0.45), 20) +
           _line((0.2, -0.45), (0.2, -0.1), 12) +
           _line((0.2, -0.1), (0.0, -0.1), 10)]],
    "H": [[_line((-0.25, 0.45), (-0.25, -0.45), 20)],
          [_line((0.25, 0.45), (0.25, -0.45), 20)],
          [_line((-0.25, 0.0), (0.25, 0.0), 12)]],
    "I": [[_line((0.0, 0.45), (0.0, -0.45), 25)],
          [_line((-0.15, 0.45), (0.15, 0.45), 10)],
          [_line((-0.15, -0.45), (0.15, -0.45), 10)]],
    "J": [[_line((0.15, 0.45), (0.15, -0.2), 20) +
           _bezier_curve((0.15, -0.2), (0.15, -0.45), (-0.15, -0.35), 15)]],
    "K": [[_line((-0.25, 0.45), (-0.25, -0.45), 25)],
          [_line((0.25, 0.4), (-0.25, 0.0), 15) + _line((-0.25, 0.0), (0.25, -0.45), 15)]],
    "L": [[_line((-0.25, 0.45), (-0.25, -0.45), 20) + _line((-0.25, -0.45), (0.25, -0.45), 15)]],
    "M": [[_line((-0.3, -0.45), (-0.3, 0.45), 15) +
           _line((-0.3, 0.45), (0.0, -0.1), 15) +
           _line((0.0, -0.1), (0.3, 0.45), 15) +
           _line((0.3, 0.45), (0.3, -0.45), 15)]],
    "N": [[_line((-0.25, -0.45), (-0.25, 0.45), 18) +
           _line((-0.25, 0.45), (0.25, -0.45), 22) +
           _line((0.25, -0.45), (0.25, 0.45), 18)]],
    "O": [[_bezier_curve((0.0, 0.45), (-0.3, 0.45), (-0.3, 0.0), 15) +
           _bezier_curve((-0.3, 0.0), (-0.3, -0.45), (0.0, -0.45), 15) +
           _bezier_curve((0.0, -0.45), (0.3, -0.45), (0.3, 0.0), 15) +
           _bezier_curve((0.3, 0.0), (0.3, 0.45), (0.0, 0.45), 15)]],
    "P": [[_line((-0.25, -0.45), (-0.25, 0.45), 25)],
          [_bezier_curve((-0.25, 0.45), (0.25, 0.45), (0.25, 0.1), 15) +
           _bezier_curve((0.25, 0.1), (0.25, 0.0), (-0.25, 0.0), 15)]],
    "Q": [[_bezier_curve((0.0, 0.45), (-0.3, 0.45), (-0.3, 0.0), 15) +
           _bezier_curve((-0.3, 0.0), (-0.3, -0.45), (0.0, -0.45), 15) +
           _bezier_curve((0.0, -0.45), (0.3, -0.45), (0.3, 0.0), 15) +
           _bezier_curve((0.3, 0.0), (0.3, 0.45), (0.0, 0.45), 15)],
          [_line((0.05, -0.15), (0.3, -0.45), 12)]],
    "R": [[_line((-0.25, -0.45), (-0.25, 0.45), 25)],
          [_bezier_curve((-0.25, 0.45), (0.25, 0.45), (0.25, 0.08), 15) +
           _bezier_curve((0.25, 0.08), (0.25, 0.0), (-0.25, 0.0), 15) +
           _line((-0.05, 0.0), (0.25, -0.45), 15)]],
    "S": [[_bezier_curve((0.2, 0.35), (0.0, 0.45), (-0.15, 0.35), 15) +
           _bezier_curve((-0.15, 0.35), (-0.25, 0.1), (0.0, 0.0), 15) +
           _bezier_curve((0.0, 0.0), (0.25, -0.1), (0.15, -0.35), 15) +
           _bezier_curve((0.15, -0.35), (0.0, -0.45), (-0.2, -0.35), 15)]],
    "T": [[_line((-0.25, 0.45), (0.25, 0.45), 15)],
          [_line((0.0, 0.45), (0.0, -0.45), 25)]],
    "U": [[_bezier_curve((-0.25, 0.45), (-0.25, -0.45), (0.0, -0.45), 20) +
           _bezier_curve((0.0, -0.45), (0.25, -0.45), (0.25, 0.45), 20)]],
    "V": [[_line((-0.25, 0.45), (0.0, -0.45), 20) + _line((0.0, -0.45), (0.25, 0.45), 20)]],
    "W": [[_line((-0.3, 0.45), (-0.15, -0.45), 15) +
           _line((-0.15, -0.45), (0.0, 0.1), 15) +
           _line((0.0, 0.1), (0.15, -0.45), 15) +
           _line((0.15, -0.45), (0.3, 0.45), 15)]],
    "X": [[_line((-0.25, 0.45), (0.25, -0.45), 25)],
          [_line((0.25, 0.45), (-0.25, -0.45), 25)]],
    "Y": [[_line((-0.25, 0.45), (0.0, 0.0), 15) +
           _line((0.25, 0.45), (0.0, 0.0), 15) +
           _line((0.0, 0.0), (0.0, -0.45), 15)]],
    "Z": [[_line((-0.25, 0.45), (0.25, 0.45), 15) +
           _line((0.25, 0.45), (-0.25, -0.45), 25) +
           _line((-0.25, -0.45), (0.25, -0.45), 15)]],

    # Lowercase Letters (a-z)
    "a": [[_bezier_curve((0.2, 0.1), (-0.2, 0.2), (-0.2, -0.1), 15) +
           _bezier_curve((-0.2, -0.1), (-0.2, -0.35), (0.2, -0.35), 15) +
           _line((0.2, 0.15), (0.2, -0.35), 20)]],
    "b": [[_line((-0.2, 0.45), (-0.2, -0.35), 25) +
           _bezier_curve((-0.2, -0.35), (0.25, -0.35), (0.2, -0.05), 15) +
           _bezier_curve((0.2, -0.05), (0.2, 0.15), (-0.2, 0.1), 15)]],
    "c": [[_bezier_curve((0.2, 0.15), (-0.2, 0.2), (-0.2, -0.1), 15) +
           _bezier_curve((-0.2, -0.1), (-0.2, -0.35), (0.2, -0.3), 15)]],
    "d": [[_bezier_curve((0.2, 0.1), (-0.2, 0.2), (-0.2, -0.1), 15) +
           _bezier_curve((-0.2, -0.1), (-0.2, -0.35), (0.2, -0.35), 15) +
           _line((0.2, 0.45), (0.2, -0.35), 30)]],
    "e": [[_line((-0.2, -0.05), (0.2, -0.05), 12) +
           _bezier_curve((0.2, -0.05), (0.2, 0.2), (-0.2, 0.1), 15) +
           _bezier_curve((-0.2, 0.1), (-0.2, -0.35), (0.2, -0.3), 15)]],
    "f": [[_bezier_curve((0.15, 0.45), (0.0, 0.45), (-0.05, 0.3), 12) +
           _line((-0.05, 0.3), (-0.05, -0.35), 25)],
          [_line((-0.18, 0.15), (0.12, 0.15), 10)]],
    "g": [[_bezier_curve((0.2, 0.1), (-0.2, 0.2), (-0.2, -0.1), 15) +
           _bezier_curve((-0.2, -0.1), (-0.2, -0.35), (0.2, -0.35), 15) +
           _line((0.2, 0.15), (0.2, -0.5), 25) +
           _bezier_curve((0.2, -0.5), (0.0, -0.6), (-0.2, -0.45), 15)]],
    "h": [[_line((-0.2, 0.45), (-0.2, -0.35), 25) +
           _bezier_curve((-0.2, 0.0), (0.1, 0.15), (0.2, -0.05), 15) +
           _line((0.2, -0.05), (0.2, -0.35), 15)]],
    "i": [[_line((0.0, 0.15), (0.0, -0.35), 20)],
          [_line((0.0, 0.35), (0.0, 0.38), 5)]],
    "j": [[_line((0.1, 0.15), (0.1, -0.5), 20) +
           _bezier_curve((0.1, -0.5), (0.0, -0.6), (-0.15, -0.45), 12)],
          [_line((0.1, 0.35), (0.1, 0.38), 5)]],
    "k": [[_line((-0.2, 0.45), (-0.2, -0.35), 25)],
          [_line((0.18, 0.15), (-0.2, -0.1), 15) + _line((-0.2, -0.1), (0.2, -0.35), 15)]],
    "l": [[_line((0.0, 0.45), (0.0, -0.35), 25)]],
    "m": [[_line((-0.25, 0.15), (-0.25, -0.35), 18) +
           _bezier_curve((-0.25, 0.05), (-0.1, 0.2), (0.0, -0.05), 15) +
           _line((0.0, -0.05), (0.0, -0.35), 12) +
           _bezier_curve((0.0, 0.05), (0.15, 0.2), (0.25, -0.05), 15) +
           _line((0.25, -0.05), (0.25, -0.35), 12)]],
    "n": [[_line((-0.2, 0.15), (-0.2, -0.35), 20) +
           _bezier_curve((-0.2, 0.05), (0.05, 0.2), (0.2, -0.05), 15) +
           _line((0.2, -0.05), (0.2, -0.35), 15)]],
    "o": [[_bezier_curve((0.0, 0.15), (-0.25, 0.15), (-0.25, -0.1), 15) +
           _bezier_curve((-0.25, -0.1), (-0.25, -0.35), (0.0, -0.35), 15) +
           _bezier_curve((0.0, -0.35), (0.25, -0.35), (0.25, -0.1), 15) +
           _bezier_curve((0.25, -0.1), (0.25, 0.15), (0.0, 0.15), 15)]],
    "p": [[_line((-0.2, 0.15), (-0.2, -0.55), 25) +
           _bezier_curve((-0.2, 0.15), (0.25, 0.15), (0.2, -0.1), 15) +
           _bezier_curve((0.2, -0.1), (0.25, -0.35), (-0.2, -0.35), 15)]],
    "q": [[_bezier_curve((0.2, 0.1), (-0.2, 0.2), (-0.2, -0.1), 15) +
           _bezier_curve((-0.2, -0.1), (-0.2, -0.35), (0.2, -0.35), 15) +
           _line((0.2, 0.15), (0.2, -0.55), 25)]],
    "r": [[_line((-0.2, 0.15), (-0.2, -0.35), 20) +
           _bezier_curve((-0.2, 0.05), (0.0, 0.2), (0.2, 0.1), 15)]],
    "s": [[_bezier_curve((0.15, 0.12), (-0.15, 0.18), (0.0, 0.0), 15) +
           _bezier_curve((0.0, 0.0), (0.2, -0.1), (-0.15, -0.3), 15)]],
    "t": [[_line((0.0, 0.35), (0.0, -0.35), 25)],
          [_line((-0.15, 0.15), (0.15, 0.15), 10)]],
    "u": [[_bezier_curve((-0.2, 0.15), (-0.2, -0.35), (0.0, -0.35), 15) +
           _bezier_curve((0.0, -0.35), (0.2, -0.35), (0.2, 0.15), 15) +
           _line((0.2, 0.15), (0.2, -0.35), 12)]],
    "v": [[_line((-0.2, 0.15), (0.0, -0.35), 15) + _line((0.0, -0.35), (0.2, 0.15), 15)]],
    "w": [[_line((-0.25, 0.15), (-0.12, -0.35), 12) +
           _line((-0.12, -0.35), (0.0, 0.0), 12) +
           _line((0.0, 0.0), (0.12, -0.35), 12) +
           _line((0.12, -0.35), (0.25, 0.15), 12)]],
    "x": [[_line((-0.2, 0.15), (0.2, -0.35), 18)],
          [_line((0.2, 0.15), (-0.2, -0.35), 18)]],
    "y": [[_line((-0.2, 0.15), (0.0, -0.1), 12) +
           _line((0.2, 0.15), (-0.1, -0.5), 20)]],
    "z": [[_line((-0.2, 0.15), (0.2, 0.15), 12) +
           _line((0.2, 0.15), (-0.2, -0.35), 18) +
           _line((-0.2, -0.35), (0.2, -0.35), 12)]],
}


class SyntheticStrokeGenerator:
    """Generates synthetic trajectories for single characters and continuous words."""

    def __init__(self, char_templates: Optional[Dict] = None):
        self.templates = char_templates or CHAR_TEMPLATES
        self.alphabet = sorted(list(self.templates.keys()))
        self.char_to_id = {c: i for i, c in enumerate(self.alphabet)}
        self.id_to_char = {i: c for i, c in enumerate(self.alphabet)}

    def generate_character(
        self,
        char: str,
        user_id: str = "synthetic",
        jitter_std: float = 0.005,
    ) -> Trajectory:
        """Generates a synthetic air-writing trajectory for a given character."""
        if char not in self.templates:
            raise ValueError(f"Character '{char}' not found in templates.")

        stroke_groups = self.templates[char]
        traj = Trajectory(label=char, user_id=user_id)

        t_sim = 0.0
        for stroke_pts in stroke_groups:
            # Flatten if stroke_pts is wrapped in an extra list
            if len(stroke_pts) > 0 and isinstance(stroke_pts[0], list):
                flattened_pts = [p for sub in stroke_pts for p in sub]
            else:
                flattened_pts = stroke_pts

            stroke = Stroke()
            # Fluctuate stroke speed
            dt_step = random.uniform(0.015, 0.035)
            for pt in flattened_pts:
                # Add natural motor noise
                x = float(pt[0]) + random.gauss(0.0, jitter_std)
                y = float(pt[1]) + random.gauss(0.0, jitter_std)
                stroke.add_point(Point(x=x, y=y, z=0.0, timestamp=t_sim, is_pen_down=True))
                t_sim += dt_step
            traj.add_stroke(stroke)
            # Pen-up delay between multi-stroke components
            t_sim += random.uniform(0.1, 0.25)

        return traj

    def generate_word(
        self,
        word: str,
        user_id: str = "synthetic",
        char_spacing: float = 0.65,
    ) -> Trajectory:
        """Generates continuous word trajectory concatenating letters horizontally."""
        traj = Trajectory(label=word, user_id=user_id)
        current_x_offset = 0.0
        t_sim = 0.0

        for char in word:
            if char not in self.templates:
                continue
            char_traj = self.generate_character(char, user_id=user_id)
            for stroke in char_traj.strokes:
                word_stroke = Stroke()
                for pt in stroke.points:
                    word_stroke.add_point(
                        Point(
                            x=pt.x + current_x_offset,
                            y=pt.y,
                            z=pt.z,
                            timestamp=t_sim + pt.timestamp,
                            is_pen_down=True,
                        )
                    )
                traj.add_stroke(word_stroke)

            current_x_offset += char_spacing + random.uniform(-0.05, 0.05)
            t_sim += 0.3

        return traj

    def generate_balanced_dataset(
        self,
        samples_per_class: int = 50,
        augmenter=None,
    ) -> List[Trajectory]:
        """Generates a balanced dataset of all 62 alphanumeric characters."""
        dataset: List[Trajectory] = []
        for char in self.alphabet:
            for s_idx in range(samples_per_class):
                user = f"user_{s_idx % 5}"
                traj = self.generate_character(char, user_id=user)
                dataset.append(traj)
        return dataset
