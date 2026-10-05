"""AirScript dataset, augmentations, and trajectory generation."""
from .augmentations import (
    CameraTremor,
    JitterNoise,
    PointDropout,
    RandomRotation,
    RandomScale,
    RandomShear,
    SpeedWarping,
    TrajectoryAugmenter,
)
from .dataset import (
    AirScriptCharDataset,
    AirScriptWordDataset,
    Vocabulary,
    ctc_collate_fn,
    load_dataset_npz,
    save_dataset_npz,
    split_cross_user,
    split_per_user,
)
from .synthetic import CHAR_TEMPLATES, SyntheticStrokeGenerator

__all__ = [
    "TrajectoryAugmenter",
    "RandomRotation",
    "RandomScale",
    "RandomShear",
    "JitterNoise",
    "SpeedWarping",
    "PointDropout",
    "CameraTremor",
    "AirScriptCharDataset",
    "AirScriptWordDataset",
    "Vocabulary",
    "ctc_collate_fn",
    "split_per_user",
    "split_cross_user",
    "save_dataset_npz",
    "load_dataset_npz",
    "SyntheticStrokeGenerator",
    "CHAR_TEMPLATES",
]
