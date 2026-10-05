"""PyTorch Dataset classes for single character and continuous word air-writing."""
import json
from pathlib import Path
import random
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from airscript.core.trajectory import Trajectory
from airscript.core.preprocessor import TrajectoryPreprocessor
from airscript.dataset.augmentations import TrajectoryAugmenter


class Vocabulary:
    """Manages character mapping and CTC tokenization."""

    def __init__(self, include_space: bool = True):
        # Index 0 is reserved for CTC blank token
        self.blank_token = "<blank>"
        self.blank_id = 0

        # Alphanumeric characters
        digits = [str(i) for i in range(10)]
        upper = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
        lower = [chr(i) for i in range(ord("a"), ord("z") + 1)]

        chars = digits + upper + lower
        if include_space:
            chars.append(" ")

        self.char_list = [self.blank_token] + chars
        self.char_to_id = {c: i for i, c in enumerate(self.char_list)}
        self.id_to_char = {i: c for i, c in enumerate(self.char_list)}

    def __len__(self) -> int:
        return len(self.char_list)

    def encode(self, text: str) -> List[int]:
        """Encodes string into list of token IDs."""
        return [self.char_to_id[c] for c in text if c in self.char_to_id]

    def decode(self, token_ids: List[int], remove_blank: bool = True) -> str:
        """Decodes token IDs into string."""
        chars = []
        for tid in token_ids:
            if remove_blank and tid == self.blank_id:
                continue
            if tid in self.id_to_char:
                chars.append(self.id_to_char[tid])
        return "".join(chars)


class AirScriptCharDataset(Dataset):
    """PyTorch Dataset for single character air-writing recognition."""

    def __init__(
        self,
        trajectories: List[Trajectory],
        vocab: Optional[Vocabulary] = None,
        target_points: int = 64,
        augmenter: Optional[TrajectoryAugmenter] = None,
        preprocessor: Optional[TrajectoryPreprocessor] = None,
    ):
        self.trajectories = trajectories
        self.vocab = vocab or Vocabulary(include_space=False)
        self.target_points = target_points
        self.augmenter = augmenter
        self.preprocessor = preprocessor or TrajectoryPreprocessor(target_points=target_points)

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        traj = self.trajectories[idx]
        pts = traj.to_numpy()

        # Extract (x, y)
        if len(pts) == 0:
            xy = np.zeros((self.target_points, 2), dtype=np.float32)
        else:
            xy = pts[:, :2].copy()

        # Apply spatial/temporal augmentation if provided
        if self.augmenter is not None:
            xy = self.augmenter(xy)

        # Preprocess into (target_points, 8) feature matrix
        features = self.preprocessor.process(xy, target_points=self.target_points)

        # Target label ID
        label_char = traj.label if traj.label else "0"
        label_id = self.vocab.char_to_id.get(label_char, 1)

        user_id = traj.user_id or "unknown"
        return torch.from_numpy(features), label_id, user_id


class AirScriptWordDataset(Dataset):
    """PyTorch Dataset for continuous multi-character/word air-writing with CTC."""

    def __init__(
        self,
        trajectories: List[Trajectory],
        vocab: Optional[Vocabulary] = None,
        augmenter: Optional[TrajectoryAugmenter] = None,
        preprocessor: Optional[TrajectoryPreprocessor] = None,
        max_seq_len: int = 256,
    ):
        self.trajectories = trajectories
        self.vocab = vocab or Vocabulary(include_space=True)
        self.augmenter = augmenter
        self.preprocessor = preprocessor or TrajectoryPreprocessor()
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, List[int], str]:
        traj = self.trajectories[idx]
        pts = traj.to_numpy()

        if len(pts) == 0:
            xy = np.zeros((32, 2), dtype=np.float32)
        else:
            xy = pts[:, :2].copy()

        if self.augmenter is not None:
            xy = self.augmenter(xy)

        # Continuous writing: scale target points proportionally to path length
        # Approximate 16 points per character
        num_chars = len(traj.label) if traj.label else 1
        num_pts = min(max(num_chars * 24, 32), self.max_seq_len)
        features = self.preprocessor.process(xy, target_points=num_pts)

        label_ids = self.vocab.encode(traj.label or "")
        user_id = traj.user_id or "unknown"

        return features, label_ids, user_id


def ctc_collate_fn(batch: List[Tuple[np.ndarray, List[int], str]]):
    """Collate function for variable length trajectories with CTC loss.

    Pads feature sequences and collates target token sequences.
    """
    features_list, targets_list, user_ids = zip(*batch)

    # Sequence lengths
    seq_lengths = torch.tensor([len(f) for f in features_list], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in targets_list], dtype=torch.long)
    max_len = int(seq_lengths.max().item())
    feat_dim = features_list[0].shape[1]

    # Pad feature tensors
    padded_features = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    for i, f in enumerate(features_list):
        padded_features[i, :len(f), :] = torch.from_numpy(f)

    # Flatten targets for torch.nn.CTCLoss
    flat_targets = torch.tensor([t for seq in targets_list for t in seq], dtype=torch.long)

    return padded_features, flat_targets, seq_lengths, target_lengths, user_ids


def split_per_user(
    trajectories: List[Trajectory],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Trajectory], List[Trajectory], List[Trajectory]]:
    """Splits samples per-user so all users are represented in train, val, and test."""
    rng = random.Random(seed)
    user_buckets: Dict[str, List[Trajectory]] = {}
    for t in trajectories:
        u = t.user_id or "default"
        user_buckets.setdefault(u, []).append(t)

    train, val, test = [], [], []
    for u, samples in user_buckets.items():
        shuffled = samples.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(shuffled[:n_train])
        val.extend(shuffled[n_train:n_train + n_val])
        test.extend(shuffled[n_train + n_val:])

    return train, val, test


def split_cross_user(
    trajectories: List[Trajectory],
    test_users: List[str],
    val_users: Optional[List[str]] = None,
) -> Tuple[List[Trajectory], List[Trajectory], List[Trajectory]]:
    """Splits dataset across users to evaluate out-of-distribution user generalization."""
    val_users = val_users or []
    train, val, test = [], [], []

    for t in trajectories:
        u = t.user_id or "default"
        if u in test_users:
            test.append(t)
        elif u in val_users:
            val.append(t)
        else:
            train.append(t)

    return train, val, test


def save_dataset_npz(trajectories: List[Trajectory], output_path: Union[str, Path]):
    """Saves trajectory dataset to compressed NPZ archive."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = [t.to_dict() for t in trajectories]
    json_bytes = json.dumps(records).encode("utf-8")
    np.savez_compressed(output_path, data=np.frombuffer(json_bytes, dtype=np.uint8))


def load_dataset_npz(file_path: Union[str, Path]) -> List[Trajectory]:
    """Loads trajectory dataset from compressed NPZ archive."""
    with np.load(file_path) as archive:
        raw_bytes = archive["data"].tobytes()
        records = json.loads(raw_bytes.decode("utf-8"))
    return [Trajectory.from_dict(r) for r in records]
