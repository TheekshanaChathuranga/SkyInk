"""Unit tests for datasets, augmentations, vocabulary, and data loaders."""
import numpy as np
import pytest
import torch

from airscript.dataset.augmentations import (
    CameraTremor,
    JitterNoise,
    PointDropout,
    RandomRotation,
    RandomScale,
    RandomShear,
    SpeedWarping,
    TrajectoryAugmenter,
)
from airscript.dataset.dataset import (
    AirScriptCharDataset,
    AirScriptWordDataset,
    Vocabulary,
    ctc_collate_fn,
    load_dataset_npz,
    save_dataset_npz,
    split_cross_user,
    split_per_user,
)
from airscript.dataset.synthetic import SyntheticStrokeGenerator


def test_augmentations():
    pts = np.linspace(0, 1, 40)[:, None].repeat(2, axis=1).astype(np.float32)

    # Test each transform produces valid non-empty arrays with same shape
    rot = RandomRotation(max_angle_deg=20.0, prob=1.0)
    out_rot = rot(pts)
    assert out_rot.shape == pts.shape
    assert not np.isnan(out_rot).any()

    scale = RandomScale(min_scale=0.8, max_scale=1.2, prob=1.0)
    out_scale = scale(pts)
    assert out_scale.shape == pts.shape

    shear = RandomShear(max_shear=0.2, prob=1.0)
    out_shear = shear(pts)
    assert out_shear.shape == pts.shape

    jitter = JitterNoise(std=0.01, prob=1.0)
    out_jitter = jitter(pts)
    assert out_jitter.shape == pts.shape
    assert not np.allclose(pts, out_jitter)

    speed = SpeedWarping(warp_factor=0.3, prob=1.0)
    out_speed = speed(pts)
    assert out_speed.shape == pts.shape

    drop = PointDropout(drop_prob=0.1, prob=1.0)
    out_drop = drop(pts)
    assert out_drop.shape == pts.shape

    tremor = CameraTremor(max_amplitude=0.01, prob=1.0)
    out_tremor = tremor(pts)
    assert out_tremor.shape == pts.shape

    augmenter = TrajectoryAugmenter()
    out_all = augmenter(pts)
    assert out_all.shape == pts.shape


def test_vocabulary():
    vocab = Vocabulary(include_space=True)
    assert vocab.blank_id == 0
    assert len(vocab) == 64  # blank + 10 digits + 26 upper + 26 lower + space

    encoded = vocab.encode("Air123")
    decoded = vocab.decode(encoded)
    assert decoded == "Air123"


def test_char_dataset():
    gen = SyntheticStrokeGenerator()
    trajectories = [
        gen.generate_character("A", user_id="user_1"),
        gen.generate_character("B", user_id="user_2"),
        gen.generate_character("5", user_id="user_1"),
    ]

    dataset = AirScriptCharDataset(trajectories, target_points=64)
    assert len(dataset) == 3

    features, label_id, user_id = dataset[0]
    assert isinstance(features, torch.Tensor)
    assert features.shape == (64, 8)
    assert isinstance(label_id, int)
    assert user_id == "user_1"


def test_word_dataset_and_ctc_collate():
    gen = SyntheticStrokeGenerator()
    trajectories = [
        gen.generate_word("Hi", user_id="user_1"),
        gen.generate_word("Cat", user_id="user_2"),
    ]

    dataset = AirScriptWordDataset(trajectories)
    batch = [dataset[0], dataset[1]]

    padded_features, flat_targets, seq_lens, target_lens, user_ids = ctc_collate_fn(batch)

    assert padded_features.dim() == 3  # (batch, max_seq_len, 8)
    assert padded_features.shape[0] == 2
    assert padded_features.shape[2] == 8
    assert seq_lens.shape[0] == 2
    assert target_lens.tolist() == [2, 3]  # "Hi" is 2 chars, "Cat" is 3 chars
    assert len(flat_targets) == 5


def test_dataset_splitting():
    gen = SyntheticStrokeGenerator()
    trajectories = [
        gen.generate_character("A", user_id="user_1"),
        gen.generate_character("A", user_id="user_1"),
        gen.generate_character("B", user_id="user_2"),
        gen.generate_character("B", user_id="user_3"),
    ]

    # Cross user split
    train, val, test = split_cross_user(trajectories, test_users=["user_3"], val_users=["user_2"])
    assert len(test) == 1
    assert test[0].user_id == "user_3"
    assert len(val) == 1
    assert val[0].user_id == "user_2"
    assert len(train) == 2
    assert all(t.user_id == "user_1" for t in train)


def test_npz_serialization(tmp_path):
    gen = SyntheticStrokeGenerator()
    trajs = [gen.generate_character("X", user_id="tester")]
    file_path = tmp_path / "test_data.npz"

    save_dataset_npz(trajs, file_path)
    loaded = load_dataset_npz(file_path)

    assert len(loaded) == 1
    assert loaded[0].label == "X"
    assert loaded[0].user_id == "tester"
    assert loaded[0].total_points() == trajs[0].total_points()
