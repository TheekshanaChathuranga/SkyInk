"""Configuration definitions and YAML loader for AirScript."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class TrackerConfig:
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.7
    max_num_hands: int = 1
    # Pinch detection: normalized distance between thumb tip (4) and index tip (8)
    pinch_threshold: float = 0.06
    # One Euro filter params
    filter_min_cutoff: float = 1.0
    filter_beta: float = 0.05
    filter_d_cutoff: float = 1.0


@dataclass
class PreprocessingConfig:
    target_points: int = 64  # Standard fixed length for single character
    min_stroke_points: int = 6
    resample_spacing: float = 0.015
    smoothing_window: int = 5
    smoothing_polyorder: int = 2
    feature_names: List[str] = field(
        default_factory=lambda: ["x", "y", "dx", "dy", "speed", "sin_theta", "cos_theta", "curvature"]
    )


@dataclass
class ModelConfig:
    name: str = "cnn_gru"  # 'cnn_gru' or 'transformer_ctc'
    input_dim: int = 8
    num_classes: int = 63  # 26 lowercase + 26 uppercase + 10 digits + 1 blank/pad
    # CNN-GRU hyperparameters
    cnn_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    gru_hidden_size: int = 128
    gru_num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    # Transformer hyperparameters
    d_model: int = 128
    nhead: int = 4
    num_encoder_layers: int = 3
    dim_feedforward: int = 256


@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    patience: int = 7
    checkpoint_dir: str = "checkpoints"
    device: str = "auto"  # 'cpu', 'cuda', 'auto'


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    debounce_timeout_ms: int = 800  # Timeout after pen-up to trigger word recognition
    confidence_threshold: float = 0.35


@dataclass
class AppConfig:
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Loads configuration from YAML file or returns defaults."""
    config = AppConfig()
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f) or {}

        if "tracker" in raw_dict:
            config.tracker = TrackerConfig(**raw_dict["tracker"])
        if "preprocessing" in raw_dict:
            config.preprocessing = PreprocessingConfig(**raw_dict["preprocessing"])
        if "model" in raw_dict:
            config.model = ModelConfig(**raw_dict["model"])
        if "training" in raw_dict:
            config.training = TrainingConfig(**raw_dict["training"])
        if "server" in raw_dict:
            config.server = ServerConfig(**raw_dict["server"])

    return config
