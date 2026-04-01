from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .reference import REFERENCE


@dataclass(frozen=True)
class LinkageConfig:
    angle_samples: int = 360
    min_valid_ratio: float = 0.98
    stance_height_fraction: float = 0.02
    min_stance_samples: int = 12
    ground_tolerance_abs: float = 0.15

    @property
    def dim_x(self) -> int:
        return REFERENCE.dim_x

    @property
    def optimized_names(self) -> tuple[str, ...]:
        return REFERENCE.optimized_names

    def canonical_x11(self) -> np.ndarray:
        return REFERENCE.canonical_array()

    def bounds_array(self) -> np.ndarray:
        return REFERENCE.bounds_array()

    def geometry_dict(self, x11: np.ndarray) -> dict[str, float]:
        return REFERENCE.geometry_dict(x11)


@dataclass(frozen=True)
class DataConfig:
    n_samples: int = 6000
    max_attempt_factor: int = 80
    sample_batch_size: int = 256
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    seed: int = 42


@dataclass(frozen=True)
class ModelConfig:
    latent_dim: int = 12
    hidden_dim: int = 256
    beta: float = 5e-3
    regression_noise_std: float = 0.08


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 256
    epochs: int = 80
    lr: float = 1e-3
    weight_decay: float = 1e-5
    early_stop_patience: int = 12
    device: str = "auto"


@dataclass(frozen=True)
class EvalConfig:
    n_queries_id: int = 120
    n_queries_ood: int = 80
    budget_per_query: int = 128
    epsilon: float = 1.0
    include_one_shot_cvae: bool = True
    ablation_k: tuple[int, ...] = (8, 16, 32, 64, 96, 128)


@dataclass(frozen=True)
class ExperimentConfig:
    linkage: LinkageConfig = LinkageConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()
    output_dir: Path = Path("runs/default")
