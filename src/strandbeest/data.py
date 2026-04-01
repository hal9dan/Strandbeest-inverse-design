from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from .config import DataConfig, LinkageConfig
from .kinematics import evaluate_design


@dataclass
class DatasetBundle:
    x: np.ndarray
    y: np.ndarray
    valid_ratio: np.ndarray
    violation: np.ndarray
    attempts: int
    accepted: int

    @property
    def invalid_rate(self) -> float:
        if self.attempts == 0:
            return 0.0
        return float(1.0 - self.accepted / self.attempts)


def sample_x11(rng: np.random.Generator, bounds: np.ndarray, n: int) -> np.ndarray:
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return rng.uniform(lo, hi, size=(n, bounds.shape[0])).astype(np.float64)


def generate_dataset(linkage_cfg: LinkageConfig, data_cfg: DataConfig) -> DatasetBundle:
    rng = np.random.default_rng(data_cfg.seed)
    bounds = linkage_cfg.bounds_array()

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    valid_list: list[float] = []
    viol_list: list[float] = []

    max_attempts = data_cfg.n_samples * data_cfg.max_attempt_factor
    attempts = 0
    pbar = tqdm(total=data_cfg.n_samples, desc="Generating feasible x11 dataset")

    while len(x_list) < data_cfg.n_samples and attempts < max_attempts:
        batch_n = min(data_cfg.sample_batch_size, max_attempts - attempts)
        batch = sample_x11(rng, bounds, batch_n)
        attempts += batch_n
        for x11 in batch:
            res = evaluate_design(x11, linkage_cfg)
            if not res.feasible:
                continue
            x_list.append(x11.astype(np.float32))
            y_list.append(res.metrics.astype(np.float32))
            valid_list.append(float(res.valid_ratio))
            viol_list.append(float(res.violation))
            pbar.update(1)
            if len(x_list) >= data_cfg.n_samples:
                break
    pbar.close()

    accepted = len(x_list)
    if accepted < data_cfg.n_samples:
        raise RuntimeError(
            f"Could not generate enough feasible x11 samples: {accepted}/{data_cfg.n_samples}. "
            f"Current invalid rate={1.0 - accepted / max(1, attempts):.3f}. "
            "Tighten bounds or increase max_attempt_factor."
        )

    return DatasetBundle(
        x=np.asarray(x_list, dtype=np.float32),
        y=np.asarray(y_list, dtype=np.float32),
        valid_ratio=np.asarray(valid_list, dtype=np.float32),
        violation=np.asarray(viol_list, dtype=np.float32),
        attempts=attempts,
        accepted=accepted,
    )


def split_indices(n: int, train_ratio: float, val_ratio: float, seed: int) -> dict[str, np.ndarray]:
    if not (0.0 < train_ratio < 1.0 and 0.0 < val_ratio < 1.0 and train_ratio + val_ratio < 1.0):
        raise ValueError("Invalid split ratios.")

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def fit_standardizer(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return arr.mean(axis=0), arr.std(axis=0)


def sample_queries(
    y_test: np.ndarray, n_id: int, n_ood: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 17)
    id_idx = rng.choice(len(y_test), size=min(n_id, len(y_test)), replace=False)
    id_queries = y_test[id_idx]

    y_min = y_test.min(axis=0)
    y_max = y_test.max(axis=0)
    span = np.maximum(y_max - y_min, 1e-6)
    ood_low = y_min - 0.25 * span
    ood_high = y_max + 0.25 * span
    ood_queries = rng.uniform(ood_low, ood_high, size=(n_ood, y_test.shape[1])).astype(np.float32)

    queries = np.vstack([id_queries, ood_queries]).astype(np.float32)
    labels = np.array(["ID"] * len(id_queries) + ["OOD"] * len(ood_queries), dtype=object)
    return queries, labels, id_idx
