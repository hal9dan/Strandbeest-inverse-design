from __future__ import annotations

import numpy as np


def clamp_to_bounds(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return np.clip(x, lo, hi)


def sample_random(bounds: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    return rng.uniform(lo, hi, size=(k, bounds.shape[0])).astype(np.float32)


def evolutionary_search(
    objective_fn,
    bounds: np.ndarray,
    budget: int,
    seed: int,
    pop_size: int = 40,
    elite_frac: float = 0.35,
    mutation_std: float = 0.10,
) -> np.ndarray:
    """Budget-capped evolutionary optimizer. Returns all evaluated candidates."""
    rng = np.random.default_rng(seed)
    dim = bounds.shape[0]
    lo = bounds[:, 0]
    hi = bounds[:, 1]

    pop = rng.uniform(lo, hi, size=(pop_size, dim))
    all_points = []
    all_scores = []

    eval_count = 0
    while eval_count < budget:
        remain = budget - eval_count
        eval_pop = pop[:remain]
        scores = np.asarray([objective_fn(x) for x in eval_pop], dtype=np.float64)
        all_points.append(eval_pop.copy())
        all_scores.append(scores.copy())
        eval_count += len(eval_pop)
        if eval_count >= budget:
            break

        elite_n = max(2, int(pop_size * elite_frac))
        elite_idx = np.argsort(scores)[:elite_n]
        elite = eval_pop[elite_idx]

        children = []
        while len(children) < pop_size:
            i, j = rng.integers(0, elite.shape[0], size=2)
            a = rng.uniform(0.0, 1.0)
            child = a * elite[i] + (1.0 - a) * elite[j]
            noise = rng.normal(scale=mutation_std * (hi - lo), size=dim)
            child = clamp_to_bounds(child + noise, bounds)
            children.append(child)
        pop = np.asarray(children, dtype=np.float64)

    return np.vstack(all_points).astype(np.float32)

