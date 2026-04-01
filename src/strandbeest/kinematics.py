from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np

from .config import LinkageConfig


@dataclass
class PoseResult:
    valid: bool
    points: dict[str, np.ndarray]


@dataclass
class EvalResult:
    feasible: bool
    metrics: np.ndarray
    trajectory: np.ndarray
    valid_ratio: float
    violation: float
    valid_mask: np.ndarray


def _add_angle_scad(v: np.ndarray, angle_deg: float, length: float) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    return np.array(
        [v[0] + np.cos(angle_rad) * length, v[1] - np.sin(angle_rad) * length],
        dtype=np.float64,
    )


def _get_angle_scad(v1: np.ndarray, v2: np.ndarray) -> float:
    return math.degrees(math.atan2(v2[0] - v1[0], v2[1] - v1[1]))


def _vvll2d(v1: np.ndarray, v2: np.ndarray, l1: float, l2: float) -> Optional[np.ndarray]:
    span = math.hypot(v1[0] - v2[0], v1[1] - v2[1])
    denom = -abs(2.0 * span * l1)
    if span < 1e-12 or denom == 0.0:
        return None
    cos_arg = (l2 * l2 - l1 * l1 - span * span) / denom
    if cos_arg < -1.0 - 1e-9 or cos_arg > 1.0 + 1e-9:
        return None
    cos_arg = max(-1.0, min(1.0, cos_arg))
    ang = math.degrees(math.acos(cos_arg))
    ang += _get_angle_scad(v2, v1) - 90.0
    return _add_angle_scad(v1, ang, -l1)


def _to_cartesian(points: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.array([p[0], -p[1]], dtype=np.float64) for name, p in points.items()}


def solve_pose(x11: np.ndarray, theta_deg: float, cfg: LinkageConfig) -> PoseResult:
    geom = cfg.geometry_dict(x11)

    z = np.array([0.0, 0.0], dtype=np.float64)
    y = np.array([geom["a"], geom["l"]], dtype=np.float64)
    x = _add_angle_scad(z, theta_deg, geom["m"])

    w = _vvll2d(x, y, geom["j"], geom["b"])
    if w is None:
        return PoseResult(valid=False, points={})
    v = _vvll2d(w, y, geom["e"], geom["d"])
    if v is None:
        return PoseResult(valid=False, points={})
    u = _vvll2d(y, x, geom["c"], geom["k"])
    if u is None:
        return PoseResult(valid=False, points={})
    t = _vvll2d(v, u, geom["f"], geom["g"])
    if t is None:
        return PoseResult(valid=False, points={})
    s = _vvll2d(t, u, geom["h"], geom["i"])
    if s is None:
        return PoseResult(valid=False, points={})

    points = {"Z": z, "Y": y, "X": x, "W": w, "V": v, "U": u, "T": t, "S": s}
    return PoseResult(valid=True, points=_to_cartesian(points))


def foot_trajectory(
    x11: np.ndarray, theta_samples: np.ndarray, cfg: LinkageConfig
) -> tuple[np.ndarray, np.ndarray]:
    theta_samples = np.asarray(theta_samples, dtype=np.float64)
    traj = np.full((theta_samples.size, 2), np.nan, dtype=np.float64)
    valid = np.zeros(theta_samples.size, dtype=bool)

    for idx, theta_deg in enumerate(theta_samples):
        pose = solve_pose(x11, float(theta_deg), cfg)
        if not pose.valid:
            continue
        traj[idx] = pose.points["S"]
        valid[idx] = True
    return traj, valid


def _fill_periodic_1d(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    if valid_mask.all():
        return values.copy()
    idx = np.arange(values.size)
    valid_idx = idx[valid_mask]
    valid_values = values[valid_mask]
    if valid_idx.size == 0:
        return values.copy()

    period = values.size
    ext_idx = np.concatenate([valid_idx - period, valid_idx, valid_idx + period])
    ext_values = np.concatenate([valid_values, valid_values, valid_values])
    filled = np.interp(idx, ext_idx, ext_values)
    return filled.astype(np.float64)


def fill_trajectory(trajectory: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    if trajectory.shape[0] != valid_mask.shape[0]:
        raise ValueError("trajectory and valid_mask must have the same length")
    filled = trajectory.copy()
    filled[:, 0] = _fill_periodic_1d(trajectory[:, 0], valid_mask)
    filled[:, 1] = _fill_periodic_1d(trajectory[:, 1], valid_mask)
    return filled


def _largest_cyclic_segment(mask: np.ndarray) -> np.ndarray:
    if mask.size == 0 or not mask.any():
        return np.zeros_like(mask, dtype=bool)

    doubled = np.concatenate([mask, mask])
    best_start = 0
    best_len = 0
    cur_start = None
    cur_len = 0

    for idx, flag in enumerate(doubled):
        if flag:
            if cur_start is None:
                cur_start = idx
                cur_len = 1
            else:
                cur_len += 1
            if cur_len > best_len and cur_len <= mask.size:
                best_start = cur_start
                best_len = cur_len
        else:
            cur_start = None
            cur_len = 0

    out = np.zeros(mask.size, dtype=bool)
    for offset in range(best_len):
        out[(best_start + offset) % mask.size] = True
    return out


def compute_metrics(trajectory: np.ndarray, cfg: LinkageConfig) -> np.ndarray:
    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        raise ValueError(f"Expected trajectory shape (T, 2), got {trajectory.shape}")
    if not np.all(np.isfinite(trajectory)):
        return np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float64)

    x = trajectory[:, 0]
    y = trajectory[:, 1]
    y_span = float(np.ptp(y))
    ground = float(np.min(y))
    stance_tol = max(cfg.ground_tolerance_abs, cfg.stance_height_fraction * max(y_span, 1e-6))
    stance = y <= ground + stance_tol

    if stance.sum() < cfg.min_stance_samples:
        return np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float64)

    primary_stance = _largest_cyclic_segment(stance)
    if primary_stance.sum() < cfg.min_stance_samples:
        return np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float64)

    theta = np.linspace(0.0, 2.0 * np.pi, trajectory.shape[0], endpoint=False)
    vx = np.gradient(x, theta)

    step_length = float(np.max(x[primary_stance]) - np.min(x[primary_stance]))
    swing = ~stance
    clearance = float(np.max(y[swing]) - ground) if swing.any() else 0.0
    duty_factor = float(np.mean(stance))
    smoothness = float(np.var(vx[stance]))

    return np.array([step_length, clearance, duty_factor, smoothness], dtype=np.float64)


def evaluate_design(x11: np.ndarray, cfg: LinkageConfig) -> EvalResult:
    x11 = np.asarray(x11, dtype=np.float64)
    if x11.shape != (cfg.dim_x,):
        raise ValueError(f"Expected x11 shape {(cfg.dim_x,)}, got {x11.shape}")

    theta_samples = np.linspace(0.0, 360.0, cfg.angle_samples, endpoint=False)
    traj, valid_mask = foot_trajectory(x11, theta_samples, cfg)
    valid_ratio = float(np.mean(valid_mask))

    full_traj = fill_trajectory(traj, valid_mask) if valid_mask.any() else traj
    metrics = compute_metrics(full_traj, cfg) if valid_ratio >= cfg.min_valid_ratio else np.array(
        [np.nan, np.nan, np.nan, np.nan], dtype=np.float64
    )

    feasible = bool(valid_ratio >= cfg.min_valid_ratio and np.all(np.isfinite(metrics)))
    violation = max(0.0, cfg.min_valid_ratio - valid_ratio)
    if not np.all(np.isfinite(metrics)):
        violation += 1.0

    return EvalResult(
        feasible=feasible,
        metrics=metrics,
        trajectory=full_traj,
        valid_ratio=valid_ratio,
        violation=float(violation),
        valid_mask=valid_mask,
    )
