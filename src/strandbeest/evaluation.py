from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .kinematics import EvalResult, evaluate_design


@dataclass
class CandidateEval:
    x: np.ndarray
    feasible: bool
    violation: float
    error: float
    metrics: np.ndarray
    trajectory: np.ndarray


def evaluate_candidates(
    candidates: np.ndarray,
    target: np.ndarray,
    eval_fn: Callable[[np.ndarray], EvalResult],
    metric_scale: np.ndarray | None = None,
) -> list[CandidateEval]:
    out: list[CandidateEval] = []
    if metric_scale is None:
        metric_scale = np.ones_like(target, dtype=np.float64)
    metric_scale = np.clip(np.asarray(metric_scale, dtype=np.float64), 1e-8, None)
    for x in candidates:
        res = eval_fn(x)
        if not res.feasible:
            err = np.inf
        else:
            err = float(np.linalg.norm((res.metrics - target) / metric_scale))
        out.append(
            CandidateEval(
                x=x,
                feasible=res.feasible,
                violation=res.violation,
                error=err,
                metrics=res.metrics,
                trajectory=res.trajectory,
            )
        )
    return out


def summarize_candidate_set(cands: list[CandidateEval], epsilon: float) -> dict[str, float]:
    feasible = np.array([c.feasible for c in cands], dtype=bool)
    errs = np.array([c.error for c in cands], dtype=np.float64)
    viols = np.array([c.violation for c in cands], dtype=np.float64)
    best_err = float(np.min(errs)) if len(errs) else np.inf
    return {
        "best_error": best_err,
        "success": float(best_err <= epsilon),
        "viol_rate": float(1.0 - feasible.mean()) if len(feasible) else 1.0,
        "mean_violation": float(viols.mean()) if len(viols) else np.inf,
    }


def aggregate_results(records: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    grouped = (
        df.groupby(["split", "method"], as_index=False)
        .agg(
            success_at_eps=("success", "mean"),
            best_of_k_error=("best_error", "mean"),
            violation_rate=("viol_rate", "mean"),
            n_queries=("query_id", "count"),
        )
        .sort_values(["split", "best_of_k_error"])
    )
    return grouped


def default_eval_fn(linkage_cfg):
    return lambda x: evaluate_design(x, linkage_cfg)
