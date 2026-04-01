from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .baselines import clamp_to_bounds, evolutionary_search, sample_random
from .config import ExperimentConfig
from .data import fit_standardizer, generate_dataset, sample_queries, split_indices
from .evaluation import aggregate_results, evaluate_candidates, summarize_candidate_set
from .kinematics import evaluate_design
from .models import (
    Normalizer,
    pick_device,
    sample_cvae,
    sample_regressor,
    train_cvae,
    train_regressor,
    torch,
)
from .plots import (
    plot_ablation_k,
    plot_reference_schematic,
    plot_training_curves,
    plot_trajectory_panel,
)
from .reference import REFERENCE


def _mkdirs(base: Path) -> dict[str, Path]:
    paths = {
        "root": base,
        "data": base / "data",
        "models": base / "models",
        "tables": base / "tables",
        "figs": base / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _fit_normalizer(x_train: np.ndarray, y_train: np.ndarray) -> Normalizer:
    x_mean, x_std = fit_standardizer(x_train)
    y_mean, y_std = fit_standardizer(y_train)
    return Normalizer(
        x_mean=x_mean.astype(np.float32),
        x_std=x_std.astype(np.float32),
        y_mean=y_mean.astype(np.float32),
        y_std=y_std.astype(np.float32),
    )


def _serialize_cfg(cfg: ExperimentConfig) -> dict:
    out = asdict(cfg)
    out["output_dir"] = str(cfg.output_dir)
    out["reference"] = REFERENCE.manifest()
    return out


def _objective_for_target(target: np.ndarray, linkage_cfg, metric_scale: np.ndarray):
    def _objective(x11: np.ndarray) -> float:
        res = evaluate_design(x11, linkage_cfg)
        if not res.feasible:
            return 1e6 + 100.0 * res.violation
        return float(np.linalg.norm((res.metrics - target) / metric_scale))

    return _objective


def _select_best(cands):
    cands_sorted = sorted(cands, key=lambda c: c.error)
    return cands_sorted[0] if cands_sorted else None


def _write_reference_table(out_path: Path) -> None:
    bounds = REFERENCE.bounds_array()
    rows = []
    for idx, name in enumerate(REFERENCE.optimized_names):
        rows.append(
            {
                "name": name,
                "canonical_value": float(REFERENCE.canonical_x11[idx]),
                "lower_bound": float(bounds[idx, 0]),
                "upper_bound": float(bounds[idx, 1]),
                "role": "optimized_length_element",
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _persist_query_outputs(
    records: list[dict[str, object]], detail_path: Path, summary_path: Path, budget: int
) -> None:
    detail_df = pd.DataFrame(records)
    detail_df.to_csv(detail_path, index=False)

    if detail_df.empty:
        summary_df = pd.DataFrame()
    else:
        summary_df = aggregate_results(records)
        summary_df["budget_per_query"] = budget
    summary_df.to_csv(summary_path, index=False)


def run_full_pipeline(cfg: ExperimentConfig) -> dict[str, Path]:
    paths = _mkdirs(cfg.output_dir)
    linkage_cfg = cfg.linkage
    bounds = linkage_cfg.bounds_array()
    budget = cfg.eval.budget_per_query

    _write_reference_table(paths["tables"] / "reference_lengths.csv")
    with open(paths["root"] / "reference_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(REFERENCE.manifest(), fh, indent=2)
    plot_reference_schematic(paths["figs"] / "reference_schematic.png", linkage_cfg)

    dataset = generate_dataset(linkage_cfg, cfg.data)
    splits = split_indices(
        len(dataset.x), cfg.data.train_ratio, cfg.data.val_ratio, seed=cfg.data.seed
    )
    x_train, y_train = dataset.x[splits["train"]], dataset.y[splits["train"]]
    x_val, y_val = dataset.x[splits["val"]], dataset.y[splits["val"]]
    x_test, y_test = dataset.x[splits["test"]], dataset.y[splits["test"]]

    normalizer = _fit_normalizer(x_train, y_train)
    x_train_std = normalizer.norm_x(x_train)
    y_train_std = normalizer.norm_y(y_train)
    x_val_std = normalizer.norm_x(x_val)
    y_val_std = normalizer.norm_y(y_val)
    metric_scale = np.clip(normalizer.y_std.astype(np.float64), 1e-8, None)

    np.savez_compressed(
        paths["data"] / "dataset.npz",
        x=dataset.x,
        y=dataset.y,
        valid_ratio=dataset.valid_ratio,
        violation=dataset.violation,
        train_idx=splits["train"],
        val_idx=splits["val"],
        test_idx=splits["test"],
        attempts=dataset.attempts,
        accepted=dataset.accepted,
        invalid_rate=dataset.invalid_rate,
        optimized_names=np.asarray(REFERENCE.optimized_names, dtype=object),
        canonical_x11=np.asarray(REFERENCE.canonical_x11, dtype=np.float32),
        bounds=bounds.astype(np.float32),
    )
    np.savez_compressed(
        paths["data"] / "normalizer.npz",
        x_mean=normalizer.x_mean,
        x_std=normalizer.x_std,
        y_mean=normalizer.y_mean,
        y_std=normalizer.y_std,
    )
    with open(paths["root"] / "config.json", "w", encoding="utf-8") as fh:
        json.dump(_serialize_cfg(cfg), fh, indent=2)

    models = {}
    model_enabled = torch is not None
    device = "cpu"
    if model_enabled:
        try:
            device = pick_device(cfg.train.device)
            cvae, cvae_hist = train_cvae(
                x_train_std, y_train_std, x_val_std, y_val_std, cfg.model, cfg.train
            )
            cvae = cvae.to(device)
            models["cvae"] = cvae
            plot_training_curves(
                cvae_hist.train_loss,
                cvae_hist.val_loss,
                paths["figs"] / "train_cvae.png",
                "cVAE Loss",
            )

            reg, reg_hist = train_regressor(
                x_train_std, y_train_std, x_val_std, y_val_std, cfg.model, cfg.train
            )
            reg = reg.to(device)
            models["reg"] = reg
            plot_training_curves(
                reg_hist.train_loss,
                reg_hist.val_loss,
                paths["figs"] / "train_regressor.png",
                "Conditional MLP Loss",
            )

            torch.save(cvae.state_dict(), paths["models"] / "cvae.pt")
            torch.save(reg.state_dict(), paths["models"] / "regressor.pt")
        except Exception as exc:
            model_enabled = False
            print(f"Warning: training failed, skipping learned baselines: {exc}")
    else:
        print("torch not available; running search baselines only.")

    queries, split_labels, _ = sample_queries(
        y_test, cfg.eval.n_queries_id, cfg.eval.n_queries_ood, seed=cfg.data.seed
    )

    detail_path = paths["tables"] / "query_results.csv"
    summary_path = paths["tables"] / "summary_results.csv"

    method_order = ["Random search", "Evolutionary search"]
    if model_enabled:
        method_order.extend(["Cond. regression (MLP)", "cVAE + post-select (ours)"])
        if cfg.eval.include_one_shot_cvae:
            method_order.append("cVAE one-shot (ours)")
    expected_methods = set(method_order)

    records: list[dict[str, object]] = []
    completed_queries: set[int] = set()
    if detail_path.exists():
        existing_df = pd.read_csv(detail_path)
        if not existing_df.empty:
            query_method_sets = (
                existing_df.groupby("query_id")["method"].agg(lambda s: set(s.astype(str))).to_dict()
            )
            completed_queries = {
                int(query_id)
                for query_id, methods in query_method_sets.items()
                if methods == expected_methods
            }
            existing_df = existing_df[existing_df["query_id"].isin(sorted(completed_queries))]
            records = existing_df.to_dict(orient="records")
            if completed_queries:
                print(
                    f"Resuming from {detail_path}; "
                    f"skipping {len(completed_queries)} completed queries."
                )

    panel_store = {}
    iterator = tqdm(enumerate(zip(queries, split_labels)), desc="Benchmark queries", total=len(queries))
    for query_id, (target, split_name) in iterator:
        if query_id in completed_queries:
            continue

        seed_base = cfg.data.seed + 10_000 + query_id * 97
        method_candidates: dict[str, tuple[np.ndarray, int]] = {}

        method_candidates["Random search"] = (
            sample_random(bounds, budget, seed=seed_base + 1),
            budget,
        )
        method_candidates["Evolutionary search"] = (
            evolutionary_search(
                objective_fn=_objective_for_target(target, linkage_cfg, metric_scale),
                bounds=bounds,
                budget=budget,
                seed=seed_base + 2,
                pop_size=min(48, budget),
            ),
            budget,
        )

        if model_enabled:
            method_candidates["Cond. regression (MLP)"] = (
                clamp_to_bounds(
                    sample_regressor(
                        models["reg"],
                        target,
                        budget,
                        normalizer,
                        noise_std=cfg.model.regression_noise_std,
                        seed=seed_base + 3,
                        device=device,
                    ),
                    bounds,
                ),
                budget,
            )

            cvae_budget_samples = clamp_to_bounds(
                sample_cvae(
                    models["cvae"],
                    target,
                    budget,
                    normalizer,
                    seed=seed_base + 4,
                    device=device,
                ),
                bounds,
            )
            method_candidates["cVAE + post-select (ours)"] = (cvae_budget_samples, budget)

            if cfg.eval.include_one_shot_cvae:
                one_shot = clamp_to_bounds(
                    sample_cvae(
                        models["cvae"],
                        target,
                        1,
                        normalizer,
                        seed=seed_base + 5,
                        device=device,
                    ),
                    bounds,
                )
                method_candidates["cVAE one-shot (ours)"] = (one_shot, 1)

        try:
            for method, (cand_x, eval_budget_used) in method_candidates.items():
                cands = evaluate_candidates(
                    cand_x,
                    target,
                    lambda x: evaluate_design(x, linkage_cfg),
                    metric_scale=metric_scale,
                )
                stats = summarize_candidate_set(cands, epsilon=cfg.eval.epsilon)
                records.append(
                    {
                        "query_id": query_id,
                        "split": split_name,
                        "method": method,
                        "budget_per_query": budget,
                        "evaluator_calls_for_selection": eval_budget_used,
                        **stats,
                    }
                )

                if query_id < 3:
                    best = _select_best(cands)
                    if best is not None and np.isfinite(best.error):
                        panel_store.setdefault(
                            query_id, {"target_name": f"{split_name}-{query_id}", "methods": []}
                        )
                        panel_store[query_id]["methods"].append(
                            {"name": method, "traj": best.trajectory}
                        )
        except Exception as exc:
            _persist_query_outputs(records, detail_path, summary_path, budget)
            raise RuntimeError(
                f"Benchmark failed at query_id={query_id} ({split_name}) while evaluating methods."
            ) from exc

        _persist_query_outputs(records, detail_path, summary_path, budget)

    if panel_store:
        panel_data = [panel_store[key] for key in sorted(panel_store)]
        plot_trajectory_panel(panel_data, paths["figs"] / "trajectory_panel.png")

    id_queries = queries[np.where(split_labels == "ID")[0]]
    ablation_records = []
    if len(id_queries) > 0:
        probe_queries = id_queries[: min(25, len(id_queries))]
        for k in cfg.eval.ablation_k:
            for local_id, target in enumerate(probe_queries):
                seed_base = cfg.data.seed + 50_000 + local_id * 131 + k * 7
                random_k = sample_random(bounds, k, seed=seed_base + 1)
                cands = evaluate_candidates(
                    random_k,
                    target,
                    lambda x: evaluate_design(x, linkage_cfg),
                    metric_scale=metric_scale,
                )
                stat = summarize_candidate_set(cands, epsilon=cfg.eval.epsilon)
                ablation_records.append(
                    {"method": "Random search", "k": k, "best_error": stat["best_error"]}
                )

                if model_enabled:
                    cvae_k = clamp_to_bounds(
                        sample_cvae(
                            models["cvae"], target, k, normalizer, seed=seed_base + 2, device=device
                        ),
                        bounds,
                    )
                    cands = evaluate_candidates(
                        cvae_k,
                        target,
                        lambda x: evaluate_design(x, linkage_cfg),
                        metric_scale=metric_scale,
                    )
                    stat = summarize_candidate_set(cands, epsilon=cfg.eval.epsilon)
                    ablation_records.append(
                        {"method": "cVAE + post-select (ours)", "k": k, "best_error": stat["best_error"]}
                    )

        ablation_df = pd.DataFrame(ablation_records)
        ablation_df = ablation_df.groupby(["method", "k"], as_index=False)["best_error"].mean()
        ablation_df.to_csv(paths["tables"] / "ablation_k.csv", index=False)
        plot_ablation_k(ablation_df, paths["figs"] / "ablation_k.png")

    return paths
