from __future__ import annotations

import os
import argparse
from pathlib import Path
import sys

for env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[env_name] = "1"

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from strandbeest.config import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    LinkageConfig,
    ModelConfig,
    TrainConfig,
)
from strandbeest.pipeline import run_full_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full Strandbeest inverse-design pipeline.")
    p.add_argument("--output", type=Path, default=Path("runs/default"))
    p.add_argument("--n-samples", type=int, default=6000)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--budget-per-query", type=int, default=128)
    p.add_argument("--n-queries-id", type=int, default=120)
    p.add_argument("--n-queries-ood", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        linkage=LinkageConfig(),
        data=DataConfig(n_samples=args.n_samples, seed=args.seed),
        model=ModelConfig(),
        train=TrainConfig(epochs=args.epochs, batch_size=args.batch_size, device=args.device),
        eval=EvalConfig(
            budget_per_query=args.budget_per_query,
            n_queries_id=args.n_queries_id,
            n_queries_ood=args.n_queries_ood,
        ),
        output_dir=args.output,
    )
    paths = run_full_pipeline(cfg)
    print("Pipeline finished. Artifacts:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
