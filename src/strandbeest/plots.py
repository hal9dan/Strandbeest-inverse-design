from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import LinkageConfig
from .kinematics import solve_pose
from .reference import REFERENCE


def plot_training_curves(
    train_loss: list[float], val_loss: list[float], out_path: Path, title: str
) -> None:
    plt.figure(figsize=(6.2, 4.4))
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_ablation_k(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(7.0, 4.5))
    for method, group in df.groupby("method"):
        group = group.sort_values("k")
        plt.plot(group["k"], group["best_error"], marker="o", label=method)
    plt.xlabel("K / evaluator calls")
    plt.ylabel("Best-of-K Error")
    plt.title("Ablation on Evaluation Budget")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_trajectory_panel(panel_data, out_path: Path) -> None:
    n_rows = len(panel_data)
    n_cols = max(len(row["methods"]) for row in panel_data)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.5 * n_cols, 2.8 * n_rows), squeeze=False
    )

    for r, row in enumerate(panel_data):
        for c in range(n_cols):
            ax = axes[r, c]
            if c >= len(row["methods"]):
                ax.axis("off")
                continue
            item = row["methods"][c]
            traj = item["traj"]
            mask = np.all(np.isfinite(traj), axis=1)
            ax.plot(traj[mask, 0], traj[mask, 1], lw=1.3)
            ax.set_title(f'{row["target_name"]} | {item["name"]}', fontsize=9)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_reference_schematic(out_path: Path, cfg: LinkageConfig, theta_deg: float = 210.0) -> None:
    pose = solve_pose(REFERENCE.canonical_array(), theta_deg=theta_deg, cfg=cfg)
    if not pose.valid:
        raise RuntimeError("Canonical reference pose is invalid; cannot render schematic.")

    pts = pose.points
    edges = [
        ("Z", "X", "m"),
        ("X", "W", "j"),
        ("W", "Y", "b"),
        ("Y", "U", "c"),
        ("Y", "V", "d"),
        ("W", "V", "e"),
        ("V", "T", "f"),
        ("U", "T", "g"),
        ("T", "S", "h"),
        ("U", "S", "i"),
        ("X", "U", "k"),
    ]

    plt.figure(figsize=(6.8, 5.0))
    for p0, p1, label in edges:
        a = pts[p0]
        b = pts[p1]
        plt.plot([a[0], b[0]], [a[1], b[1]], color="#2f2f2f", lw=2.0)
        mid = 0.5 * (a + b)
        plt.text(mid[0], mid[1], label, fontsize=10, color="#b00020")

    for name, p in pts.items():
        plt.scatter([p[0]], [p[1]], s=28, color="#111111")
        plt.text(p[0] + 0.8, p[1] + 0.8, name, fontsize=8, color="#444444")

    support = np.array([[0.0, 0.0], [REFERENCE.support.a_offset, -REFERENCE.support.l_offset]])
    plt.plot(support[:, 0], support[:, 1], "--", color="#888888", lw=1.3)
    plt.text(
        0.5 * (support[0, 0] + support[1, 0]) - 1.0,
        0.5 * (support[0, 1] + support[1, 1]) + 1.2,
        "fixed support (a,l)",
        fontsize=8,
        color="#666666",
    )

    plt.axis("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Canonical 11-element Jansen Reference")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

