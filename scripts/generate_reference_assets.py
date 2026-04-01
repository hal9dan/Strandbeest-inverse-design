from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from strandbeest.config import LinkageConfig
from strandbeest.plots import plot_reference_schematic
from strandbeest.reference import REFERENCE


def main() -> None:
    pic_dir = ROOT / "571" / "pics"
    pic_dir.mkdir(parents=True, exist_ok=True)
    plot_reference_schematic(pic_dir / "jansen_11_schematic.png", LinkageConfig())

    rows = []
    bounds = REFERENCE.bounds_array()
    for idx, name in enumerate(REFERENCE.optimized_names):
        rows.append(
            {
                "name": name,
                "canonical_value": float(REFERENCE.canonical_x11[idx]),
                "lower_bound": float(bounds[idx, 0]),
                "upper_bound": float(bounds[idx, 1]),
            }
        )
    out_csv = ROOT / "571" / "jansen_reference_lengths.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()

