from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SupportGeometry:
    a_offset: float = 38.0
    l_offset: float = 7.8


@dataclass(frozen=True)
class JansenReference:

    name: str = "OpenSCAD/Wikibooks Jansen linkage"
    source_url: str = (
        "https://en.wikibooks.org/wiki/OpenSCAD_User_Manual/Printable_version#Theo_Jansen_linkage"
    )
    source_note: str = (
        "Canonical linkage values adapted from the Wikibooks/OpenSCAD Theo Jansen linkage example."
    )
    optimized_names: tuple[str, ...] = ("m", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k")
    canonical_x11: tuple[float, ...] = (
        15.0,  # m: crank radius
        41.5,  # b
        39.3,  # c
        40.1,  # d
        55.8,  # e
        39.4,  # f
        36.7,  # g
        65.7,  # h
        49.0,  # i
        50.0,  # j
        61.9,  # k
    )
    support: SupportGeometry = SupportGeometry()
    lower_scale: float = 0.82
    upper_scale: float = 1.18

    @property
    def dim_x(self) -> int:
        return len(self.optimized_names)

    def canonical_array(self) -> np.ndarray:
        return np.asarray(self.canonical_x11, dtype=np.float64)

    def bounds_array(self) -> np.ndarray:
        base = self.canonical_array()
        lo = np.maximum(1e-3, self.lower_scale * base)
        hi = self.upper_scale * base
        return np.column_stack([lo, hi]).astype(np.float64)

    def geometry_dict(self, x11: np.ndarray) -> dict[str, float]:
        x11 = np.asarray(x11, dtype=np.float64)
        if x11.shape != (self.dim_x,):
            raise ValueError(f"Expected shape {(self.dim_x,)}, got {x11.shape}")
        geom = {name: float(value) for name, value in zip(self.optimized_names, x11)}
        geom["a"] = float(self.support.a_offset)
        geom["l"] = float(self.support.l_offset)
        return geom

    def manifest(self) -> dict[str, object]:
        out = asdict(self)
        out["canonical_x11"] = list(self.canonical_x11)
        out["optimized_names"] = list(self.optimized_names)
        return out


REFERENCE = JansenReference()

