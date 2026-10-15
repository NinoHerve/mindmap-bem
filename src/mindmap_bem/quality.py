"""
Module Name: quality.py
Description: Sanity checks a BEM must pass before computing a forward model.

Author: Nino Herve
License: MIT
"""

import numpy as np
import pyvista as pv

PAIRS = (("inner_skull", "outer_skull"), ("outer_skull", "skin"))


def layer_gaps(layers: dict[str, pv.PolyData]) -> dict[str, dict[str, float]]:
    """
    Signed distance (mm) from each layer's vertices to the next layer out.

    Positive means inside the outer layer; a negative minimum means the
    surfaces intersect, which MNE rejects (or worse, silently mis-solves when
    they merely touch).

    Returns
    -------
    gaps: dict
        ``{"inner_skull -> outer_skull": {"min", "p5", "median", "n_outside"}, ...}``
    """
    gaps = {}
    for inner, outer in PAIRS:
        # pyvista's implicit distance is negative inside a closed surface
        d = -inner_distance(layers[inner], layers[outer])
        gaps[f"{inner} -> {outer}"] = {
            "min": float(d.min()),
            "p5": float(np.percentile(d, 5)),
            "median": float(np.median(d)),
            "n_outside": int((d < 0).sum()),
        }

    return gaps


def inner_distance(points: pv.PolyData, surface: pv.PolyData) -> np.ndarray:
    """Signed distance of ``points`` to closed ``surface`` (negative inside)."""
    return np.asarray(points.compute_implicit_distance(surface)["implicit_distance"])


def check_bem(layers: dict[str, pv.PolyData], min_gap: float = 1.0, verbose: bool = True) -> bool:
    """
    True if every layer is a closed manifold and nested at least ``min_gap``
    mm inside the next one.
    """
    ok = True
    for name, mesh in layers.items():
        closed = mesh.n_open_edges == 0 and mesh.is_manifold
        ok &= closed
        if verbose:
            print(f"{name:12s} {mesh.n_points:6d} vertices, closed manifold: {closed}")

    for pair, g in layer_gaps(layers).items():
        nested = g["min"] >= min_gap
        ok &= nested
        if verbose:
            flag = "" if nested else f"  <-- below {min_gap} mm ({g['n_outside']} vertices outside)"
            print(f"{pair:26s} gap min {g['min']:5.2f} mm, median {g['median']:5.2f} mm{flag}")

    return bool(ok)
