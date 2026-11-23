"""
Module Name: quality.py
Description: Sanity checks a BEM must pass before computing a forward model.

Author: Nino Herve
License: MIT
"""

import numpy as np
import pyvista as pv

from mindmap_bem.io import polydata_to_faces

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


def count_folds(mesh: pv.PolyData, max_angle: float = 120.0) -> int:
    """Number of triangles folded back onto a neighbour (see ``folded_faces``)."""
    return int(folded_faces(mesh, max_angle).sum())


def folded_faces(mesh: pv.PolyData, max_angle: float = 120.0) -> np.ndarray:
    """
    Boolean per triangle: True if it is folded back onto a neighbour.

    A triangle counts as folded if, across at least one of its edges, its
    normal and its neighbour's differ by more than ``max_angle`` degrees.
    A triangle turned inside out next to a normal one gives about 180.

    Normals come from the triangles' own vertex order. A fold doesn't change
    which vertices are connected, so a folded mesh is still a closed manifold:
    this is the check that catches it.
    """
    faces = polydata_to_faces(mesh)
    pts = np.asarray(mesh.points, dtype=float)
    normals = np.cross(pts[faces[:, 1]] - pts[faces[:, 0]], pts[faces[:, 2]] - pts[faces[:, 0]])
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, length, out=np.zeros_like(normals), where=length > 0)

    # each edge as a sorted vertex pair, with the face it belongs to
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(faces)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owner = edges[order], owner[order]

    # consecutive equal edges are the two faces sharing it
    shared = np.all(edges[1:] == edges[:-1], axis=1)
    a, b = owner[:-1][shared], owner[1:][shared]
    folded = np.einsum("ij,ij->i", normals[a], normals[b]) < np.cos(np.radians(max_angle))

    out = np.zeros(len(faces), dtype=bool)
    out[a[folded]] = out[b[folded]] = True

    return out


def _bem_report(layers: dict[str, pv.PolyData], min_gap: float = 1.0) -> tuple[bool, list[str]]:
    """``check_bem``'s result and the lines it prints."""
    ok, lines = True, []
    for name, mesh in layers.items():
        closed = mesh.n_open_edges == 0 and mesh.is_manifold
        folds = count_folds(mesh)
        good = closed and folds == 0
        ok &= good
        flag = "" if good else "  <-- invalid surface"
        lines.append(f"{name:12s} {mesh.n_points:6d} vertices, closed manifold: {closed}, "
                     f"folded faces: {folds}{flag}")

    for pair, g in layer_gaps(layers).items():
        nested = g["min"] >= min_gap
        ok &= nested
        flag = "" if nested else f"  <-- below {min_gap} mm ({g['n_outside']} vertices outside)"
        lines.append(f"{pair:26s} gap min {g['min']:5.2f} mm, median {g['median']:5.2f} mm{flag}")

    return bool(ok), lines


def check_bem(layers: dict[str, pv.PolyData], min_gap: float = 1.0, verbose: bool = True) -> bool:
    """
    True if every layer is a closed manifold without folds and nested at least
    ``min_gap`` mm inside the next one.
    """
    ok, lines = _bem_report(layers, min_gap)
    if verbose:
        print("\n".join(lines))

    return ok
