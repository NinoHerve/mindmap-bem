"""
Module Name: quality.py
Description: Sanity checks a BEM must pass before computing a forward model,
and metrics to compare the forward models of different BEMs.

Author: Nino Herve
License: MIT
"""

import numpy as np
import pyvista as pv

from mindmap_bem.io import polydata_to_faces
from mindmap_bem.mesh import _edge_face_pairs, _folded

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
    return _folded(pts, faces, _edge_face_pairs(faces), max_angle)


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


# --- forward models -------------------------------------------------------
# Leadfields are (n_channels, n_sources) arrays: one column per source.

def rdm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Relative difference measure between two leadfields, per source.

    ``|| a / ||a|| - b / ||b|| ||`` for each column: the difference in
    topography shape, ignoring magnitude. 0 means identical topographies, 2
    opposite ones. Compare leadfields with the same reference, e.g. from
    ``common_leadfields``.
    """
    return np.linalg.norm(a/np.linalg.norm(a, axis=0) - b/np.linalg.norm(b, axis=0), axis=0)


def lnmag(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Log magnitude ratio between two leadfields, per source.

    ``ln(||a|| / ||b||)`` for each column. 0 means the same magnitude, > 0
    that ``a`` is stronger. Compare leadfields with the same reference, e.g.
    from ``common_leadfields``.
    """
    return np.log(np.linalg.norm(a, axis=0)/np.linalg.norm(b, axis=0))


def common_leadfields(fwds: dict) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    """
    Average-referenced leadfields restricted to the sources present in every
    forward solution.

    Forward solutions computed with different BEMs can keep different sources
    (MNE drops those closer than ``mindist`` to the inner skull), so they must
    be restricted to the same sources before comparing them column by column.

    Parameters
    ----------
    fwds: dict
        ``{name: forward}``: MNE fixed-orientation forward solutions (see
        ``mne.convert_forward_solution(..., force_fixed=True)``) with the same
        source spaces. MNE itself isn't needed: only their ``"src"`` and
        ``"sol"`` entries are read.

    Returns
    -------
    leadfields: dict
        ``{name: (n_channels, n_common_sources) array}``, average referenced.

    vertices: list of np.ndarray
        Vertex numbers of the common sources, one array per source space
        (e.g. left and right hemisphere), in the order of the columns. Their
        positions are ``src[i]["rr"][vertices[i]]``.
    """
    n_spaces = {len(f["src"]) for f in fwds.values()}
    if len(n_spaces) != 1:
        raise ValueError(f"The forward solutions have different numbers of source spaces: {sorted(n_spaces)}")
    n_spaces = n_spaces.pop()

    for name, f in fwds.items():
        n_sources = sum(int(s["nuse"]) for s in f["src"])
        if f["sol"]["data"].shape[1] != n_sources:
            raise ValueError(
                f"{name}: {f['sol']['data'].shape[1]} leadfield columns for {n_sources} sources, so not a "
                "fixed-orientation forward solution (use mne.convert_forward_solution(fwd, force_fixed=True))."
            )

    vertices = [
        np.array(sorted(set.intersection(*(set(f["src"][i]["vertno"]) for f in fwds.values()))), dtype=int)
        for i in range(n_spaces)
    ]

    leadfields = {}
    for name, f in fwds.items():
        offsets = np.cumsum([0] + [int(s["nuse"]) for s in f["src"]])
        cols = np.concatenate([
            offsets[i] + np.searchsorted(f["src"][i]["vertno"], vertices[i]) for i in range(n_spaces)
        ])
        G = f["sol"]["data"][:, cols]
        leadfields[name] = G - G.mean(axis=0)          # average reference

    return leadfields, vertices
