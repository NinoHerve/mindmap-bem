"""
Module Name: mesh.py
Description: Triangle-mesh operations used by the fit, in numpy.

The fit used to call VTK for its normals and smoothing, which made its result
depend on the VTK version. These replace them, so every environment computes
the same fit (up to floating point).

All functions take vertex positions ``pts`` (n, 3) and triangles ``faces``
(m, 3); the connectivity of a fitted mesh never changes, so everything that
depends only on ``faces`` is computed once.

Author: Nino Herve
License: MIT
"""

import numpy as np
import scipy.sparse as sp


def _edges(faces: np.ndarray) -> np.ndarray:
    """Unique undirected edges (k, 2), each as a sorted vertex pair."""
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    return np.unique(edges, axis=0)


def _adjacency(faces: np.ndarray, n_points: int) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Vertex adjacency matrix, degree of each vertex and the edges."""
    edges = _edges(faces)
    i, j = np.concatenate([edges[:, 0], edges[:, 1]]), np.concatenate([edges[:, 1], edges[:, 0]])
    A = sp.csr_matrix((np.ones(len(i)), (i, j)), shape=(n_points, n_points))
    deg = np.asarray(A.sum(axis=1)).ravel()

    return A, deg, edges


def _face_normals(pts: np.ndarray, faces: np.ndarray, unit: bool = True) -> np.ndarray:
    """Triangle normals from the vertex order (right-hand rule)."""
    normals = np.cross(pts[faces[:, 1]] - pts[faces[:, 0]], pts[faces[:, 2]] - pts[faces[:, 0]])
    if not unit:
        return normals
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(normals, length, out=np.zeros_like(normals), where=length > 0)


def _vertex_normals(pts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unit vertex normals: the mean of the unit normals of the triangles around each vertex (as VTK)."""
    fn = _face_normals(pts, faces)
    normals = np.zeros_like(pts)
    for k in range(3):
        np.add.at(normals, faces[:, k], fn)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.divide(normals, length, out=np.zeros_like(normals), where=length > 0)


def _laplacian_smooth(pts: np.ndarray, A: sp.csr_matrix, deg: np.ndarray, n_iter: int,
                      relaxation: float) -> np.ndarray:
    """Move each vertex ``relaxation`` of the way to its neighbours' centroid, ``n_iter`` times."""
    for _ in range(n_iter):
        pts = pts + relaxation * (A @ pts / deg[:, None] - pts)
    return pts


def _taubin_smooth(pts: np.ndarray, A: sp.csr_matrix, deg: np.ndarray, n_iter: int,
                   lam: float = 0.33, mu: float = -0.34) -> np.ndarray:
    """
    Taubin's lambda/mu smoothing: smooths without shrinking the surface.

    The defaults (pass band ~0.1) move a fitted surface as much as VTK >= 9.4's
    ``smooth_taubin(n_iter=5, pass_band=0.1)``, which the fit used before.
    """
    for _ in range(n_iter):
        pts = pts + lam * (A @ pts / deg[:, None] - pts)
        pts = pts + mu * (A @ pts / deg[:, None] - pts)
    return pts


def _edge_face_pairs(faces: np.ndarray) -> np.ndarray:
    """(k, 2) indices of the two triangles on each side of every shared edge."""
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(faces)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owner = edges[order], owner[order]

    # consecutive equal edges are the two faces sharing it
    shared = np.all(edges[1:] == edges[:-1], axis=1)
    return np.stack([owner[:-1][shared], owner[1:][shared]], axis=1)


def _folded(pts: np.ndarray, faces: np.ndarray, pairs: np.ndarray, max_angle: float = 120.0) -> np.ndarray:
    """Boolean per triangle: its normal and a neighbour's differ by more than ``max_angle`` degrees."""
    fn = _face_normals(pts, faces)
    a, b = pairs[:, 0], pairs[:, 1]
    folded = np.einsum("ij,ij->i", fn[a], fn[b]) < np.cos(np.radians(max_angle))

    out = np.zeros(len(faces), dtype=bool)
    out[a[folded]] = out[b[folded]] = True

    return out


