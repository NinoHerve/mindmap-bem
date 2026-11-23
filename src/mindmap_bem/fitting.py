"""
Module Name: fitting.py
Description: Deform an icosahedral sphere onto a binary mask.

The surface shrink-wraps the mask by following a smoothed signed distance
field, optionally constrained to stay inside an outer envelope and/or outside
an inner one. Every layer is the same deformed icosahedron, so its
connectivity is always closed and manifold, and nesting is enforced during the
fit rather than repaired afterwards.

The fit cannot fold: after each iteration, any part of the surface that
folded onto itself is put back where it was (the fold guard). A region the
guard keeps holding back can miss its target, so ``build_bem`` still checks
the result (``check_bem``) before writing it. Normals and smoothing are
computed in numpy (``mesh.py``), so the result doesn't depend on the VTK
version.

Fitting happens in "grid mm" coordinates (voxel index * voxel size) of a
RAS-reoriented volume, so all parameters are in mm whatever the voxel size or
acquisition orientation.

Author: Nino Herve
License: MIT
"""

from importlib import resources

import numpy as np
import nibabel as nib
import pyvista as pv
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_erosion, center_of_mass

from mindmap_bem.io import polydata_to_faces, surf_to_polydata
from mindmap_bem.masks import prepare_mask, signed_distance_interpolator
from mindmap_bem.mesh import (
    _adjacency,
    _edge_face_pairs,
    _folded,
    _laplacian_smooth,
    _taubin_smooth,
    _vertex_normals,
)


def ico_sphere() -> pv.PolyData:
    """
    Unit ico-5 sphere (10242 vertices) with MNE's vertex/triangle ordering.

    Matching MNE's icosahedron keeps the meshes compatible with MNE's own
    ``ico`` downsampling in ``mne.make_bem_model``.
    """
    with resources.files("mindmap_bem").joinpath("data", "ico5.npz").open("rb") as f:
        ico = np.load(f)
        return surf_to_polydata(ico["rr"], ico["tris"])


def to_ras(img: nib.spatialimages.SpatialImage) -> nib.spatialimages.SpatialImage:
    """Reorder voxel axes to (closest) RAS, so the third axis points up."""
    return nib.as_closest_canonical(img)


def fit_surface_to_volume(
    volume: np.ndarray,
    zooms: tuple[float, float, float] = (1.0, 1.0, 1.0),
    iterations: int = 100,
    step_size: float = 0.6,
    threshold: float = 0.5,
    n_smooth: int = 1,
    z_floor: float | None = None,
    outer_interpolator: RegularGridInterpolator | None = None,
    inner_interpolator: RegularGridInterpolator | None = None,
    outer_gap: float = 3.0,
    inner_gap: float = 2.0,
    init_radius: float | None = None,
    sphere: pv.PolyData | None = None,
) -> tuple[pv.PolyData, RegularGridInterpolator]:
    """
    Fit an icosahedral sphere to a mask, with Z-capping and containment
    constraints. Coordinates are grid mm (voxel index * voxel size).

    Parameters
    ----------
    volume: np.ndarray
        Mask volume, RAS-oriented.

    zooms: tuple
        Voxel size (mm).

    iterations: int
        Number of iterations.

    step_size: float
        Step size per iteration (mm).

    threshold: float
        Voxels above threshold are inside the mask.

    n_smooth: int
        Laplacian smoothing iterations applied after each step.

    z_floor: float | None
        Mask below this height (mm above the bottom of the volume) is ignored.

    outer_interpolator: RegularGridInterpolator | None
        Signed distance of an envelope the surface must stay inside of, by at
        least ``outer_gap`` mm.

    inner_interpolator: RegularGridInterpolator | None
        Signed distance of a volume the surface must stay outside of, by at
        least ``inner_gap`` mm. The 2 mm default is the minimum skull thickness
        (see ``docs/decisions.md``).

    init_radius: float | None
        Radius (mm) of the starting sphere, centred on the mask. Default: just
        large enough to enclose the mask (``start_radius``).

    sphere: pv.PolyData | None
        Unit sphere to deform. Defaults to ``ico_sphere()``.

    Returns
    -------
    sphere: pv.PolyData
        Fitted surface (grid mm). ``field_data["n_reverted"]``: number of
        vertex moves the fold guard undid, summed over the iterations (0 for
        most heads).

    interpolator: RegularGridInterpolator
        Signed distance field of the mask, to constrain the next layer.
    """
    solid_mask = prepare_mask(volume, threshold, z_floor, zooms)
    if not solid_mask.any():
        raise ValueError("Mask is empty (check labels, threshold and z_floor).")

    interpolator = signed_distance_interpolator(solid_mask, zooms)

    # start with a large sphere outside the mask
    com = np.asarray(center_of_mass(solid_mask)) * zooms
    if init_radius is None:
        init_radius = start_radius(solid_mask, zooms, com)
    sphere = ico_sphere() if sphere is None else sphere.copy()
    pts = np.asarray(sphere.points, dtype=float) * init_radius + com

    # connectivity never changes: everything that depends on it only once
    faces = polydata_to_faces(sphere)
    A, deg, _ = _adjacency(faces, len(pts))
    pairs = _edge_face_pairs(faces)
    n_reverted = 0

    for i in range(iterations):
        previous = pts
        normals = _vertex_normals(pts, faces)

        # current move based on target mask
        combined_d = interpolator(pts)

        # pull back if we are exiting the "outer" container
        if outer_interpolator is not None:
            combined_d = np.minimum(combined_d, outer_interpolator(pts) - outer_gap)

        # push away from inner container
        if inner_interpolator is not None:
            combined_d = np.maximum(combined_d, inner_interpolator(pts) + inner_gap)

        # damping to prevent jitter
        dampening = np.clip(np.abs(combined_d) / 5.0, 0.2, 1.0)
        amount = np.clip(combined_d, -2, 2) * step_size * dampening
        pts = pts + normals * amount[:, None]

        # smooth at each iteration to keep the mesh regular
        pts = _laplacian_smooth(pts, A, deg, n_smooth, relaxation=0.05)
        if i % 10 == 0:
            # prevent triangle stretching in complex areas like the skull base
            pts = _taubin_smooth(pts, A, deg, n_iter=5)

        # fold guard: undo this iteration where it folded the surface
        pts, n = _revert_folds(pts, previous, faces, pairs, A)
        n_reverted += n

    sphere.points = pts                    # keeps sphere.faces untouched
    sphere.field_data["n_reverted"] = np.array([n_reverted])

    return sphere, interpolator


def start_radius(
    mask: np.ndarray,
    zooms: tuple[float, float, float] = (1.0, 1.0, 1.0),
    center: np.ndarray | None = None,
    margin: float = 5.0,
) -> float:
    """
    Radius (mm) of a sphere around ``center`` (grid mm, default: the mask's
    centre of mass) that encloses every voxel of ``mask``, plus ``margin`` mm.

    Starting the fit just outside the mask keeps the distance the surface has
    to travel, and how much it is compressed on the way, as small as possible.
    """
    zooms = np.asarray(zooms, dtype=float)
    if center is None:
        center = np.asarray(center_of_mass(mask)) * zooms
    # the farthest voxel is on the boundary: corners of the boundary voxels are enough
    boundary = mask & ~binary_erosion(mask)
    corners = (np.argwhere(boundary)[:, None, :] + np.array(list(np.ndindex(2, 2, 2))) - 0.5) * zooms
    return float(np.linalg.norm(corners.reshape(-1, 3) - center, axis=1).max() + margin)


def _revert_folds(pts, previous, faces, pairs, A, retries: int = 3) -> tuple[np.ndarray, int]:
    """
    Put the vertices of folded triangles back where they were before the
    iteration (``previous``, fold-free), growing the reverted region by one
    ring of neighbours per retry, and the whole mesh as a last resort.

    Returns the fold-free positions and the number of vertices reverted.
    """
    folded = _folded(pts, faces, pairs)
    if not folded.any():
        return pts, 0

    revert = np.zeros(len(pts), dtype=bool)
    revert[faces[folded].ravel()] = True
    for _ in range(retries):
        pts = np.where(revert[:, None], previous, pts)
        folded = _folded(pts, faces, pairs)
        if not folded.any():
            return pts, int(revert.sum())
        revert[faces[folded].ravel()] = True
        revert |= (A @ revert.astype(float)) > 0          # and their neighbours

    return previous, len(pts)


def grid_to_world(img: nib.spatialimages.SpatialImage) -> np.ndarray:
    """4x4 transform from grid mm (voxel index * voxel size) to scanner RAS."""
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)
    return img.affine @ np.diag([*(1 / zooms), 1.0])


def surface_to_mask(surface: pv.PolyData, img: nib.spatialimages.SpatialImage) -> np.ndarray:
    """
    Fill a closed surface on the voxel grid of an image.

    The inverse of the fit: turns e.g. a BEM layer back into a mask, to combine
    surfaces with masks (unions, margins) before fitting a new surface.

    Parameters
    ----------
    surface: pv.PolyData
        Closed surface in scanner RAS (mm), the frame of ``img``'s affine.
        For surfaces in FreeSurfer's tkRAS (e.g. written with ``fs_t1``),
        transform them back first: ``transform_layers(layers,
        np.linalg.inv(ras_to_tkras(fs_t1)))``.

    img: nibabel image
        Defines the voxel grid; any orientation and voxel size.

    Returns
    -------
    mask: np.ndarray
        Boolean array of ``img.shape[:3]``: True for voxels inside the surface.
    """
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    mesh = surface.transform(np.linalg.inv(grid_to_world(img)), inplace=False)    # -> grid mm
    grid = pv.ImageData(dimensions=img.shape[:3], spacing=zooms, origin=(0.0, 0.0, 0.0))
    mask = mesh.voxelize_binary_mask(reference_volume=grid)

    return np.asarray(mask.point_data["mask"]).reshape(img.shape[:3], order="F").astype(bool)


def fit_surface_to_mask(
    mask: nib.spatialimages.SpatialImage,
    **kwargs,
) -> tuple[pv.PolyData, RegularGridInterpolator]:
    """
    Fit an icosahedral sphere to an MRI mask and return it in world (scanner
    RAS, mm) coordinates.

    Parameters
    ----------
    mask: nibabel image
        Mask volume with its voxel -> RAS affine. Must already be RAS-oriented
        (see ``to_ras``) if ``z_floor`` or the returned interpolator are used.

    kwargs: dict
        See ``fit_surface_to_volume``.

    Returns
    -------
    surface: pv.PolyData
        Fitted surface in scanner RAS (mm), outward normals.

    interpolator: RegularGridInterpolator
        Signed distance field of the mask (grid mm).
    """
    volume = np.asanyarray(mask.dataobj)
    zooms = tuple(float(z) for z in mask.header.get_zooms()[:3])

    surface, interp = fit_surface_to_volume(volume, zooms=zooms, **kwargs)

    # grid mm -> scanner RAS
    trans = grid_to_world(mask)
    surface.transform(trans, inplace=True)

    # fix the winding order and normals
    if np.linalg.det(trans) < 0:
        surface.flip_faces(inplace=True)

    surface.compute_normals(inplace=True, consistent_normals=True, auto_orient_normals=False)

    return surface, interp
