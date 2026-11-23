"""
Module Name: fitting.py
Description: Deform an icosahedral sphere onto a binary mask.

The surface shrink-wraps the mask by following a smoothed signed distance
field, optionally constrained to stay inside an outer envelope and/or outside
an inner one. Every layer is the same deformed icosahedron, so its
connectivity is always closed and manifold, and nesting is enforced during the
fit rather than repaired afterwards. The fit can still fold or miss its target,
so ``build_bem`` checks the result (``check_bem``) before writing it.

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
from scipy.ndimage import center_of_mass

from mindmap_bem.io import surf_to_polydata
from mindmap_bem.masks import prepare_mask, signed_distance_interpolator


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
    min_buffer: float = 3.0,
    max_buffer: float = 3.0,
    init_radius: float = 120.0,
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
        least ``min_buffer`` mm.

    inner_interpolator: RegularGridInterpolator | None
        Signed distance of a volume the surface must stay outside of, by at
        least ``max_buffer`` mm.

    init_radius: float
        Radius (mm) of the starting sphere, centred on the mask.

    sphere: pv.PolyData | None
        Unit sphere to deform. Defaults to ``ico_sphere()``.

    Returns
    -------
    sphere: pv.PolyData
        Fitted surface (grid mm).

    interpolator: RegularGridInterpolator
        Signed distance field of the mask, to constrain the next layer.
    """
    solid_mask = prepare_mask(volume, threshold, z_floor, zooms)
    if not solid_mask.any():
        raise ValueError("Mask is empty (check labels, threshold and z_floor).")

    interpolator = signed_distance_interpolator(solid_mask, zooms)

    # start with a large sphere outside the mask
    com = np.asarray(center_of_mass(solid_mask)) * zooms
    sphere = ico_sphere() if sphere is None else sphere.copy()
    sphere.points = sphere.points * init_radius + com

    for i in range(iterations):
        sphere.compute_normals(inplace=True, consistent_normals=False)
        vertices = sphere.points
        normals = sphere.point_data["Normals"]

        # current move based on target mask
        combined_d = interpolator(vertices)

        # pull back if we are exiting the "outer" container
        if outer_interpolator is not None:
            combined_d = np.minimum(combined_d, outer_interpolator(vertices) - min_buffer)

        # push away from inner container
        if inner_interpolator is not None:
            combined_d = np.maximum(combined_d, inner_interpolator(vertices) + max_buffer)

        move = np.clip(combined_d, -2, 2)

        # damping to prevent jitter
        dampening = np.clip(np.abs(combined_d) / 5.0, 0.2, 1.0)
        sphere.points += normals * (move[:, np.newaxis] * step_size * dampening[:, np.newaxis])

        # smooth at each iteration to preserve manifold
        smoothed = sphere.smooth(n_iter=n_smooth, relaxation_factor=0.05)
        sphere.points = smoothed.points        # keeps sphere.faces untouched

        if i % 10 == 0:
            # prevent triangle stretching in complex areas like the skull base
            taubin = sphere.smooth_taubin(n_iter=5, pass_band=0.1)
            sphere.points = taubin.points

    return sphere, interpolator


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
