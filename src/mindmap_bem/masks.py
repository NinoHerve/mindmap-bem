"""
Module Name: masks.py
Description: Binary mask operations used before surface fitting.

All lengths are in mm; ``zooms`` gives the voxel size along each axis.

Author: Nino Herve
License: MIT
"""

import numpy as np
from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter
from scipy.interpolate import RegularGridInterpolator


def labels_to_mask(seg: np.ndarray, labels: list[int]) -> np.ndarray:
    """Boolean mask of voxels whose value is in ``labels``."""
    return np.isin(seg, labels)


def close_mask(
    mask: np.ndarray,
    radius: float = 10.0,
    zooms: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Binary morphological closing with a ball of ``radius`` mm.

    Implemented with Euclidean distance transforms. The ball includes voxels
    within ``radius`` plus half a voxel (ITK's convention) and the volume is
    zero-padded, which reproduces ANTs' ``image.morphology("close", radius)``
    exactly on 1 mm isotropic images.
    """
    if radius <= 0:
        return mask.astype(bool)

    zooms = np.asarray(zooms, dtype=float)
    pad = int(np.ceil(radius / zooms.min())) + 1
    m = np.pad(mask.astype(bool), pad)
    r = radius + 0.5 * zooms.min()

    dilated = distance_transform_edt(~m, sampling=zooms) <= r
    closed = distance_transform_edt(dilated, sampling=zooms) > r

    return closed[pad:-pad, pad:-pad, pad:-pad]


def fill_mask_all_axes(mask: np.ndarray) -> np.ndarray:
    """
    Applies binary_fill_holes across all three primary axes to seal
    complex 3D tunnels.
    """
    filled = mask.astype(bool)
    for axis in range(3):
        for i in range(filled.shape[axis]):
            slc = [slice(None)] * 3     # handles any axis
            slc[axis] = i
            filled[tuple(slc)] = binary_fill_holes(filled[tuple(slc)])

    return filled


def prepare_mask(
    volume: np.ndarray,
    threshold: float = 0.5,
    z_floor: float | None = None,
    zooms: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Binarize, fill holes and cap a mask before fitting.

    Parameters
    ----------
    volume: np.ndarray
        Mask (or soft mask) volume, RAS-oriented (third axis inferior -> superior).

    threshold: float
        Voxels above threshold are inside.

    z_floor: float | None
        Everything less than ``z_floor`` mm above the bottom of the volume is
        removed. Caps the surface above the neck.

    zooms: tuple
        Voxel size (mm).
    """
    solid_mask = fill_mask_all_axes(volume > threshold)
    if z_floor is not None:
        solid_mask[:, :, :int(round(z_floor / zooms[2]))] = 0    # soft floor

    return solid_mask


def signed_distance_interpolator(
    solid_mask: np.ndarray,
    zooms: tuple[float, float, float] = (1.0, 1.0, 1.0),
    sigma: float = 1.2,
) -> RegularGridInterpolator:
    """
    Smoothed signed distance map (mm) of a mask.

    Positive inside the mask, negative outside. Sampled in "grid mm"
    coordinates (voxel index * voxel size). Used both as the force field
    pulling a surface onto the mask and as a containment constraint for the
    neighbouring layers.

    Parameters
    ----------
    solid_mask: np.ndarray
        Boolean mask.

    zooms: tuple
        Voxel size (mm).

    sigma: float
        Gaussian smoothing of the distance map (mm).
    """
    zooms = np.asarray(zooms, dtype=float)

    # This guides the vertices much more smoothly than raw intensities
    dist_map = (
        distance_transform_edt(solid_mask, sampling=zooms)
        - distance_transform_edt(~solid_mask, sampling=zooms)
    )
    dist_map = gaussian_filter(dist_map, sigma=sigma / zooms)

    # allows to sample at non-integer vertex positions
    coords = [np.arange(s) * z for s, z in zip(dist_map.shape, zooms)]
    return RegularGridInterpolator(coords, dist_map, bounds_error=False, fill_value=None)
