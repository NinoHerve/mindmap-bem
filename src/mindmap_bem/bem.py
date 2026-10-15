"""
Module Name: bem.py
Description: Build nested 3-layer BEM surfaces from a head segmentation.

Layers are fitted from the inside out: the inner skull is pushed away from the
brain (optional), the outer skull away from the inner skull, and the skin away
from the outer skull. Each layer therefore encloses the previous one by
construction.

Author: Nino Herve
License: MIT
"""

import warnings
from pathlib import Path

import numpy as np
import nibabel as nib
import pyvista as pv

from mindmap_bem.fitting import fit_surface_to_mask, to_ras
from mindmap_bem.io import make_volume_info, polydata_transform, ras_to_tkras, write_bem_surfaces
from mindmap_bem.labels import read_bem_labels
from mindmap_bem.masks import close_mask, labels_to_mask, prepare_mask, signed_distance_interpolator

# mm above the bottom of the field of view removed below each layer
# (caps surfaces above the neck)
DEFAULT_Z_FLOORS = {"inner_skull": 45.0, "outer_skull": 35.0, "skin": 20.0}


def build_bem_layers(
    seg: nib.spatialimages.SpatialImage,
    labels: str | Path | dict = "charm",
    brain_mask: nib.spatialimages.SpatialImage | None = None,
    closing_radius: float = 10.0,
    z_floors: dict[str, float | None] | None = None,
    n_smooth: int = 20,
    **kwargs,
) -> dict[str, pv.PolyData]:
    """
    Build BEM layers from a segmentation.

    Parameters
    ----------
    seg: nibabel image
        Integer label volume (e.g. CHARM's ``segmentation/labeling.nii.gz``),
        any orientation and voxel size.

    labels: str | Path | dict
        Label preset name, YAML config path, parsed config, or an already
        resolved ``{layer: [ids]}`` dict. See ``mindmap_bem.labels``.

    brain_mask: nibabel image | None
        Brain mask on the same voxel grid as ``seg``. If given, the inner skull
        is kept at least ``max_buffer`` mm outside of it.

    closing_radius: float
        Radius (mm) of the morphological closing applied to each mask.

    z_floors: dict | None
        Per-layer floor (mm above the bottom of the field of view), see
        ``DEFAULT_Z_FLOORS``. Set a layer to None to disable.

    n_smooth: int
        Laplacian smoothing iterations per fitting step.

    kwargs: dict
        Passed to ``fit_surface_to_volume`` (iterations, step_size, ...).

    Returns
    -------
    layers: dict[str, pv.PolyData]
        ``inner_skull``, ``outer_skull`` and ``skin`` surfaces in scanner RAS (mm).
    """
    if not (isinstance(labels, dict) and "layers" not in labels):
        labels = read_bem_labels(labels)

    z_floors = {**DEFAULT_Z_FLOORS, **(z_floors or {})}

    # fit on a RAS grid so that the z floors act along inferior -> superior
    seg = to_ras(seg)
    seg_data = np.asanyarray(seg.dataobj)
    zooms = tuple(float(z) for z in seg.header.get_zooms()[:3])

    def layer_mask(layer):
        mask = close_mask(labels_to_mask(seg_data, labels[layer]), closing_radius, zooms)
        return nib.Nifti1Image(mask.astype(np.uint8), seg.affine)

    # brain constraint (only its distance field is needed)
    brain_interp = None
    if brain_mask is not None:
        brain_mask = to_ras(brain_mask)
        _check_same_grid(seg, brain_mask)
        brain = close_mask(np.asanyarray(brain_mask.dataobj) > 0.5, closing_radius, zooms)
        brain_interp = signed_distance_interpolator(prepare_mask(brain, zooms=zooms), zooms)

    # inner skull
    iskull_mesh, iskull_interp = fit_surface_to_mask(
        layer_mask("inner_skull"), inner_interpolator=brain_interp,
        z_floor=z_floors["inner_skull"], n_smooth=n_smooth, **kwargs,
    )

    # outer skull
    oskull_mesh, oskull_interp = fit_surface_to_mask(
        layer_mask("outer_skull"), inner_interpolator=iskull_interp,
        z_floor=z_floors["outer_skull"], n_smooth=n_smooth, **kwargs,
    )

    # skin
    skin_mesh, _ = fit_surface_to_mask(
        layer_mask("skin"), inner_interpolator=oskull_interp,
        z_floor=z_floors["skin"], n_smooth=n_smooth, **kwargs,
    )

    return {
        "inner_skull": iskull_mesh,
        "outer_skull": oskull_mesh,
        "skin": skin_mesh,
    }


def transform_layers(layers: dict[str, pv.PolyData], trans: np.ndarray) -> dict[str, pv.PolyData]:
    """Apply a 4x4 transform to every BEM layer (in place)."""
    for k in layers:
        layers[k] = polydata_transform(layers[k], trans)

    return layers


def build_bem(
    seg: str | Path | nib.spatialimages.SpatialImage,
    out_dir: str | Path,
    labels: str | Path | dict = "charm",
    brain_mask: str | Path | nib.spatialimages.SpatialImage | None = None,
    fs_t1: str | Path | None = None,
    overwrite: bool = False,
    **kwargs,
) -> dict[str, pv.PolyData]:
    """
    Segmentation in, MNE-readable surfaces out.

    Fits the layers and writes ``inner_skull.surf``, ``outer_skull.surf`` and
    ``outer_skin.surf`` to ``out_dir`` (for MNE: ``$SUBJECTS_DIR/<subject>/bem``).

    Parameters
    ----------
    seg: path or nibabel image
        Integer label volume.

    out_dir: path
        Output folder.

    fs_t1: path | None
        Coordinate frame of the output surfaces:

        - None (default): scanner RAS of the segmentation. No FreeSurfer
          needed; compute your head -> MRI ``trans`` against these surfaces.
        - path to a FreeSurfer recon's ``mri/T1.mgz`` (same subject, same
          scanner space): that recon's tkRAS, i.e. MNE's "MRI" frame when the
          source space comes from FreeSurfer.

    labels, brain_mask, kwargs:
        See ``build_bem_layers``.

    Returns
    -------
    layers: dict[str, pv.PolyData]
        Layers in the output frame (mm).
    """
    seg = nib.load(seg) if isinstance(seg, (str, Path)) else seg
    if isinstance(brain_mask, (str, Path)):
        brain_mask = nib.load(brain_mask)

    layers = build_bem_layers(seg, labels, brain_mask=brain_mask, **kwargs)

    volume_info = None
    if fs_t1 is not None:
        trans = ras_to_tkras(fs_t1)
        if np.linalg.det(trans[:3, :3]) < 0:
            warnings.warn(
                f"{fs_t1} is not in FreeSurfer's LIA voxel order, so its tkRAS is a mirror of "
                "scanner RAS. The surfaces are valid but MNE cannot decimate them "
                "(use ico=None in mne.make_bem_model). Create the file with setup_mne_subject instead.",
                stacklevel=2,
            )
        layers = transform_layers(layers, trans)
        volume_info = make_volume_info(fs_t1)

    write_bem_surfaces(out_dir, layers, volume_info, overwrite=overwrite)

    return layers


def _check_same_grid(a, b):
    if a.shape[:3] != b.shape[:3] or not np.allclose(a.affine, b.affine, atol=1e-3):
        raise ValueError(
            "brain_mask must be on the same voxel grid as the segmentation "
            f"(shapes {a.shape[:3]} vs {b.shape[:3]}); resample it first."
        )
