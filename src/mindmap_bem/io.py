"""
Module Name: io.py
Description: Mesh conversion and FreeSurfer/MNE surface I/O.

Author: Nino Herve
License: MIT
"""

from pathlib import Path

import numpy as np
import nibabel as nib
import pyvista as pv

# BEM layer -> file name expected by mne.make_bem_model
BEM_FILES = {
    "inner_skull": "inner_skull.surf",
    "outer_skull": "outer_skull.surf",
    "skin": "outer_skin.surf",
}


def surf_to_polydata(vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    """
    Converts a list of vertices and faces to a PolyData structure.
    """
    # pyvista faces are a flat [..., n, v1, ..., vn, ...] array
    pv_faces = np.empty((faces.shape[0], faces.shape[1] + 1), dtype=np.int64)
    pv_faces[:, 1:] = faces
    pv_faces[:, 0] = faces.shape[1]

    return pv.PolyData(vertices, pv_faces.ravel())


def polydata_to_faces(mesh: pv.PolyData) -> np.ndarray:
    """(n_faces, 3) triangle array of a fully triangulated PolyData."""
    return mesh.faces.reshape(-1, 4)[:, 1:]


def polydata_transform(polydata: pv.PolyData, trans: np.ndarray) -> pv.PolyData:
    """
    Apply a 4x4 transform on PolyData vertices, keeping outward normals.
    """
    polydata.transform(trans, inplace=True)

    if np.linalg.det(trans) < 0:
        polydata.flip_faces(inplace=True)

    polydata.compute_normals(inplace=True, consistent_normals=False, auto_orient_normals=False)

    return polydata


def ras_to_tkras(t1: str | Path | nib.MGHImage) -> np.ndarray:
    """
    Scanner RAS -> FreeSurfer surface RAS (tkRAS) transform of the subject's
    FreeSurfer ``mri/T1.mgz``.

    Only needed to combine the BEM with a FreeSurfer reconstruction (whose
    surfaces, and MNE's "MRI" frame, are in that recon's tkRAS).
    """
    mri = nib.load(t1) if isinstance(t1, (str, Path)) else t1
    if not isinstance(mri, nib.MGHImage):
        raise TypeError("ras_to_tkras expects a FreeSurfer .mgz volume (e.g. mri/T1.mgz).")

    return mri.header.get_vox2ras_tkr() @ np.linalg.inv(mri.header.get_vox2ras())


def make_volume_info(t1: str | Path) -> dict:
    """
    Volume geometry of a T1 (e.g. FreeSurfer's T1.mgz), to embed in written
    surfaces so that freeview places them correctly over that volume.
    """
    img = nib.load(t1)

    volume = np.array(img.shape[:3])
    voxelsize = np.array(img.header.get_zooms()[:3], dtype=float)

    # direction cosines: affine columns without the voxel size
    mdc = img.affine[:3, :3] / voxelsize

    # "cras": scanner coordinates of the volume centre (FreeSurfer's c_ras)
    cras = nib.affines.apply_affine(img.affine, volume / 2.0)

    return {
        "head": np.array([2, 0, 20], dtype=np.int32),   # generic header tag
        "valid": "1  # volume info valid",
        "filename": str(t1),
        "volume": volume,
        "voxelsize": voxelsize,
        "xras": mdc[:, 0],
        "yras": mdc[:, 1],
        "zras": mdc[:, 2],
        "cras": cras,
    }


def setup_mne_subject(
    t1: str | Path | nib.spatialimages.SpatialImage,
    subjects_dir: str | Path,
    subject: str,
) -> Path:
    """
    Make sure ``subjects_dir/subject/mri/T1.mgz`` exists, creating it from ``t1``.

    MNE locates a subject's MRI through this file and uses its FreeSurfer
    surface RAS (tkRAS) as the "MRI" coordinate frame. With it in place, MNE
    works without FreeSurfer: write the BEM with
    ``build_bem(..., fs_t1=<returned path>)``.

    An existing ``T1.mgz`` (e.g. from a FreeSurfer recon of the same scan) is
    kept as is.

    Returns
    -------
    t1_mgz: Path
    """
    t1_mgz = Path(subjects_dir)/subject/"mri"/"T1.mgz"
    if t1_mgz.exists():
        return t1_mgz

    img = nib.load(t1) if isinstance(t1, (str, Path)) else t1

    # FreeSurfer's tkRAS formula assumes LIA voxel order (as in conformed
    # volumes); in any other order scanner -> tkRAS is a reflection, which
    # would flip the BEM triangles and break MNE's ico decimation.
    # Reordering the axes keeps the image itself unchanged.
    ornt = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(img.affine), nib.orientations.axcodes2ornt("LIA"),
    )
    img = img.as_reoriented(ornt)

    t1_mgz.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.MGHImage(np.asanyarray(img.dataobj, dtype=np.float32), img.affine), t1_mgz)

    return t1_mgz


def write_bem_surfaces(
    folder: str | Path,
    layers: dict[str, pv.PolyData],
    volume_info: dict | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """
    Write BEM layers as FreeSurfer surfaces (``inner_skull.surf``,
    ``outer_skull.surf``, ``outer_skin.surf``), the format and names
    ``mne.make_bem_model`` reads.

    ``folder`` is typically ``$SUBJECTS_DIR/<subject>/bem``. Pass
    ``volume_info`` (see ``make_volume_info``) only when the surfaces are in
    that volume's FreeSurfer tkRAS.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    paths = {layer: folder/fname for layer, fname in BEM_FILES.items()}
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"BEM surfaces already exist (use overwrite=True): {existing}")

    for layer, path in paths.items():
        nib.freesurfer.io.write_geometry(
            path, layers[layer].points, polydata_to_faces(layers[layer]), volume_info=volume_info,
        )

    return paths


def read_bem_surfaces(folder: str | Path) -> dict[str, pv.PolyData]:
    """Read BEM surfaces written by ``write_bem_surfaces`` (or FreeSurfer)."""
    folder = Path(folder)

    layers = {}
    for layer, fname in BEM_FILES.items():
        vertices, faces = nib.freesurfer.io.read_geometry(folder/fname)
        layers[layer] = surf_to_polydata(vertices, faces)

    return layers
