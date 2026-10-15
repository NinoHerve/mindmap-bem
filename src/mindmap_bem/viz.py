"""
Module Name: viz.py
Description: Look at BEM surfaces with the viewer of your choice.

Backends:

- ``"pyvista"``: interactive 3D (surfaces + orthogonal MRI slices).
- ``"matplotlib"``: surface outlines over MRI slices (needs ``[viz]``).
- ``"freeview"``: FreeSurfer's viewer (needs FreeSurfer installed).
- ``"mne"``: ``mne.viz.plot_bem`` (needs ``[mne]``).

Author: Nino Herve
License: MIT
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
import pyvista as pv

from mindmap_bem.io import (
    BEM_FILES,
    make_volume_info,
    polydata_transform,
    ras_to_tkras,
    read_bem_surfaces,
    setup_mne_subject,
    write_bem_surfaces,
)

BACKENDS = ("pyvista", "matplotlib", "freeview", "mne")
COLORS = {"inner_skull": "#e4572e", "outer_skull": "#f3a712", "skin": "#29bf12"}


def plot_bem(
    bem: str | Path | dict[str, pv.PolyData],
    t1: str | Path | nib.spatialimages.SpatialImage,
    backend: str = "pyvista",
    fs_t1: str | Path | None = None,
    **kwargs,
):
    """
    Plot BEM surfaces over the MRI they were built from.

    Parameters
    ----------
    bem: path | dict
        BEM folder (``*.surf`` files) or layers from ``build_bem_layers``.

    t1: path | nibabel image
        The T1 the BEM was built from (scanner RAS).

    backend: str
        One of ``BACKENDS``.

    fs_t1: path | None
        If the surfaces were written in a FreeSurfer recon's tkRAS
        (``build_bem(..., fs_t1=...)``), that recon's ``mri/T1.mgz``.

    kwargs: dict
        Passed to the backend (see ``plot_pyvista``, ``plot_slices``,
        ``open_freeview``, ``plot_mne``).

    Returns
    -------
    The backend's figure/plotter, or the freeview process.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got '{backend}'")

    layers = read_bem_surfaces(bem) if isinstance(bem, (str, Path)) else {k: v.copy() for k, v in bem.items()}
    t1 = nib.load(t1) if isinstance(t1, (str, Path)) else t1

    # work in the T1's scanner RAS
    if fs_t1 is not None:
        for k in layers:
            layers[k] = polydata_transform(layers[k], np.linalg.inv(ras_to_tkras(fs_t1)))

    plotters = {"pyvista": plot_pyvista, "matplotlib": plot_slices, "freeview": open_freeview, "mne": plot_mne}
    return plotters[backend](layers, t1, **kwargs)


# --- pyvista --------------------------------------------------------------

def plot_pyvista(
    layers: dict[str, pv.PolyData],
    t1: nib.spatialimages.SpatialImage,
    cutaway: bool = True,
    show_mri: bool | None = None,
    opacity: float | None = None,
    show: bool = True,
    **plotter_kwargs,
) -> pv.Plotter:
    """
    3D view of the surfaces.

    Parameters
    ----------
    cutaway: bool
        Clip the surfaces at the mid-sagittal plane (right half removed) so the
        nested layers are visible.

    show_mri: bool | None
        Show three orthogonal MRI slices. Default: only without cutaway.

    opacity: float | None
        Surface opacity. Default: 1 with cutaway, 0.35 without.
    """
    p = pv.Plotter(**plotter_kwargs)
    opacity = (1.0 if cutaway else 0.35) if opacity is None else opacity
    show_mri = (not cutaway) if show_mri is None else show_mri
    center = np.asarray(layers["skin"].center) if "skin" in layers else np.zeros(3)

    if show_mri:
        grid = _to_image_data(t1)
        p.add_mesh(grid.slice_orthogonal(x=center[0], y=center[1], z=center[2]), cmap="gray",
                   show_scalar_bar=False, clim=np.percentile(grid["T1"], [1, 99.5]))

    for k, mesh in layers.items():
        if cutaway:
            mesh = mesh.clip(normal="x", origin=center)     # keeps the left half
        p.add_mesh(mesh, color=COLORS.get(k), opacity=opacity, label=k, smooth_shading=True)

    p.add_legend(bcolor=None, face="rectangle")
    p.camera_position = "yz"                # look from the right, at the cut
    p.camera.azimuth = 35
    p.camera.elevation = 15
    if show:
        p.show()

    return p


def _to_image_data(img: nib.spatialimages.SpatialImage) -> pv.ImageData:
    """nibabel volume -> pyvista ImageData in scanner RAS."""
    img = nib.as_closest_canonical(img)
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=float)

    grid = pv.ImageData(dimensions=img.shape[:3], spacing=zooms, origin=img.affine[:3, 3])
    grid.direction_matrix = img.affine[:3, :3] / zooms
    grid.point_data["T1"] = np.asanyarray(img.dataobj, dtype=np.float32).ravel(order="F")

    return grid


# --- matplotlib -----------------------------------------------------------

def plot_slices(
    layers: dict[str, pv.PolyData],
    t1: nib.spatialimages.SpatialImage,
    n_slices: int = 5,
    views: tuple[str, ...] = ("sagittal", "coronal", "axial"),
    extent: tuple[float, float] = (0.2, 0.8),
    linewidth: float = 1.2,
    colors: dict[str, str] | None = None,
):
    """
    Surface outlines over evenly spaced MRI slices.

    Parameters
    ----------
    n_slices: int
        Slices per view.

    views: tuple[str]
        Any of "sagittal", "coronal", "axial" (one row each).

    extent: tuple[float, float]
        Fraction of the volume spanned by the slices along each view axis.

    colors: dict | None
        Line color per layer name.
    """
    colors = {**COLORS, **(colors or {})}
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("The matplotlib backend needs matplotlib: pip install 'mindmap-bem[viz]'") from e

    img = nib.as_closest_canonical(t1)
    data = np.asanyarray(img.dataobj, dtype=np.float32)
    vmax = np.percentile(data, 99.5)
    world_to_vox = np.linalg.inv(img.affine)
    axis = {"sagittal": 0, "coronal": 1, "axial": 2}

    fig, axes = plt.subplots(len(views), n_slices, figsize=(2.6 * n_slices, 2.8 * len(views)), squeeze=False)

    for row, view in enumerate(views):
        k = axis[view]
        keep = [a for a in range(3) if a != k]     # in-plane voxel axes
        indices = np.linspace(*extent, n_slices) * (data.shape[k] - 1)

        for col, idx in enumerate(np.round(indices).astype(int)):
            ax = axes[row, col]
            image = np.take(data, idx, axis=k)
            ax.imshow(image.T, origin="lower", cmap="gray", vmin=0, vmax=vmax)

            # slicing plane in world coordinates
            center = np.array(data.shape) / 2.0
            center[k] = idx
            origin = nib.affines.apply_affine(img.affine, center)
            normal = img.affine[:3, k]

            for name, mesh in layers.items():
                cut = mesh.slice(normal=normal, origin=origin)
                if cut.n_points == 0:
                    continue
                vox = nib.affines.apply_affine(world_to_vox, cut.points)[:, keep]
                for line in _polylines(cut):
                    ax.plot(vox[line, 0], vox[line, 1], color=colors.get(name), lw=linewidth)

            ax.set_title(f"{view} {idx}", fontsize=8)
            ax.set_axis_off()

    handles = [plt.Line2D([], [], color=colors.get(k), label=k) for k in layers]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    return fig


def plot_comparison(
    bems: dict[str, dict[str, pv.PolyData]],
    t1: str | Path | nib.spatialimages.SpatialImage,
    layer: str = "inner_skull",
    n_slices: int = 5,
    views: tuple[str, ...] = ("sagittal", "coronal", "axial"),
    extent: tuple[float, float] = (0.2, 0.8),
    linewidth: float = 1.2,
):
    """
    Outlines of the same layer from several BEMs over MRI slices.

    Parameters
    ----------
    bems: dict
        ``{label: layers}``, all in the T1's scanner RAS (or ``{label: bem_dir}``).

    layer: str
        Which layer to compare ("inner_skull", "outer_skull" or "skin").
    """
    bems = {k: read_bem_surfaces(v) if isinstance(v, (str, Path)) else v for k, v in bems.items()}
    t1 = nib.load(t1) if isinstance(t1, (str, Path)) else t1

    import matplotlib.pyplot as plt
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    styled = {label: (b[layer], colors[i % len(colors)]) for i, (label, b) in enumerate(bems.items())}

    fig = plot_slices(
        {label: mesh for label, (mesh, _) in styled.items()}, t1, n_slices, views, extent, linewidth,
        colors={label: c for label, (_, c) in styled.items()},
    )
    fig.suptitle(layer, fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.93)
    return fig


def _polylines(cut: pv.PolyData) -> list[np.ndarray]:
    """Point index arrays of each line cell of a sliced mesh."""
    lines, out, i = cut.lines, [], 0
    while i < len(lines):
        n = lines[i]
        out.append(lines[i + 1 : i + 1 + n])
        i += n + 1
    return out


# --- FreeSurfer-space helpers (freeview, MNE) ------------------------------

def _tk_subject(layers, t1, subjects_dir: Path, subject: str = "bem_view") -> Path:
    """
    Lay out ``subjects_dir/subject/{mri/T1.mgz, bem/*.surf}`` with surfaces in
    the T1's tkRAS, the frame freeview and MNE expect.
    """
    mgz = setup_mne_subject(t1, subjects_dir, subject)

    tk_layers = {k: polydata_transform(v.copy(), ras_to_tkras(mgz)) for k, v in layers.items()}
    write_bem_surfaces(subjects_dir/subject/"bem", tk_layers, make_volume_info(mgz), overwrite=True)

    return subjects_dir/subject


def open_freeview(
    layers: dict[str, pv.PolyData],
    t1: nib.spatialimages.SpatialImage,
    workdir: str | Path | None = None,
    freeview_cmd: str = "freeview",
) -> subprocess.Popen:
    """Open freeview (non-blocking) with the T1 and the BEM outlines."""
    if shutil.which(freeview_cmd) is None:
        raise FileNotFoundError("freeview not found: install FreeSurfer and source its setup script.")

    workdir = Path(workdir or tempfile.mkdtemp(prefix="mindmap_bem_view_"))
    sub_dir = _tk_subject(layers, t1, workdir)

    cmd = [freeview_cmd, "-v", str(sub_dir/"mri"/"T1.mgz")]
    for k, fname in BEM_FILES.items():
        rgb = ",".join(str(int(COLORS[k][i:i + 2], 16)) for i in (1, 3, 5))
        cmd += ["-f", f"{sub_dir/'bem'/fname}:edgecolor={rgb}:name={k}"]

    return subprocess.Popen(cmd)


def plot_mne(
    layers: dict[str, pv.PolyData],
    t1: nib.spatialimages.SpatialImage,
    workdir: str | Path | None = None,
    **kwargs,
):
    """``mne.viz.plot_bem`` on a temporary MNE subject made from the T1."""
    try:
        import mne
    except ImportError as e:
        raise ImportError("The mne backend needs MNE: pip install 'mindmap-bem[mne]'") from e

    workdir = Path(workdir or tempfile.mkdtemp(prefix="mindmap_bem_view_"))
    sub_dir = _tk_subject(layers, t1, workdir)

    kwargs.setdefault("orientation", "coronal")
    return mne.viz.plot_bem(subject=sub_dir.name, subjects_dir=workdir, **kwargs)
