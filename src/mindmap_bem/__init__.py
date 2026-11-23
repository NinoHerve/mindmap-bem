"""Nested, MNE-compatible BEM surfaces from head segmentations."""

from mindmap_bem.bem import build_bem, build_bem_layers, transform_layers
from mindmap_bem.charm import bem_from_t1, charm_labeling, run_charm, stop_charm
from mindmap_bem.fitting import fit_surface_to_mask, fit_surface_to_volume, grid_to_world, surface_to_mask, to_ras
from mindmap_bem.io import (
    make_volume_info,
    polydata_to_faces,
    ras_to_tkras,
    read_bem_surfaces,
    setup_mne_subject,
    write_bem_surfaces,
)
from mindmap_bem.labels import list_presets, read_bem_labels
from mindmap_bem.masks import close_mask, signed_distance_interpolator
from mindmap_bem.quality import check_bem, common_leadfields, count_folds, folded_faces, layer_gaps, lnmag, rdm
from mindmap_bem import datasets, viz

__version__ = "0.1.0"

__all__ = [
    "datasets",
    "viz",
    "check_bem",
    "layer_gaps",
    "count_folds",
    "folded_faces",
    "rdm",
    "lnmag",
    "common_leadfields",
    "bem_from_t1",
    "run_charm",
    "stop_charm",
    "charm_labeling",
    "build_bem",
    "build_bem_layers",
    "transform_layers",
    "fit_surface_to_mask",
    "fit_surface_to_volume",
    "surface_to_mask",
    "close_mask",
    "signed_distance_interpolator",
    "to_ras",
    "grid_to_world",
    "polydata_to_faces",
    "make_volume_info",
    "ras_to_tkras",
    "read_bem_surfaces",
    "setup_mne_subject",
    "write_bem_surfaces",
    "list_presets",
    "read_bem_labels",
]
