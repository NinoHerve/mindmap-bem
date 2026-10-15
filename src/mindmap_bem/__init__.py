"""Nested, MNE-compatible BEM surfaces from head segmentations."""

from mindmap_bem.bem import build_bem, build_bem_layers, transform_layers
from mindmap_bem.charm import bem_from_t1, charm_labeling, run_charm
from mindmap_bem.fitting import fit_surface_to_mask, fit_surface_to_volume
from mindmap_bem.io import (
    make_volume_info,
    ras_to_tkras,
    read_bem_surfaces,
    setup_mne_subject,
    write_bem_surfaces,
)
from mindmap_bem.labels import list_presets, read_bem_labels
from mindmap_bem.quality import check_bem, layer_gaps
from mindmap_bem import datasets, viz

__version__ = "0.1.0"

__all__ = [
    "datasets",
    "viz",
    "check_bem",
    "layer_gaps",
    "bem_from_t1",
    "run_charm",
    "charm_labeling",
    "build_bem",
    "build_bem_layers",
    "transform_layers",
    "fit_surface_to_mask",
    "fit_surface_to_volume",
    "make_volume_info",
    "ras_to_tkras",
    "read_bem_surfaces",
    "setup_mne_subject",
    "write_bem_surfaces",
    "list_presets",
    "read_bem_labels",
]
