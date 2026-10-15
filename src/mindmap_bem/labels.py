"""
Module Name: labels.py
Description: Map segmentation labels to BEM compartments.

A label config has two sections:

- ``label_map``: readable tissue category -> list of integer segmentation IDs.
- ``layers``: BEM layer -> list of categories *or other layers* (inheritance),
  so that each layer contains everything enclosed by the layer below it.

Author: Nino Herve
License: MIT
"""

from importlib import resources
from pathlib import Path

import yaml

LAYERS = ("inner_skull", "outer_skull", "skin")


def list_presets() -> list[str]:
    """Return the names of the bundled label presets."""
    files = resources.files("mindmap_bem").joinpath("data").iterdir()
    return sorted(f.name.removesuffix(".yml") for f in files if f.name.endswith(".yml"))


def read_bem_labels(config: str | Path | dict) -> dict[str, list[int]]:
    """
    Resolve a label config into ``{layer_name: [label_ids]}``.

    Parameters
    ----------
    config: str | Path | dict
        A preset name (see ``list_presets()``, e.g. ``"charm"``), a path to a
        YAML file, or an already parsed config dict.

    Returns
    -------
    labels: dict[str, list[int]]
        Sorted, unique label IDs enclosed by each layer.
    """
    if isinstance(config, (str, Path)):
        config = _load_config(config)

    label_map = config.get("label_map", {})
    layers_def = config.get("layers", {})

    missing = [layer for layer in LAYERS if layer not in layers_def]
    if missing:
        raise KeyError(f"Label config is missing layers: {missing}")

    def resolve_item(item: str, seen: tuple[str, ...]) -> list[int]:
        # category in label_map
        if item in label_map:
            return list(label_map[item])

        # reference to another layer (inheritance)
        if item in layers_def:
            if item in seen:
                raise ValueError(f"Circular layer inheritance: {' -> '.join(seen + (item,))}")
            indices = []
            for sub_item in layers_def[item]:
                indices.extend(resolve_item(sub_item, seen + (item,)))
            return indices

        raise KeyError(f"'{item}' not found in label_map or layers.")

    return {layer: sorted(set(resolve_item(layer, ()))) for layer in layers_def}


def _load_config(config: str | Path) -> dict:
    path = Path(config)
    if path.suffix in (".yml", ".yaml") or path.exists():
        with open(path) as f:
            return yaml.safe_load(f)

    if str(config) not in list_presets():
        raise ValueError(f"Unknown label preset '{config}'. Available: {list_presets()}")

    text = resources.files("mindmap_bem").joinpath("data", f"{config}.yml").read_text()
    return yaml.safe_load(text)
