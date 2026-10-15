# mindmap-bem

Nested, watertight BEM surfaces for EEG/MEG forward models, from any
T1-weighted MRI. FreeSurfer is optional.

```
T1 (+ T2) ──CHARM──▶ head segmentation ──fit──▶ inner_skull / outer_skull / outer_skin .surf ──▶ mne.make_bem_model
```

The head is segmented with [SimNIBS CHARM](https://simnibs.github.io/simnibs/).
Each BEM layer (inner skull, outer skull, skin) is then an icosahedral sphere
(MNE's ico-5, 10242 vertices) that shrink-wraps the corresponding tissue mask
by following a smoothed signed distance field. Layers are fitted from the
inside out, and each one is constrained to stay a few mm outside the
previous one, so:

- every surface is closed and manifold with the same topology,
- the layers are **nested by construction**, with no overlapping skull
  surfaces to fix afterwards (a common failure of the watershed approach),
- the output is in the format and file names `mne.make_bem_model` reads.

Extracted from [mindmap](https://github.com/NinoHerve/mindmap).

## Install

```bash
pip install "mindmap-bem @ git+https://github.com/NinoHerve/mindmap-bem"
```

Core dependencies: numpy, scipy, nibabel, pyvista, pyyaml. Optional extras:

| extra | adds | for |
|---|---|---|
| `[mne]` | MNE | building forward models, `plot_bem(backend="mne")` |
| `[viz]` | matplotlib | slice plots |
| `[tutorials]` | `[mne]` + `[viz]` + jupyter + `pyvista[jupyter]` | running the tutorials |

MNE is not pinned: install the MNE version you want first, and pip keeps it.

For the segmentation step, install [SimNIBS ≥ 4.0](https://simnibs.github.io/simnibs/build/html/installation/simnibs_installer.html)
(separate installer; `charm` must be on your `PATH`). Not needed if you
already have a segmentation.

## Tutorials

| notebook | what | data (downloaded on first run) |
|---|---|---|
| [`01_bem_from_t1`](tutorials/01_bem_from_t1.ipynb) | T1 → CHARM → BEM → quality check → viewer of your choice → MNE | OpenNeuro ds000117 T1 (11 MB) |
| [`02_comparison_with_mne`](tutorials/02_comparison_with_mne.ipynb) | mindmap-bem vs MNE's watershed and FLASH BEMs: geometry and EEG forward models | MNE `sample` (1.5 GB) |

Data go to `~/mindmap_bem_data` (or `$MINDMAP_BEM_DATA`).

## Quick start: from a T1

```bash
mindmap-bem run sub01 sub01_T1w.nii.gz --out derivatives/sub01 --subjects-dir derivatives/subjects
```

This runs CHARM (segmentation only, ~15 min) into `derivatives/sub01/m2m_sub01/`
and writes an MNE subject:

```
derivatives/subjects/sub01/
├── mri/T1.mgz                                    # your T1, FreeSurfer format
└── bem/{inner_skull,outer_skull,outer_skin}.surf
```

Rerunning reuses the existing segmentation (`--rerun-charm` forces CHARM again,
`--overwrite` rewrites the surfaces).

Python equivalent:

```python
import mindmap_bem as mb

mb.bem_from_t1("sub01", "sub01_T1w.nii.gz", "derivatives/sub01", subjects_dir="derivatives/subjects")
```

From Python, CHARM's output goes to `derivatives/sub01/charm_sub01.log` rather than
the screen (pass `charm_kwargs={"verbose": True}` to see it); the command line shows it.

Then, in MNE:

```python
model = mne.make_bem_model("sub01", ico=4, conductivity=(0.3, 0.006, 0.3), subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)
```

### From an existing segmentation

```python
t1_mgz = mb.setup_mne_subject(t1, subjects_dir, "sub01")
mb.build_bem(seg, subjects_dir/"sub01"/"bem", fs_t1=t1_mgz)
```

`mb.build_bem_layers(seg_img)` returns the three `pyvista.PolyData` surfaces
(scanner RAS) without writing anything.

## Coordinate frames: with or without FreeSurfer

MNE finds a subject's MRI through `subjects_dir/<subject>/mri/T1.mgz`, and
uses that file's FreeSurfer surface coordinates as the subject's "MRI" frame.
The BEM, the source space and the head→MRI `trans` must all be in that frame.

**Without FreeSurfer.** `setup_mne_subject` (or `--subjects-dir`) creates
`mri/T1.mgz` from your T1, reordering its voxel axes to FreeSurfer's LIA
convention (the image itself is unchanged), and the BEM is written in its
frame. MNE then works as for any FreeSurfer subject: `mne.viz.plot_bem`,
the coregistration GUI (`mne coreg`) to compute `trans`, volume source
spaces, forward models. Cortical (surface) source spaces still need cortical
surfaces from somewhere.

**With FreeSurfer.** Point `--subjects-dir` at your FreeSurfer
`SUBJECTS_DIR`: the existing `T1.mgz` of the reconstruction is used, so the
BEM lines up with FreeSurfer's cortical surfaces. The T1 given to CHARM and the
one FreeSurfer reconstructed must be the same scan.

`fs_t1=` / `--fs-t1` (any `T1.mgz`) and the default of `build_bem` (scanner RAS
of the T1, no `T1.mgz` at all) are available for other pipelines.

## Other segmentations: label configs

Layers are defined from segmentation label IDs in a small YAML file. Each
layer lists tissue categories, or other layers it inherits from:

```yaml
label_map:
  Brain: [2, 3, 41, 42]
  Bone:  [515, 516]
  Scalp: [511]

layers:
  inner_skull: [Brain]
  outer_skull: [inner_skull, Bone]
  skin:        [outer_skull, Scalp]
```

The bundled `charm` preset ([`data/charm.yml`](src/mindmap_bem/data/charm.yml))
maps CHARM's detailed `segmentation/labeling.nii.gz` (not `final_tissues.nii.gz`).
Use `--labels my_labels.yml` with `build` for other segmenters.

## Parameters

The segmentation is reoriented to RAS internally, so any acquisition
orientation and voxel size works. All lengths are in mm.

| parameter | default | meaning |
|---|---|---|
| `z_floors` | inner 45, outer 35, skin 20 | height above the **bottom of the field of view** below which each mask is cut, to cap the surfaces above the neck. Adjust for scans with an unusual inferior extent |
| `closing_radius` | 10 | morphological closing of each mask |
| `n_smooth` | 20 | Laplacian smoothing iterations per step |
| `iterations` | 100 | fitting iterations |
| `max_buffer` | 3 | minimum gap between consecutive layers |
| `step_size` | 0.6 | displacement per iteration |
| `init_radius` | 120 | starting sphere radius around the head's centre of mass |

## Checking and viewing the result

```python
mb.check_bem(layers)          # closed surfaces, each >= 1 mm inside the next
mb.viz.plot_bem(bem_dir, t1, backend="pyvista")   # or "matplotlib", "mne", "freeview"
```

`plot_bem` takes `fs_t1=<the subject's T1.mgz>` for surfaces written in that frame.
Always inspect the result before computing a forward model.

## Tests

```bash
pip install -e ".[test]"
pytest
```

The CHARM wrapper is tested against a stand-in executable, so SimNIBS is not
needed to run the tests.

## License

MIT. SimNIBS/CHARM is a separate program under its own (GPL-3) license and is
not distributed with this package.
