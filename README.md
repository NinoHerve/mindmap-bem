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
- the layers are **nested during the fit**, with no overlapping skull
  surfaces to fix afterwards (a common failure of the watershed approach),
  and `build_bem` checks the result before writing: it refuses to write
  folded or intersecting surfaces (`check=False` / `--no-check` to override),
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
| [`01_bem_from_t1`](https://github.com/NinoHerve/mindmap-bem/blob/main/tutorials/01_bem_from_t1.ipynb) | T1 → CHARM → BEM → quality check → viewer of your choice → MNE | OpenNeuro ds000117 T1 (11 MB) |
| [`02_comparison_with_mne`](https://github.com/NinoHerve/mindmap-bem/blob/main/tutorials/02_comparison_with_mne.ipynb) | mindmap-bem vs MNE's watershed and FLASH BEMs: geometry and EEG forward models | MNE `sample` (1.5 GB) |

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

The bundled `charm` preset ([`data/charm.yml`](https://github.com/NinoHerve/mindmap-bem/blob/main/src/mindmap_bem/data/charm.yml))
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
| `inner_gap` | 2 | minimum gap between consecutive layers: each layer stays this far outside both the previous layer's mask and its fitted surface. Sets the minimum skull thickness |
| `step_size` | 0.6 | displacement per iteration |
| `init_radius` | from the mask | starting sphere radius around the mask's centre of mass; by default just large enough to enclose the mask (+ 5 mm) |

## Checking and viewing the result

```python
mb.check_bem(layers)          # closed surfaces without folds, each >= 1 mm inside the next
mb.viz.plot_bem(bem_dir, t1, backend="pyvista")   # or "matplotlib", "mne", "freeview"
```

`plot_bem` takes `fs_t1=<the subject's T1.mgz>` for surfaces written in that frame.
Always inspect the result before computing a forward model.

To compare the forward models of two BEMs, `mb.common_leadfields` restricts MNE
fixed-orientation forward solutions to their common sources (average referenced),
and `mb.rdm` / `mb.lnmag` compute the topography and magnitude differences per
source. Tutorial 2 shows them in use.

## Building your own surfaces

The building blocks of `build_bem_layers` are public, to fit surfaces to masks of
your own (e.g. one skin around several BEMs):

| function | does |
|---|---|
| `to_ras(img)` | reorder a volume's axes to RAS (needed before `z_floor`) |
| `close_mask(mask, radius, zooms)` | morphological closing with a ball of `radius` mm |
| `surface_to_mask(surface, img)` | fill a closed surface (scanner RAS) on an image's voxel grid |
| `signed_distance_interpolator(mask, zooms)` | smoothed signed distance field of a mask, to constrain a fit (`inner_interpolator=` / `outer_interpolator=`) |
| `fit_surface_to_mask(mask_img, ...)` | fit an ico-5 sphere to a mask, in scanner RAS |
| `grid_to_world(img)` | transform from "grid mm" (voxel index × voxel size, the fit's frame) to scanner RAS |
| `polydata_to_faces(mesh)` | `(n_faces, 3)` triangle array of a surface |

```python
mask = mb.surface_to_mask(layers["outer_skull"], img) | (scalp > 0)
skin, _ = mb.fit_surface_to_mask(nib.Nifti1Image(mb.close_mask(mask, 10, zooms).astype("uint8"), img.affine))
```

## Known issues

**Sharp concave corners, such as the skull base, can hold the fit back.** There,
moving along the normals would fold a few triangles of the surface onto
itself. The fit prevents this: after each iteration it undoes the moves that
folded the surface (the fold guard), so the surfaces it returns never fold.
The count of undone moves is in `layer.field_data["n_reverted"]`. It is 0 for
most heads; on ds000117, 2 of 16 subjects needed it, with no other effect than a
few thinner triangles at the skull base.

A region the guard keeps holding back can stay short of its target, and in the
worst case cross the next layer. `build_bem` checks every BEM before writing
it and raises an error instead of writing intersecting surfaces (`check=False`
/ `--no-check` writes them anyway, for inspection). If that happens, try other
`z_floors` first: the rim of the cut is the usual corner.

Background on these and other design choices: [`docs/decisions.md`](https://github.com/NinoHerve/mindmap-bem/blob/main/docs/decisions.md).

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
