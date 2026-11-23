# Design decisions

Why mindmap-bem works the way it does: one entry per decision, newest first.
Each says what the problem was, what was chosen, what was tried and rejected,
and the evidence. Add an entry whenever a choice would otherwise have to be
rediscovered.

---

## 2026-11-23: minimum gap between layers (`inner_gap`) lowered from 3 to 2 mm

**Problem.** Each layer is kept at least `inner_gap` mm outside the previous
one, so `inner_gap` is the minimum skull thickness of the model. Adult skull is
typically about 3 to 8 mm thick (Dannhauer et al., 2011), and thinner in
places (e.g. the temporal squama). With 3 mm, the outer skull was pushed
outward wherever CHARM's skull is thinner than that, making the skull
artificially thick there. A gap is needed at all for numerical reasons: MNE
rejects intersecting surfaces, and its BEM loses accuracy as surfaces get
closer.

**Decision.** `inner_gap = 2.0` by default. Precedent: two EEG head-modelling
studies impose the same 2 mm minimum skull thickness on their models.
- Dannhauer et al. (2011): "The minimum skull thickness was assured to be 2 mm".
- Lanfer et al. (2012) eroded the skull mask by 2 mm "to guarantee a minimum
  thickness for the inner and outer compact bone layer".

Neither paper measures the anatomical minimum. They chose 2 mm as a modelling
floor. Dannhauer et al. also note that at 1 mm voxels, partial-volume effects
matter in thin skull sections, so a 1 mm skull is not resolved reliably
anyway.

**Evidence.** The 16 ds000117 subjects + MNE `sample` were refitted with
`inner_gap` from 1 to 3.5 mm. Medians over the 17 heads:

| `inner_gap` | outer skull at the limit (< gap + 0.5 mm) | outer skull > 1 mm outside CHARM's skull | skull thickness, 5th percentile | closest ico-4 surfaces MNE uses (min) |
|---|---|---|---|---|
| 1.0 | 0.1% | 5.3% | 2.92 mm | 1.41 mm |
| 2.0 | 2.6% | 6.1% | 2.95 mm | 1.99 mm |
| 3.0 | 8.7% | 8.3% | 3.19 mm | 2.68 mm |
| 3.5 | 14.9% | 10.1% | 3.45 mm | 2.99 mm |

- **Validity.** Every gap passes `check_bem`, and MNE's `make_bem_model` (ico-4)
  and `make_bem_solution` succeed: 102 of 102.
- **What moves.** Only the outer skull changes, and only locally. Compared with
  3 mm, a 2 mm gap moves it 0.48 mm at the 95th percentile and 2.1 mm at most.
  The inner skull is unchanged, and the skin moves < 0.6 mm.
- **How much thickening is left.** About 5% of the outer skull lies > 1 mm
  outside CHARM's skull even at 1 mm, from the fit's smoothing. At 2 mm the gap
  adds less than 1 percentage point on top of that; at 3 mm it added 3.
- **Numerical accuracy.** For three concentric spheres the EEG has an exact
  solution (`mne.make_sphere_model`). On such spheres (inner skull 80 mm, skin
  92 mm, conductivities 0.3/0.006/0.3 S/m, 73 electrodes), MNE's ico-4 BEM
  error vs the exact solution grows slowly as the skull thins. Median RDM:
  0.010 (7 mm), 0.012 (3 mm), 0.014 (2 mm), 0.017 (1 mm). For sources within
  15 mm of the skull: 0.017 (3 mm), 0.022 (2 mm), 0.031 (1 mm).
- **Tutorial 2.** The `sample` EEG forward model with 2 mm vs 3 mm: RDM 0.005
  (95%: 0.028), lnMAG +0.002. Against FLASH: RDM 0.057 instead of 0.054,
  |lnMAG| 95% 0.173 instead of 0.144. This moves slightly away from FLASH, but
  FLASH can't arbitrate thin skull (next point).
- **FLASH appears to have its own floor of about 3.5 mm.** This is an
  observation on one subject (MNE `sample`), not a documented rule. MNE's
  `make_flash_bem` builds the surfaces with FreeSurfer's
  `mri_make_bem_surfaces`, a closed binary that deforms an ico-4 template and
  documents no thickness option. But the skull thickness of `sample`'s FLASH
  BEM (outer skull vertices to the inner skull surface) has a hard lower edge:

  | thickness (mm) | 3.0–3.5 | 3.5–4.0 | 4.0–4.5 | 4.5–5.0 | 5.0–5.5 |
  |---|---|---|---|---|---|
  | vertices | 20 | 304 | 343 | 294 | 302 |

  The minimum is 3.43 mm and the 1st percentile 3.51 mm. A natural thickness
  distribution would taper off. A pile-up just above a cut-off is what an
  enforced minimum produces: our own fit showed it at 3 mm, with 8.7% of the
  outer skull at the limit. The FLASH skin-to-skull gap tapers off normally
  (1, 12, 60 vertices in the same bins). If FLASH does enforce ~3.5 mm, its
  skull is too thick wherever real bone is thinner. Part of our larger
  difference from FLASH at 2 mm would then come from FLASH's floor, not from
  our fit. To confirm: other FLASH subjects, or `mri_make_bem_surfaces`'
  source.

**Rejected:**
- **1 to 1.5 mm.** The numerical error grows faster below 2 mm, especially for
  shallow sources, while the surfaces barely change any more. The bone also
  can't be resolved at 1 mm voxels.
- **Keeping 3 mm.** It thickens about 9% of the outer skull to the limit,
  against 3% at 2 mm.

**References**
- Dannhauer M, Lanfer B, Wolters CH, Knösche TR (2011). Modeling of the human
  skull in EEG source analysis. *Human Brain Mapping* 32:1383–1399.
  https://doi.org/10.1002/hbm.21114
- Lanfer B, Scherg M, Dannhauer M, Knösche TR, Burger M, Wolters CH (2012).
  Influences of skull segmentation inaccuracies on EEG source analysis.
  *NeuroImage* 62:418–431. https://doi.org/10.1016/j.neuroimage.2012.05.006

---

## 2026-11-23: start sphere sized from the mask; layers kept outside the previous fitted surface

**Problem.** Two causes made layers fail on the synthetic head of
`test_z_floor_in_mm`, where all three masks are cut at the same height:

1. **The start sphere had a fixed 120 mm radius.** Around a 28 mm target, the
   lower half of the sphere had to flatten onto a small disk. Its vertices
   crowded at the rim of the cut, and the fold guard held them back there
   (163,000 reverted moves). The inner skull stopped at z = −8.5 mm instead of
   +10.
2. **Each layer was kept 3 mm outside the previous layer's mask, not its fitted
   surface.** With equal floors, the outer skull's bottom sits 3 mm below its
   mask, and the skin's bottom ended up at the same height.

**Decision:**

1. **Start radius.** It is now computed from the mask: the distance from the
   mask's centre of mass to its farthest voxel corner, plus 5 mm
   (`fitting.start_radius`). `init_radius` still overrides it.
2. **Constraint field.** Each layer is kept outside the *union* of the previous
   layer's mask and its fitted surface. The surface is turned back into a field
   (`surface_to_mask` → `signed_distance_interpolator`), and the two fields are
   combined with a `max`, the signed distance of a union of volumes. Where the
   previous surface went past its mask, the surface counts; where it stopped
   short, the mask does. The fit itself is unchanged: it still gets one
   `inner_interpolator`.

**Evidence:**

- **Synthetic head.** The start-radius experiment showed the compression is the
  cause: inner-skull reverts drop as the radius shrinks, 163,122 (120 mm),
  37,175 (80), 11,715 (60), 0 (45). With both changes, `test_z_floor_in_mm`
  passes, including `check_bem`.
- **ds000117 + `sample`.** All pass in both VTK versions, and the two versions
  agree to 4e-6 mm. Compared with the fit before this change, the median
  distance is 0.002 mm and the maximum 2.2 mm.
- **Tutorial 2 RDM/lnMAG vs FLASH.** Unchanged (0.054 / +0.029).
- **Cost.** About 10 s more per BEM, for one voxelization and two distance
  transforms per layer.

**Rejected.** Exact vertex-to-mesh distances at every iteration
(`compute_implicit_distance`). They are more precise, but they bring a VTK call
back into the loop and don't fit the field design.

**Open: is 3 mm the right `inner_gap`?** (Resolved: lowered to 2 mm, see the
entry above.) The skull can be thinner than 3 mm (the
temporal squama is often 1–2 mm). There the outer skull is pushed out and the
skull made too thick. On ds000117, 8.7% of the outer-skull vertices are within
3.5 mm of the inner skull, 2.8% within 3 mm, none within 2.5 mm. The skin never
comes near its limit (5th percentile 5.1 mm). The gap exists for numerical
reasons: MNE downsamples to ico-4 (~5–7 mm triangles), and BEM integrals lose
accuracy when surfaces are closer than about a triangle size. Lowering it needs
that accuracy checked first.

---

## 2026-11-23: the fit's normals and smoothing are computed in numpy, not VTK

**Problem.** The same segmentation gave different BEMs depending on the VTK
version installed with pyvista. With VTK 9.3.1, the inner skull of one of the 16
ds000117 subjects (sub-06) and the synthetic head of `test_z_floor_in_mm` folded
at the skull base. With VTK 9.4.2 and 9.7.2 they did not. pyvista's smoothing
defaults are the same in all three, so the difference is in VTK itself.

The fit called VTK three times per iteration: `compute_normals`, `smooth`
(Laplacian) and, every 10 iterations, `smooth_taubin` (windowed sinc). Timing
the same input in each environment showed that `smooth_taubin(n_iter=5,
pass_band=0.1)` moves vertices much more with VTK 9.3.1:

| median vertex displacement | VTK 9.3.1 | VTK 9.4.2 / 9.7.2 |
|---|---|---|
| sphere with radial noise (r = 80 mm, 10% noise) | 2.30 mm | 0.44 mm |
| fitted inner skull (sub-01) | 0.009 mm | 0.003 mm |

`compute_normals` and `smooth` gave the same results in all versions.

**Decision.** `mesh.py` computes the normals, the Laplacian smoothing and a
classic λ/μ Taubin smoothing in numpy:

- **Normals:** the mean of the unit normals of the triangles around each vertex,
  as VTK does.
- **Laplacian:** the same update as `vtkSmoothPolyDataFilter`. It matches VTK to
  0.001 mm (median) on a fitted surface.
- **Taubin:** λ = 0.33, μ = −0.34 (pass band ≈ 0.1). This moves a fitted surface
  as much as VTK ≥ 9.4's `smooth_taubin`, which the manuscript BEMs were built
  with. It is not the same filter as VTK's windowed sinc, only calibrated to
  match it.

The fit is now the same in every environment: the 16 subjects fitted with VTK
9.3.1 and 9.7.2 differ by 0.0000 mm. VTK is still used outside the fit loop:
output normals, I/O and plotting.

**Rejected.** Pinning `vtk >= 9.4`. It doesn't fix the cause, it ties users to
specific VTK versions, and the next VTK change could move the results again.

---

## 2026-11-23: a fold guard in the fit loop; vertices keep moving along normals

**Problem.** Vertices move along their mesh normals. When a triangle folds, its
normal flips and the neighbouring vertices are pushed the wrong way, so the fold
grows. A folded surface is still closed, so MNE accepts it without a warning.
The guard in `build_bem` (refusing to write folded BEMs) protected users but
gave them nothing to do about it.

**Decision.** After each iteration, `fitting._revert_folds` looks for folded
triangles: those whose normal differs by more than 120° from a neighbour's.

- If it finds any, it puts their vertices back where they were before the
  iteration.
- If the surface is still folded, it grows the reverted region by one ring of
  neighbours, up to 3 times, then reverts the whole iteration.
- The start sphere is fold-free, so the fitted surface is too.
- The number of reverted moves is in `field_data["n_reverted"]`.
- The check costs about 1.2 ms per iteration, because the edge/face pairs are
  computed once per fit.

Results on ds000117 (16 subjects) and MNE `sample`, compared with the old fit:

- **`check_bem`:** all pass, in both VTK versions.
- **Guard use:** only sub-06 and sub-13 needed it (2265 and 1228 reverted moves).
- **Surface change:** median distance to the old surfaces 0.02 mm. Differences
  above 1 mm are limited to a few hundred skull-base vertices, where the old fit
  already differed between VTK versions by up to 12 mm.
- **EEG forward model** (`sample`): RDM vs old 0.003, against 0.054 between the
  old fit and FLASH.
- **Triangle quality:** the smallest triangle angle dropped to 5.9° (sub-06) and
  9.6° (sub-13) where the guard acted, against 18.7° before.

**Known limitation.** A region the guard keeps reverting stops moving and can
stay short of its target. On the synthetic head with all three z floors at the
same height (`test_z_floor_in_mm`, a strict xfail), the inner skull stops at the
rim of the cut and crosses the outer skull. `build_bem` catches this and refuses
to write the result. (Fixed by sizing the start sphere from the mask: see the
entry above.)

**Rejected:**

- **Moving along the gradient of the distance field instead of the normals.**
  This avoids the flipped-normal feedback in principle, and it fitted the
  synthetic heads well. On real heads it folds at the rim of the z-floor cut:
  the gradient points sideways toward the rim and pulls the bottom of the sphere
  together. 29 of 33 BEMs failed `check_bem`, with the inner skull up to 19 mm
  through the outer skull.
- **A per-vertex step cap** (at most half the vertex's shortest edge). Edge
  lengths vary across the icosahedron, so neighbouring vertices were capped by
  different amounts. That alone folded a plain sphere fit.
- **Rounding the edge of the z-floor cut** (smooth minimum of the mask and floor
  distances, r = 3 to 8 mm). It still left 24 to 41 folded faces per layer.

**Open.** Each layer is kept outside the previous layer's *mask*, not its fitted
*surface*, so a layer that fails to converge isn't avoided by the next one.
