# Design decisions

Why mindmap-bem works the way it does: one entry per decision, newest first.
Each says what the problem was, what was chosen, what was tried and rejected,
and the evidence. Add an entry whenever a choice would otherwise have to be
rediscovered.

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
to write the result.

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
