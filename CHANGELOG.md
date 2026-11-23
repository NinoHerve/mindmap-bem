# Changelog

## 0.1.0 (2026-11-23)

First release.

- Nested BEM surfaces (inner skull, outer skull, skin) from a T1 via SimNIBS CHARM, or
  from any segmentation with a YAML label config, written for `mne.make_bem_model`.
- The fit can't fold and gives the same result with every VTK version; layers are kept
  at least 2 mm apart (see `docs/decisions.md`).
- `build_bem` refuses to write surfaces that fail `check_bem` (`--no-check` to override).
- Forward-model comparison metrics: `common_leadfields`, `rdm`, `lnmag`.
- `run_charm` stops CHARM on Ctrl+C or `stop_charm()`, and logs to a file of your choice.
