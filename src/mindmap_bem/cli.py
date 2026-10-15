"""
Command line interface.

    mindmap-bem run sub01 T1.nii.gz --out derivatives/sub01      # CHARM + BEM
    mindmap-bem build labeling.nii.gz --out derivatives/sub01/bem
"""

import argparse

from mindmap_bem.bem import build_bem
from mindmap_bem.charm import bem_from_t1
from mindmap_bem.labels import list_presets


def _add_fit_args(parser):
    parser.add_argument("--fs-t1",
                        help="FreeSurfer mri/T1.mgz: write surfaces in that recon's tkRAS "
                             "(default: scanner RAS, no FreeSurfer needed).")
    parser.add_argument("--brain-mask", help="Brain mask on the segmentation grid (keeps inner skull off the brain).")
    parser.add_argument("--n-smooth", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--closing-radius", type=float, default=10.0, help="mm")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing BEM surfaces.")


def _fit_kwargs(args):
    return dict(
        fs_t1=args.fs_t1, brain_mask=args.brain_mask, overwrite=args.overwrite,
        n_smooth=args.n_smooth, iterations=args.iterations, closing_radius=args.closing_radius,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mindmap-bem", description="Nested BEM surfaces from head segmentations.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="T1 -> CHARM segmentation -> BEM surfaces.")
    run.add_argument("subject", help="Subject ID (CHARM output is m2m_<subject>).")
    run.add_argument("t1", help="T1-weighted image.")
    run.add_argument("--t2", help="Optional T2-weighted image.")
    run.add_argument("--out", required=True, help="Output folder (receives m2m_<subject>/ and bem/).")
    run.add_argument("--bem-dir", help="Where to write the surfaces (default: OUT/bem).")
    run.add_argument("--subjects-dir",
                     help="MNE subjects directory: writes SUBJECTS_DIR/<subject>/{mri/T1.mgz,bem/} "
                          "so MNE works without FreeSurfer (recommended).")
    run.add_argument("--rerun-charm", action="store_true", help="Re-run CHARM even if a segmentation exists.")
    run.add_argument("--charm-args", nargs=argparse.REMAINDER, default=[],
                     help="Extra arguments passed verbatim to charm (must come last).")
    _add_fit_args(run)

    build = sub.add_parser("build", help="Fit BEM surfaces to an existing segmentation.")
    build.add_argument("seg", help="Integer label volume (e.g. CHARM m2m_<sub>/segmentation/labeling.nii.gz).")
    build.add_argument("--out", required=True, help="Output folder, e.g. $SUBJECTS_DIR/<subject>/bem.")
    build.add_argument("--labels", default="charm",
                       help=f"Label preset ({', '.join(list_presets())}) or YAML config path.")
    _add_fit_args(build)

    args = parser.parse_args(argv)

    if args.command == "run":
        bem_from_t1(
            args.subject, args.t1, args.out, t2=args.t2, bem_dir=args.bem_dir, rerun_charm=args.rerun_charm,
            subjects_dir=args.subjects_dir,
            charm_kwargs={"extra_args": tuple(args.charm_args), "verbose": True}, **_fit_kwargs(args),
        )
        where = f"{args.subjects_dir}/{args.subject}/bem" if args.subjects_dir else args.bem_dir or f"{args.out}/bem"
        print(f"BEM surfaces written to {where}")

    elif args.command == "build":
        build_bem(args.seg, args.out, labels=args.labels, **_fit_kwargs(args))
        print(f"BEM surfaces written to {args.out}")


if __name__ == "__main__":
    main()
