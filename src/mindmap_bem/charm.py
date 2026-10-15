"""
Module Name: charm.py
Description: Run SimNIBS CHARM head segmentation and build the BEM from it.

CHARM (https://simnibs.github.io/simnibs) segments the whole head (skin,
skull, CSF, brain, ...) from a T1 (optionally + T2). It is part of SimNIBS,
which must be installed separately; it is called here as an external program.

Author: Nino Herve
License: MIT
"""

import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import pyvista as pv

from mindmap_bem.bem import build_bem
from mindmap_bem.io import setup_mne_subject

SIMNIBS_URL = "https://simnibs.github.io/simnibs/build/html/installation/simnibs_installer.html"

# segmentation only: skips CHARM's cortical surfaces and FEM mesh (much faster)
DEFAULT_CHARM_STEPS = ("--initatlas", "--segment")


def charm_labeling(m2m_dir: str | Path) -> Path:
    """Path of CHARM's detailed label volume inside an ``m2m_<subject>`` folder."""
    return Path(m2m_dir)/"segmentation"/"labeling.nii.gz"


def run_charm(
    subject: str,
    t1: str | Path,
    out_dir: str | Path,
    t2: str | Path | None = None,
    steps: tuple[str, ...] = DEFAULT_CHARM_STEPS,
    extra_args: tuple[str, ...] = (),
    overwrite: bool = False,
    charm_cmd: str = "charm",
    verbose: bool = False,
) -> Path:
    """
    Run CHARM, producing ``out_dir/m2m_<subject>``.

    Reuses an existing segmentation unless ``overwrite`` is True.

    Parameters
    ----------
    subject: str
        Subject ID (CHARM names its output ``m2m_<subject>``).

    t1, t2: path
        T1-weighted image (any NIfTI), optional T2-weighted image.

    out_dir: path
        Folder in which ``m2m_<subject>`` is created.

    steps: tuple[str]
        CHARM step flags. Pass ``()`` to run the full pipeline (adds cortical
        surfaces and the FEM head mesh; much slower).

    extra_args: tuple[str]
        Any other CHARM arguments, e.g. ``("--forceqform",)``.

    overwrite: bool
        Re-run (``--forcerun``) even if a segmentation exists.

    charm_cmd: str
        CHARM executable.

    verbose: bool
        Print CHARM's output as it runs. By default it goes to
        ``out_dir/charm_<subject>.log`` instead, which keeps notebooks short.

    Returns
    -------
    m2m_dir: Path
        CHARM output folder.
    """
    out_dir = Path(out_dir).resolve()
    m2m_dir = out_dir/f"m2m_{subject}"

    if charm_labeling(m2m_dir).exists() and not overwrite:
        return m2m_dir

    if shutil.which(charm_cmd) is None:
        raise FileNotFoundError(
            f"'{charm_cmd}' not found. Install SimNIBS (>= 4.0) and make sure its bin "
            f"folder is on PATH: {SIMNIBS_URL}"
        )

    # CHARM refuses NIfTIs without a qform (e.g. written by nibabel from an
    # affine): let it use the sform instead
    if "--forcesform" not in extra_args and any(_missing_qform(f) for f in (t1, t2) if f is not None):
        print("Input has no qform (qform_code 0): running CHARM with --forcesform.")
        extra_args = (*extra_args, "--forcesform")

    cmd = [charm_cmd, subject, str(Path(t1).resolve())]
    if t2 is not None:
        cmd.append(str(Path(t2).resolve()))
        # partial runs must register the T2 explicitly (full runs do it anyway)
        if steps and "--registerT2" not in steps and "--skipregisterT2" not in extra_args:
            steps = ("--registerT2", *steps)
    cmd.extend(steps)
    cmd.extend(extra_args)
    # overwrite, or a previous run died before producing the segmentation
    if overwrite or m2m_dir.exists():
        cmd.append("--forcerun")

    # CHARM writes m2m_<subject> into its working directory
    out_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        result = subprocess.run(cmd, cwd=out_dir)
    else:
        # outside m2m_<subject>, which --forcerun may delete
        log = out_dir/f"charm_{subject}.log"
        print(f"Running CHARM (can take 15 min or more). Output: {log}")
        with open(log, "w") as f:
            result = subprocess.run(cmd, cwd=out_dir, stdout=f, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        msg = f"CHARM failed with exit code {result.returncode}"
        if not verbose:
            tail = "".join(log.read_text(errors="replace").splitlines(keepends=True)[-20:])
            msg += f". Last lines of {log}:\n{tail}"
        raise RuntimeError(msg)

    if not charm_labeling(m2m_dir).exists():
        raise RuntimeError(f"CHARM finished but {charm_labeling(m2m_dir)} is missing; see {m2m_dir}/charm_log.html")

    return m2m_dir


def bem_from_t1(
    subject: str,
    t1: str | Path,
    out_dir: str | Path,
    t2: str | Path | None = None,
    fs_t1: str | Path | None = None,
    bem_dir: str | Path | None = None,
    subjects_dir: str | Path | None = None,
    overwrite: bool = False,
    rerun_charm: bool = False,
    charm_kwargs: dict | None = None,
    **kwargs,
) -> dict[str, pv.PolyData]:
    """
    T1 in, BEM surfaces out: CHARM segmentation followed by ``build_bem``.

    Parameters
    ----------
    subject: str
        Subject ID.

    t1, t2: path
        T1-weighted image, optional T2-weighted image.

    out_dir: path
        Receives ``m2m_<subject>`` (CHARM) and, by default, ``bem/``.

    fs_t1: path | None
        FreeSurfer ``mri/T1.mgz`` to output in its tkRAS. Default: scanner RAS.

    bem_dir: path | None
        Where to write the surfaces. Default: ``out_dir/bem``.

    subjects_dir: path | None
        MNE subjects directory. If given, the BEM goes to
        ``subjects_dir/subject/bem`` in the frame of ``subjects_dir/subject/mri/T1.mgz``,
        which is created from ``t1`` if missing (see ``setup_mne_subject``).
        Recommended for use with MNE; overrides ``fs_t1`` and ``bem_dir``.

    overwrite: bool
        Overwrite existing BEM surfaces.

    rerun_charm: bool
        Re-run CHARM even if a segmentation exists (slow).

    charm_kwargs: dict | None
        Passed to ``run_charm`` (steps, extra_args, charm_cmd, verbose).

    kwargs: dict
        Passed to ``build_bem`` / ``build_bem_layers``.
    """
    m2m_dir = run_charm(subject, t1, out_dir, t2=t2, overwrite=rerun_charm, **(charm_kwargs or {}))
    if subjects_dir is not None:
        fs_t1 = setup_mne_subject(t1, subjects_dir, subject)
        bem_dir = Path(subjects_dir)/subject/"bem"
    bem_dir = Path(out_dir)/"bem" if bem_dir is None else bem_dir

    return build_bem(
        charm_labeling(m2m_dir), bem_dir, labels="charm", fs_t1=fs_t1, overwrite=overwrite, **kwargs,
    )


def _missing_qform(fname: str | Path) -> bool:
    try:
        img = nib.load(fname)
    except (FileNotFoundError, nib.filebasedimages.ImageFileError):
        return False
    return isinstance(img, nib.Nifti1Image) and int(img.header["qform_code"]) == 0 and int(img.header["sform_code"]) > 0
