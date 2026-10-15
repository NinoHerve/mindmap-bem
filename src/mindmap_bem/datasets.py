"""
Module Name: datasets.py
Description: Download example MRI data for the tutorials.

Data are fetched at runtime from public repositories, so the package itself
ships no data. Only the standard library is used.

Author: Nino Herve
License: MIT
"""

import os
import shutil
import urllib.request
from pathlib import Path

# OpenNeuro ds000117 (Wakeman & Henson, multi-subject multi-modal faces), CC0.
# T1: 1 mm isotropic MPRAGE, Siemens Trio 3T. Defaced (face removed).
DS000117_URL = (
    "https://s3.amazonaws.com/openneuro.org/ds000117/"
    "sub-{sub}/ses-mri/anat/sub-{sub}_ses-mri_T1w.nii.gz"
)
DS000117_SUBJECTS = [f"{i:02d}" for i in range(1, 17)]


def data_dir(path: str | Path | None = None) -> Path:
    """
    Folder for downloaded data: ``path``, else ``$MINDMAP_BEM_DATA``, else
    ``~/mindmap_bem_data``.
    """
    path = path or os.environ.get("MINDMAP_BEM_DATA") or Path.home()/"mindmap_bem_data"
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def fetch_example_t1(
    subject: str = "01",
    path: str | Path | None = None,
    verbose: bool = True,
) -> Path:
    """
    Download a T1-weighted MRI from OpenNeuro ds000117 (11 MB), if not cached.

    Parameters
    ----------
    subject: str
        Subject "01" to "16".

    path: path | None
        Download folder (see ``data_dir``).

    Returns
    -------
    t1: Path
        Path to ``ds000117/sub-<subject>/anat/sub-<subject>_T1w.nii.gz``.

    Notes
    -----
    Wakeman, D. G. & Henson, R. N. (2015). A multi-subject, multi-modal human
    neuroimaging dataset. Scientific Data, 2, 150001.
    """
    subject = str(subject).removeprefix("sub-").zfill(2)
    if subject not in DS000117_SUBJECTS:
        raise ValueError(f"ds000117 subjects are {DS000117_SUBJECTS[0]}..{DS000117_SUBJECTS[-1]}")

    fname = data_dir(path)/"ds000117"/f"sub-{subject}"/"anat"/f"sub-{subject}_T1w.nii.gz"
    if fname.exists():
        return fname

    url = DS000117_URL.format(sub=subject)
    if verbose:
        print(f"Downloading {url}")

    _download(url, fname)
    return fname


def _download(url: str, fname: Path):
    """Download to ``fname`` atomically, checking the announced size."""
    fname.parent.mkdir(parents=True, exist_ok=True)
    part = fname.with_name(fname.name + ".part")

    with urllib.request.urlopen(url, timeout=60) as response, open(part, "wb") as f:
        expected = response.headers.get("Content-Length")
        shutil.copyfileobj(response, f)

    if expected is not None and part.stat().st_size != int(expected):
        part.unlink()
        raise OSError(f"Incomplete download of {url}; please retry.")

    part.rename(fname)
