import os
import stat
import sys

import numpy as np
import nibabel as nib
import pytest

import mindmap_bem as mb
from mindmap_bem.masks import close_mask

# synthetic head: concentric spheres around the world origin
RADII = {"brain": 28, "bone": 34, "scalp": 40}
IDS = {"brain": 1, "bone": 2, "scalp": 3}
LABELS = {
    "label_map": {"Brain": [IDS["brain"]], "Bone": [IDS["bone"]], "Scalp": [IDS["scalp"]]},
    "layers": {
        "inner_skull": ["Brain"],
        "outer_skull": ["inner_skull", "Bone"],
        "skin": ["outer_skull", "Scalp"],
    },
}
EXPECTED = {"inner_skull": RADII["brain"], "outer_skull": RADII["bone"], "skin": RADII["scalp"]}
NO_FLOORS = {"inner_skull": None, "outer_skull": None, "skin": None}
FIT = dict(z_floors=NO_FLOORS, closing_radius=3)


def make_head(zooms=(1.0, 1.0, 1.0), fov=100.0) -> nib.Nifti1Image:
    """RAS label volume of concentric spheres centred on world (0, 0, 0)."""
    zooms = np.asarray(zooms)
    shape = tuple(int(round(fov / z)) for z in zooms)
    affine = np.diag([*zooms, 1.0])
    affine[:3, 3] = -np.array(shape) * zooms / 2

    ijk = np.indices(shape).reshape(3, -1).T
    r = np.linalg.norm(nib.affines.apply_affine(affine, ijk), axis=1).reshape(shape)

    vol = np.zeros(shape, dtype=np.int16)
    for tissue in ("scalp", "bone", "brain"):
        vol[r <= RADII[tissue]] = IDS[tissue]

    return nib.Nifti1Image(vol, affine)


def radii(layers):
    return {k: np.linalg.norm(v.points, axis=1) for k, v in layers.items()}


@pytest.fixture(scope="module")
def seg():
    return make_head()


@pytest.fixture(scope="module")
def layers(seg):
    return mb.build_bem_layers(seg, LABELS, **FIT)


# --- labels ---------------------------------------------------------------

def test_labels_inheritance():
    labels = mb.read_bem_labels(LABELS)
    assert labels == {"inner_skull": [1], "outer_skull": [1, 2], "skin": [1, 2, 3]}


def test_charm_preset():
    assert "charm" in mb.list_presets()
    labels = mb.read_bem_labels("charm")
    assert set(labels["inner_skull"]) < set(labels["outer_skull"]) < set(labels["skin"])


def test_presets_use_every_category():
    import yaml
    from importlib import resources
    for preset in mb.list_presets():
        cfg = yaml.safe_load(resources.files("mindmap_bem").joinpath("data", f"{preset}.yml").read_text())
        used = {item for items in cfg["layers"].values() for item in items}
        assert set(cfg["label_map"]) <= used, f"{preset}: unused {set(cfg['label_map']) - used}"


def test_circular_inheritance():
    bad = {"label_map": {}, "layers": {"inner_skull": ["skin"], "outer_skull": [], "skin": ["inner_skull"]}}
    with pytest.raises(ValueError, match="Circular"):
        mb.read_bem_labels(bad)


# --- masks ----------------------------------------------------------------

def test_close_mask_fills_gap():
    mask = np.zeros((30, 30, 30), dtype=bool)
    mask[5:25, 5:25, 5:14] = True
    mask[5:25, 5:25, 16:25] = True        # 2-voxel gap
    closed = close_mask(mask, radius=3)
    assert closed[15, 15, 15]
    assert closed[mask].all()             # closing is extensive


def test_close_mask_radius_in_mm():
    # same 4 mm gap: bridged by a 3 mm ball at 1 mm voxels, and still at 2 mm voxels
    for z in (1.0, 2.0):
        n = int(40 / z)
        mask = np.zeros((n, n, n), dtype=bool)
        gap = int(4 / z)
        mask[:, :, n//2 - 10//int(z) : n//2] = True
        mask[:, :, n//2 + gap : n//2 + gap + 10//int(z)] = True
        closed = close_mask(mask, radius=3, zooms=(z, z, z))
        assert closed[n//2, n//2, n//2 + gap//2]


# --- fitting --------------------------------------------------------------

@pytest.mark.parametrize("layer", ["inner_skull", "outer_skull", "skin"])
def test_layer_radius(layers, layer):
    assert abs(np.median(radii(layers)[layer]) - EXPECTED[layer]) < 2.0


def test_layers_nested(layers):
    r = radii(layers)
    assert r["outer_skull"].min() > r["inner_skull"].max()
    assert r["skin"].min() > r["outer_skull"].max()


def test_check_bem(layers):
    assert mb.check_bem(layers, verbose=False)
    gaps = mb.layer_gaps(layers)
    expected = RADII["bone"] - RADII["brain"]
    assert abs(gaps["inner_skull -> outer_skull"]["median"] - expected) < 2.0

    # swap two layers: the check must fail
    swapped = {**layers, "inner_skull": layers["outer_skull"], "outer_skull": layers["inner_skull"]}
    assert not mb.check_bem(swapped, verbose=False)
    assert mb.layer_gaps(swapped)["inner_skull -> outer_skull"]["n_outside"] > 0


def test_plot_slices(seg, layers):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    fig = mb.viz.plot_bem(layers, seg, backend="matplotlib", n_slices=3)
    lines = sum(len(ax.lines) for ax in fig.axes)
    assert lines >= 3 * 3 * 3          # 3 views x 3 slices x 3 layers


def test_fetch_example_t1_cached(tmp_path):
    # no network: a cached file is returned as is
    cached = tmp_path/"ds000117"/"sub-03"/"anat"/"sub-03_T1w.nii.gz"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"x")
    assert mb.datasets.fetch_example_t1("3", path=tmp_path) == cached
    with pytest.raises(ValueError):
        mb.datasets.fetch_example_t1("42", path=tmp_path)


def test_orientation_invariance(seg, layers):
    # same head stored LIA (FreeSurfer-like): world-space result must not change
    ornt = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(seg.affine), nib.orientations.axcodes2ornt("LIA"),
    )
    lia = seg.as_reoriented(ornt)
    assert nib.aff2axcodes(lia.affine) == ("L", "I", "A")

    other = mb.build_bem_layers(lia, LABELS, **FIT)
    for k in layers:
        np.testing.assert_allclose(other[k].points, layers[k].points, atol=1e-6)


def test_anisotropic_voxels():
    layers = mb.build_bem_layers(make_head(zooms=(0.8, 1.0, 1.25)), LABELS, **FIT)
    r = radii(layers)
    for layer, expected in EXPECTED.items():
        assert abs(np.median(r[layer]) - expected) < 2.0
    assert r["outer_skull"].min() > r["inner_skull"].max()


def test_z_floor_in_mm():
    # floor 60 mm above the bottom of a 100 mm FOV centred on 0 -> cap at z = +10 mm
    # (each layer must still enclose the previous one, hence a few mm of slack)
    seg = make_head(zooms=(1.0, 1.0, 1.25))
    floors = dict.fromkeys(NO_FLOORS, 60.0)
    layers = mb.build_bem_layers(seg, LABELS, z_floors=floors, closing_radius=3)
    for layer, mesh in layers.items():
        assert 0 < mesh.points[:, 2].min() < 15, layer
        assert mesh.points[:, 2].max() > EXPECTED[layer] - 2.0


def test_brain_mask_grid_check(seg):
    other = nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.uint8), np.eye(4))
    with pytest.raises(ValueError, match="same voxel grid"):
        mb.build_bem_layers(seg, LABELS, brain_mask=other, z_floors=NO_FLOORS)


# --- output ---------------------------------------------------------------

def test_build_bem_scanner_ras(seg, tmp_path):
    layers = mb.build_bem(seg, tmp_path/"bem", labels=LABELS, **FIT)
    back = mb.read_bem_surfaces(tmp_path/"bem")
    np.testing.assert_allclose(back["skin"].points, layers["skin"].points, atol=1e-3)

    with pytest.raises(FileExistsError):
        mb.build_bem(seg, tmp_path/"bem", labels=LABELS, **FIT)


def test_build_bem_freesurfer_frame(seg, layers, tmp_path):
    # conformed-like FreeSurfer T1: tkRAS = scanner RAS shifted by c_ras
    affine = np.array([[-1, 0, 0, 133.0], [0, 0, 1, -110.0], [0, -1, 0, 140.0], [0, 0, 0, 1]])
    fs_t1 = tmp_path/"T1.mgz"
    nib.save(nib.MGHImage(np.zeros((256, 256, 256), dtype=np.uint8), affine), fs_t1)

    out = mb.build_bem(seg, tmp_path/"bem", labels=LABELS, fs_t1=fs_t1, **FIT)

    c_ras = affine[:3, :3] @ np.array([128, 128, 128]) + affine[:3, 3]
    np.testing.assert_allclose(out["skin"].points, layers["skin"].points - c_ras, atol=1e-4)

    _, _, meta = nib.freesurfer.io.read_geometry(tmp_path/"bem"/"outer_skin.surf", read_metadata=True)
    np.testing.assert_allclose(meta["cras"], c_ras, atol=1e-4)


def test_volume_info_matches_mgh_header(tmp_path):
    # oblique, anisotropic, non-conformed volume (regression: scaled cosines / wrong c_ras)
    rot = nib.eulerangles.euler2mat(0.1, -0.2, 0.05)
    affine = np.eye(4)
    affine[:3, :3] = rot @ np.diag([0.86, 1.0, 0.86])
    affine[:3, 3] = [-90.0, 120.0, -100.0]
    fname = tmp_path/"T1.mgz"
    nib.save(nib.MGHImage(np.zeros((200, 230, 210), dtype=np.uint8), affine), fname)

    info = mb.make_volume_info(fname)
    header = nib.load(fname).header
    np.testing.assert_allclose(np.stack([info["xras"], info["yras"], info["zras"]]), header["Mdc"], atol=1e-5)
    np.testing.assert_allclose(info["cras"], header["Pxyz_c"], atol=1e-4)


def test_mne_subject_without_freesurfer(seg, tmp_path):
    mne = pytest.importorskip("mne")
    # any T1 -> MNE subject (mri/T1.mgz) -> BEM in its frame -> MNE volume source space
    t1 = nib.Nifti1Image((np.asanyarray(seg.dataobj) > 0).astype(np.float32) * 100, seg.affine)
    t1_mgz = mb.setup_mne_subject(t1, tmp_path, "synthetic")
    assert t1_mgz == tmp_path/"synthetic"/"mri"/"T1.mgz"
    assert mb.setup_mne_subject("ignored-when-present", tmp_path, "synthetic") == t1_mgz
    assert nib.aff2axcodes(nib.load(t1_mgz).affine) == ("L", "I", "A")
    np.testing.assert_allclose(mb.ras_to_tkras(t1_mgz)[:3, :3], np.eye(3), atol=1e-6)   # pure translation

    mb.build_bem(seg, tmp_path/"synthetic"/"bem", labels=LABELS, fs_t1=t1_mgz, **FIT)
    model = mne.make_bem_model("synthetic", ico=4, conductivity=[0.3, 0.006, 0.3], subjects_dir=tmp_path, verbose="error")
    bem = mne.make_bem_solution(model, verbose="error")
    src = mne.setup_volume_source_space("synthetic", pos=8.0, bem=bem, subjects_dir=tmp_path, verbose="error")
    # sources fill the inner skull sphere (r = 28 mm): count ~ volume / 8^3
    assert 0.5 < src[0]["nuse"] / (4/3*np.pi*28**3 / 8**3) < 1.5


def test_mne_accepts_surfaces(layers, tmp_path):
    mne = pytest.importorskip("mne")
    subjects_dir, subject = tmp_path, "synthetic"
    mb.write_bem_surfaces(subjects_dir/subject/"bem", layers)

    model = mne.make_bem_model(subject, ico=None, conductivity=[0.3, 0.006, 0.3], subjects_dir=subjects_dir)
    assert len(model) == 3


def test_vendored_ico_matches_mne():
    mne = pytest.importorskip("mne")
    from mne.bem import _get_ico_surface
    from mindmap_bem.fitting import ico_sphere

    ico = _get_ico_surface(5)
    sphere = ico_sphere()
    np.testing.assert_allclose(sphere.points, ico["rr"], atol=1e-6)
    np.testing.assert_array_equal(sphere.faces.reshape(-1, 4)[:, 1:], ico["tris"])


# --- CHARM ----------------------------------------------------------------

@pytest.fixture
def fake_charm(tmp_path, seg):
    """
    Executable standing in for SimNIBS' charm: logs its argv, prints progress and
    writes a labeling. Subject "fail" makes it exit with an error instead.
    """
    label_src = tmp_path/"labels_src.nii.gz"
    nib.save(seg, label_src)
    log = tmp_path/"charm_argv.txt"

    script = tmp_path/"charm"
    script.write_text(
        f"#!{sys.executable}\n"
        "import sys, shutil, pathlib\n"
        f"open({str(log)!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
        "print('charm progress')\n"
        "if sys.argv[1] == 'fail':\n"
        "    print('charm error', file=sys.stderr)\n"
        "    sys.exit(3)\n"
        "seg = pathlib.Path('m2m_' + sys.argv[1])/'segmentation'\n"
        "seg.mkdir(parents=True, exist_ok=True)\n"
        f"shutil.copy({str(label_src)!r}, seg/'labeling.nii.gz')\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script), log


def test_run_charm_command(fake_charm, tmp_path):
    cmd, log = fake_charm
    t1, t2 = tmp_path/"t1.nii.gz", tmp_path/"t2.nii.gz"

    m2m = mb.run_charm("sub01", t1, tmp_path/"out", t2=t2, charm_cmd=cmd)
    assert mb.charm_labeling(m2m).exists()
    assert log.read_text().split() == ["sub01", str(t1), str(t2), "--registerT2", "--initatlas", "--segment"]

    # existing segmentation is reused, overwrite re-runs with --forcerun
    mb.run_charm("sub01", t1, tmp_path/"out", charm_cmd=cmd)
    assert len(log.read_text().splitlines()) == 1
    mb.run_charm("sub01", t1, tmp_path/"out", charm_cmd=cmd, overwrite=True)
    assert log.read_text().splitlines()[-1].endswith("--forcerun")


def test_run_charm_forcesform_without_qform(fake_charm, tmp_path, seg):
    cmd, log = fake_charm
    t1 = tmp_path/"no_qform.nii.gz"
    img = nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.int16), np.eye(4))   # sform only
    assert int(img.header["qform_code"]) == 0
    nib.save(img, t1)

    mb.run_charm("sub02", t1, tmp_path/"out", charm_cmd=cmd)
    assert log.read_text().split()[-1] == "--forcesform"


def test_run_charm_resumes_incomplete_run(fake_charm, tmp_path):
    cmd, log = fake_charm
    (tmp_path/"out"/"m2m_sub03").mkdir(parents=True)        # crashed run, no labeling
    mb.run_charm("sub03", tmp_path/"t1.nii.gz", tmp_path/"out", charm_cmd=cmd)
    assert log.read_text().split()[-1] == "--forcerun"


def test_run_charm_output(fake_charm, tmp_path, capfd):
    cmd, _ = fake_charm
    t1 = tmp_path/"t1.nii.gz"

    # quiet by default: CHARM's output goes to a log file
    mb.run_charm("sub04", t1, tmp_path/"out", charm_cmd=cmd)
    assert "charm progress" not in capfd.readouterr().out
    assert "charm progress" in (tmp_path/"out"/"charm_sub04.log").read_text()

    mb.run_charm("sub05", t1, tmp_path/"out", charm_cmd=cmd, verbose=True)
    assert "charm progress" in capfd.readouterr().out
    assert not (tmp_path/"out"/"charm_sub05.log").exists()


def test_run_charm_failure_shows_log(fake_charm, tmp_path):
    cmd, _ = fake_charm
    with pytest.raises(RuntimeError, match="(?s)exit code 3.*charm error"):
        mb.run_charm("fail", tmp_path/"t1.nii.gz", tmp_path/"out", charm_cmd=cmd)


def test_run_charm_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="SimNIBS"):
        mb.run_charm("sub01", tmp_path/"t1.nii.gz", tmp_path, charm_cmd="definitely-not-charm")


def test_bem_from_t1(fake_charm, tmp_path, monkeypatch):
    cmd, _ = fake_charm
    # the CHARM preset's labels don't exist in the synthetic head: map them
    monkeypatch.setattr("mindmap_bem.charm.build_bem",
                        lambda seg, out, labels, **kw: mb.build_bem(seg, out, labels=LABELS, **kw))

    layers = mb.bem_from_t1("sub01", tmp_path/"t1.nii.gz", tmp_path/"out", charm_kwargs={"charm_cmd": cmd}, **FIT)
    assert (tmp_path/"out"/"bem"/"outer_skin.surf").exists()
    assert set(layers) == {"inner_skull", "outer_skull", "skin"}
