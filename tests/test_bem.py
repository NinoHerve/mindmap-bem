import os
import stat
import sys

import numpy as np
import nibabel as nib
import pytest

import mindmap_bem as mb
from mindmap_bem.cli import main as cli_main
from mindmap_bem.masks import close_mask, prepare_mask

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


def test_count_folds(layers):
    for mesh in layers.values():
        assert mb.count_folds(mesh) == 0

    # swap the two ends of an edge: both triangles sharing it turn inside out,
    # but the mesh is still a closed manifold
    folded = layers["skin"].copy()
    u, v = folded.faces.reshape(-1, 4)[0, 1:3]
    folded.points[[u, v]] = folded.points[[v, u]]
    assert folded.n_open_edges == 0 and folded.is_manifold
    assert mb.count_folds(folded) >= 2
    assert not mb.check_bem({**layers, "skin": folded}, verbose=False)


def test_fits_carry_the_fold_guard_count(layers):
    for mesh in layers.values():
        assert "n_reverted" in mesh.field_data


def test_fold_guard_reverts_folds():
    from mindmap_bem.fitting import _revert_folds, ico_sphere
    from mindmap_bem.mesh import _adjacency, _edge_face_pairs, _folded

    sphere = ico_sphere()
    faces = mb.polydata_to_faces(sphere)
    previous = np.asarray(sphere.points, dtype=float) * 50
    A, _, _ = _adjacency(faces, len(previous))
    pairs = _edge_face_pairs(faces)

    # a step that moves one vertex sideways past its neighbour folds its triangles
    pts = previous * 1.01
    neighbour = faces[np.any(faces == 0, axis=1)][0]
    neighbour = neighbour[neighbour != 0][0]
    pts[0] = previous[neighbour] + (previous[neighbour] - previous[0])
    assert _folded(pts, faces, pairs).any()

    fixed, n = _revert_folds(pts, previous, faces, pairs, A)
    assert not _folded(fixed, faces, pairs).any()
    assert 0 < n < len(pts)                                   # only the fold was undone
    assert np.array_equal(fixed[0], previous[0])
    far = np.linalg.norm(previous - previous[0], axis=1) > 20
    assert np.allclose(fixed[far], pts[far])                  # the rest keeps its step


def test_mesh_ops_match_vtk(layers):
    import pyvista as pv
    from mindmap_bem.mesh import _adjacency, _laplacian_smooth, _vertex_normals

    mesh = layers["skin"]                  # in scanner RAS, outward winding
    faces, pts = mb.polydata_to_faces(mesh), np.asarray(mesh.points, dtype=float)
    vtk_normals = mesh.compute_normals(consistent_normals=False, auto_orient_normals=False).point_data["Normals"]
    assert np.einsum("ij,ij->i", _vertex_normals(pts, faces), vtk_normals).min() > 0.999

    # Laplacian smoothing: the same update as VTK's, on a mesh that is not smooth yet
    rough = pts + np.random.default_rng(0).normal(scale=0.5, size=pts.shape)
    A, deg, _ = _adjacency(faces, len(pts))
    ours = _laplacian_smooth(rough, A, deg, n_iter=20, relaxation=0.05)
    theirs = np.asarray(pv.PolyData(rough, mesh.faces).smooth(n_iter=20, relaxation_factor=0.05).points)
    moved = np.median(np.linalg.norm(theirs - rough, axis=1))
    assert np.median(np.linalg.norm(ours - theirs, axis=1)) < 0.05 * moved


def test_plot_slices(seg, layers):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    fig = mb.viz.plot_bem(layers, seg, backend="matplotlib", n_slices=3)
    lines = sum(len(ax.lines) for ax in fig.axes)
    assert lines >= 3 * 3 * 3          # 3 views x 3 slices x 3 layers


def test_plots_without_matplotlib(seg, layers, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)          # import matplotlib -> ImportError
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match=r"mindmap-bem\[viz\]"):
        mb.viz.plot_comparison({"a": layers}, seg)
    with pytest.raises(ImportError, match=r"mindmap-bem\[viz\]"):
        mb.viz.plot_slices(layers, seg)


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


def test_prepare_mask_z_floor_in_mm():
    # 1.25 mm slices: a 10 mm floor removes the 8 lowest slices, not 10
    mask = np.ones((4, 4, 40))
    for zooms, n_removed in [((1.0, 1.0, 1.0), 10), ((1.0, 1.0, 1.25), 8), ((1.0, 1.0, 2.5), 4)]:
        cut = prepare_mask(mask, z_floor=10.0, zooms=zooms)
        assert not cut[:, :, :n_removed].any() and cut[:, :, n_removed:].all(), zooms


def test_z_floor_in_mm():
    # floor 60 mm above the bottom of a 100 mm FOV centred on 0 -> cap at z = +10 mm
    # (each layer must still enclose the previous one, hence a few mm of slack)
    seg = make_head(zooms=(1.0, 1.0, 1.25))
    floors = dict.fromkeys(NO_FLOORS, 60.0)
    layers = mb.build_bem_layers(seg, LABELS, z_floors=floors, closing_radius=3)
    for layer, mesh in layers.items():
        assert 0 < mesh.points[:, 2].min() < 15, layer
        assert mesh.points[:, 2].max() > EXPECTED[layer] - 2.0

    # all three masks are cut at the same height: only the surface constraint keeps
    # each layer's bottom outside the previous layer's
    assert mb.check_bem(layers, verbose=False)


def test_start_radius():
    from mindmap_bem.fitting import start_radius

    mask = np.zeros((60, 60, 60), dtype=bool)
    mask[10:30, 20:40, 20:40] = True                        # a box, 20 voxels each side
    zooms = (1.0, 1.0, 2.0)                                 # 20 x 20 x 40 mm
    center = np.array([19.5, 29.5, 59.0])                   # its centre (grid mm)
    half_diagonal = np.linalg.norm([10, 10, 20])
    assert start_radius(mask, zooms, margin=0) == pytest.approx(half_diagonal)
    assert start_radius(mask, zooms, center, margin=5) == pytest.approx(half_diagonal + 5)


def dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum())


def test_surface_to_mask(seg, layers):
    truth = np.asanyarray(seg.dataobj) > 0                       # the head: r <= 40 mm
    mask = mb.surface_to_mask(layers["skin"], seg)
    assert mask.shape == seg.shape and mask.dtype == bool
    assert dice(mask, truth) > 0.97

    # any orientation: the same voxels on a LIA copy of the grid
    ras, lia = nib.orientations.io_orientation(seg.affine), nib.orientations.axcodes2ornt("LIA")
    lia_img = seg.as_reoriented(nib.orientations.ornt_transform(ras, lia))
    back = nib.orientations.apply_orientation(mb.surface_to_mask(layers["skin"], lia_img),
                                              nib.orientations.ornt_transform(lia, ras))
    assert np.array_equal(back, mask)


def test_helpers_are_public():
    for name in ("surface_to_mask", "close_mask", "signed_distance_interpolator", "to_ras",
                 "grid_to_world", "polydata_to_faces"):
        assert name in mb.__all__ and callable(getattr(mb, name))


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


def test_build_bem_refuses_bad_layers(seg, layers, tmp_path, monkeypatch):
    swapped = {**layers, "inner_skull": layers["outer_skull"], "outer_skull": layers["inner_skull"]}
    monkeypatch.setattr("mindmap_bem.bem.build_bem_layers", lambda *a, **kw: swapped)

    with pytest.raises(ValueError, match="check=False"):
        mb.build_bem(seg, tmp_path/"bem", labels=LABELS)
    assert not list((tmp_path/"bem").glob("*.surf"))

    with pytest.warns(UserWarning, match="fail check_bem"):
        mb.build_bem(seg, tmp_path/"bem", labels=LABELS, check=False)
    assert len(list((tmp_path/"bem").glob("*.surf"))) == 3


def test_existing_surfaces_fail_before_fit(fake_charm, tmp_path, monkeypatch):
    cmd, log = fake_charm
    bem_dir = tmp_path/"out"/"bem"
    bem_dir.mkdir(parents=True)
    (bem_dir/"inner_skull.surf").touch()

    def no_fit(*args, **kwargs):
        raise AssertionError("fit started")
    monkeypatch.setattr("mindmap_bem.bem.build_bem_layers", no_fit)

    with pytest.raises(FileExistsError):
        mb.build_bem(tmp_path/"missing.nii.gz", bem_dir)
    with pytest.raises(FileExistsError):
        mb.bem_from_t1("sub01", tmp_path/"t1.nii.gz", tmp_path/"out", charm_kwargs={"charm_cmd": cmd})
    assert not log.exists()                  # CHARM never ran


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


# --- forward model metrics ------------------------------------------------

def test_rdm_lnmag():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(20, 7))
    assert mb.rdm(a, a).shape == mb.lnmag(a, a).shape == (7,)
    np.testing.assert_allclose(mb.rdm(a, a), 0, atol=1e-12)
    np.testing.assert_allclose(mb.lnmag(a, a), 0, atol=1e-12)
    np.testing.assert_allclose(mb.rdm(a, 2*a), 0, atol=1e-12)          # magnitude only
    np.testing.assert_allclose(mb.lnmag(a, 2*a), -np.log(2))
    np.testing.assert_allclose(mb.rdm(a, -a), 2)                      # opposite topographies


def fake_forward(vertno, n_channels=5, per_source=1):
    """Just the parts of an MNE fixed-orientation forward that common_leadfields reads.
    Column of vertex v in space i: (1000 i + v) * channel number, so that it can be
    identified after the average reference."""
    cols = [1000*i + v for i, vs in enumerate(vertno) for v in vs for _ in range(per_source)]
    data = np.outer(np.arange(1.0, n_channels + 1), cols)
    src = [{"vertno": np.asarray(vs), "nuse": len(vs)} for vs in vertno]
    return {"src": src, "sol": {"data": data}}


def test_common_leadfields():
    fwds = {"a": fake_forward([[1, 3, 5, 7], [2, 4, 6]]),
            "b": fake_forward([[1, 5, 7, 9], [2, 4]])}     # missing 3 and 6, extra 9
    G, vertices = mb.common_leadfields(fwds)

    assert [v.tolist() for v in vertices] == [[1, 5, 7], [2, 4]]
    # same sources, in the same order, in both
    expected = np.array([1, 5, 7, 1002, 1004], dtype=float)
    for g in G.values():
        np.testing.assert_allclose(g.mean(axis=0), 0, atol=1e-12)         # average reference
        np.testing.assert_allclose(g[-1] / (5 - 3), expected)              # last channel: (5 - mean 3) * id

    # one source space
    G, vertices = mb.common_leadfields({"a": fake_forward([[1, 2, 3]]), "b": fake_forward([[2, 3]])})
    assert [v.tolist() for v in vertices] == [[2, 3]]
    assert G["a"].shape == (5, 2)


def test_common_leadfields_needs_fixed_orientation():
    with pytest.raises(ValueError, match="fixed-orientation"):
        mb.common_leadfields({"a": fake_forward([[1, 2]], per_source=3)})


# --- CHARM ----------------------------------------------------------------

@pytest.fixture
def fake_charm(tmp_path, seg):
    """
    Executable standing in for SimNIBS' charm: logs its argv, prints progress and
    writes a labeling. Subject "fail" makes it exit with an error instead; subject
    "hang" starts a child process, writes both pids to pids.txt and never finishes.
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
        "if sys.argv[1] == 'hang':\n"
        "    import os, subprocess, time\n"
        "    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        "    pathlib.Path('pids.tmp').write_text(f'{os.getpid()} {child.pid}')\n"
        "    os.replace('pids.tmp', 'pids.txt')\n"
        "    time.sleep(600)\n"
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


def _wait_for(path, timeout=20.0):
    import time
    end = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < end, f"{path} never appeared"
        time.sleep(0.05)
    return [int(pid) for pid in path.read_text().split()]


def _gone(pid, timeout=20.0):
    """True once ``pid`` no longer exists (zombies are reaped by init quickly)."""
    import time
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


posix_only = pytest.mark.skipif(not hasattr(os, "killpg"), reason="process groups are POSIX only")


@posix_only
def test_run_charm_interrupt_stops_charm(fake_charm, tmp_path):
    import _thread
    import threading
    cmd, _ = fake_charm
    out = tmp_path/"out"
    pids = []

    def interrupt():
        pids.extend(_wait_for(out/"pids.txt"))
        _thread.interrupt_main()               # what Ctrl+C / a Jupyter interrupt does

    threading.Thread(target=interrupt, daemon=True).start()
    with pytest.raises(KeyboardInterrupt):
        mb.run_charm("hang", tmp_path/"t1.nii.gz", out, charm_cmd=cmd)

    assert len(pids) == 2 and all(_gone(pid) for pid in pids)     # CHARM and its child
    assert not mb.charm._RUNNING


@posix_only
def test_stop_charm_from_main_thread(fake_charm, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    cmd, _ = fake_charm
    out = tmp_path/"out"

    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(mb.run_charm, "hang", tmp_path/"t1.nii.gz", out, charm_cmd=cmd)
        pids = _wait_for(out/"pids.txt")
        assert mb.stop_charm() == 1
        with pytest.raises(RuntimeError, match="stopped"):
            future.result(timeout=30)

    assert all(_gone(pid) for pid in pids)
    assert not mb.charm._RUNNING and mb.stop_charm() == 0


def test_run_charm_log(fake_charm, tmp_path):
    cmd, _ = fake_charm
    t1, out = tmp_path/"t1.nii.gz", tmp_path/"out"

    log = tmp_path/"logs"/"sub06_charm.log"                      # folder created
    mb.run_charm("sub06", t1, out, charm_cmd=cmd, log=log)
    assert "charm progress" in log.read_text()
    assert not (out/"charm_sub06.log").exists()

    with pytest.raises(RuntimeError, match=f"(?s){log.name}.*charm error"):
        mb.run_charm("fail", t1, out, charm_cmd=cmd, log=log)

    with pytest.raises(ValueError, match="must not be inside"):
        mb.run_charm("sub07", t1, out, charm_cmd=cmd, log=out/"m2m_sub07"/"charm.log")
    with pytest.raises(ValueError, match="verbose"):
        mb.run_charm("sub07", t1, out, charm_cmd=cmd, log=log, verbose=True)


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


def test_bem_from_t1_labels(fake_charm, tmp_path, monkeypatch):
    cmd, _ = fake_charm
    seen = {}
    monkeypatch.setattr("mindmap_bem.charm.build_bem", lambda seg, out, labels, **kw: seen.update(labels=labels))

    mb.bem_from_t1("sub01", tmp_path/"t1.nii.gz", tmp_path/"out", labels="my_labels.yml",
                   charm_kwargs={"charm_cmd": cmd})
    assert seen["labels"] == "my_labels.yml"


def test_cli_run_labels(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr("mindmap_bem.cli.bem_from_t1", lambda *a, **kw: seen.update(kw))
    cli_main(["run", "sub01", "t1.nii.gz", "--out", str(tmp_path), "--labels", "my_labels.yml"])
    assert seen["labels"] == "my_labels.yml"



def test_cli_build_no_check(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr("mindmap_bem.cli.build_bem", lambda *a, **kw: seen.update(kw))
    cli_main(["build", "seg.nii.gz", "--out", str(tmp_path)])
    assert seen["check"] is True
    cli_main(["build", "seg.nii.gz", "--out", str(tmp_path), "--no-check"])
    assert seen["check"] is False


# CLI wiring only: the CLI can't set z_floors, and with the default floors the synthetic
# head folds with some VTK versions (see test_z_floor_in_mm), hence --no-check
@pytest.mark.filterwarnings("ignore:Writing BEM layers that fail check_bem")
def test_cli_build_end_to_end(seg, tmp_path, capsys):
    import yaml
    nib.save(seg, tmp_path/"seg.nii.gz")
    (tmp_path/"labels.yml").write_text(yaml.safe_dump(LABELS))
    cli_main(["build", str(tmp_path/"seg.nii.gz"), "--out", str(tmp_path/"bem"),
              "--labels", str(tmp_path/"labels.yml"), "--closing-radius", "3", "--no-check"])
    assert "BEM surfaces written to" in capsys.readouterr().out
    assert set(mb.read_bem_surfaces(tmp_path/"bem")) == {"inner_skull", "outer_skull", "skin"}

    with pytest.raises(FileExistsError):                         # no --overwrite
        cli_main(["build", str(tmp_path/"seg.nii.gz"), "--out", str(tmp_path/"bem"),
                  "--labels", str(tmp_path/"labels.yml"), "--no-check"])


def test_cli_run_arguments(monkeypatch, tmp_path, capsys):
    seen = {}
    monkeypatch.setattr("mindmap_bem.cli.bem_from_t1", lambda *a, **kw: seen.update(args=a, **kw))
    cli_main(["run", "sub01", "t1.nii.gz", "--out", str(tmp_path), "--t2", "t2.nii.gz",
              "--subjects-dir", "subjects", "--rerun-charm", "--fs-t1", "T1.mgz", "--brain-mask", "brain.nii.gz",
              "--n-smooth", "5", "--iterations", "50", "--closing-radius", "4", "--overwrite", "--no-check",
              "--charm-args", "--forceqform", "--usetransform", "x.txt"])
    assert seen["args"] == ("sub01", "t1.nii.gz", str(tmp_path))
    assert seen["t2"] == "t2.nii.gz" and seen["subjects_dir"] == "subjects" and seen["rerun_charm"] is True
    assert seen["fs_t1"] == "T1.mgz" and seen["brain_mask"] == "brain.nii.gz"
    assert (seen["n_smooth"], seen["iterations"], seen["closing_radius"]) == (5, 50, 4.0)
    assert seen["overwrite"] is True and seen["check"] is False
    assert seen["charm_kwargs"] == {"extra_args": ("--forceqform", "--usetransform", "x.txt"), "verbose": True}
    assert "subjects/sub01/bem" in capsys.readouterr().out

    # defaults
    cli_main(["run", "sub01", "t1.nii.gz", "--out", str(tmp_path)])
    assert seen["labels"] == "charm" and seen["t2"] is None and seen["rerun_charm"] is False
    assert (seen["n_smooth"], seen["iterations"], seen["closing_radius"]) == (20, 100, 10.0)
    assert seen["charm_kwargs"]["extra_args"] == () and seen["check"] is True
    assert f"{tmp_path}/bem" in capsys.readouterr().out


def test_cli_usage_errors(capsys):
    with pytest.raises(SystemExit):
        cli_main([])                                              # no command
    with pytest.raises(SystemExit):
        cli_main(["build", "seg.nii.gz"])                         # --out is required
    with pytest.raises(SystemExit) as e:
        cli_main(["--help"])
    assert e.value.code == 0 and "build" in capsys.readouterr().out
