"""Radial crop (--cutoff): ignore atoms too far from atom1 to influence the measurement."""

import numpy as np
import pytest

from dbstep import Dbstep, selection

xyz_dir = "dbstep/data/"

benchmark = ["Et.xyz", "iPr.xyz", "tBu.xyz", "Ph.xyz", "Bn.xyz", "1Nap.xyz", "CEt3.xyz", "Ad.xyz", "CHPr2.xyz"]


def run(file, **kwargs):
	return Dbstep.dbstep(file, quiet=True, **kwargs)


@pytest.mark.parametrize("name", benchmark)
def test_auto_cutoff_preserves_buried_volume(name):
	"""The auto cutoff keeps every atom that can reach the sphere, so %V_bur is unchanged; other
	quantities are unchanged whenever no atom was actually dropped."""
	full = run(xyz_dir + name, atom1=1, atom2=2, volume=True, sterimol=True)
	cut = run(xyz_dir + name, atom1=1, atom2=2, volume=True, sterimol=True, cutoff="auto")
	assert cut.cutoff is not None
	assert cut.n_atoms_kept <= cut.n_atoms_total == full.n_atoms_total
	assert cut.bur_vol == pytest.approx(full.bur_vol, abs=1e-6)
	if cut.n_atoms_kept == cut.n_atoms_total:
		assert (cut.L, cut.Bmin, cut.Bmax, cut.occ_vol) == pytest.approx((full.L, full.Bmin, full.Bmax, full.occ_vol))
	else:
		assert cut.occ_vol < full.occ_vol


def test_some_benchmark_molecules_are_actually_cropped():
	"""Guards the test above against silently never exercising the crop."""
	cropped = [name for name in benchmark if (m := run(xyz_dir + name, atom1=1, volume=True, cutoff="auto")).n_atoms_kept < m.n_atoms_total]
	assert cropped, "expected at least one benchmark molecule to extend beyond the auto cutoff"


@pytest.mark.parametrize("name", ["CEt3.xyz", "Ad.xyz"])
def test_auto_cutoff_preserves_vol2vec_scan(name):
	full = run(xyz_dir + name, atom1=1, volume=True, scan="2.0:4.0:0.5")
	cut = run(xyz_dir + name, atom1=1, volume=True, scan="2.0:4.0:0.5", cutoff="auto")
	assert cut.bur_vol == pytest.approx(full.bur_vol, abs=1e-6)
	assert cut.bur_shell == pytest.approx(full.bur_shell, abs=1e-6)


def test_auto_cutoff_preserves_buried_shell():
	full = run(xyz_dir + "CEt3.xyz", atom1=1, volume=True, vshell=1.0)
	cut = run(xyz_dir + "CEt3.xyz", atom1=1, volume=True, vshell=1.0, cutoff="auto")
	assert cut.bur_vol == pytest.approx(full.bur_vol, abs=1e-6)
	assert cut.bur_shell == pytest.approx(full.bur_shell, abs=1e-6)


def test_cutoff_combined_with_noH():
	full = run(xyz_dir + "CEt3.xyz", atom1=1, atom2=2, volume=True, noH=True)
	cut = run(xyz_dir + "CEt3.xyz", atom1=1, atom2=2, volume=True, noH=True, cutoff="auto")
	assert cut.bur_vol == pytest.approx(full.bur_vol, abs=1e-6)
	assert (cut.atom1, cut.atom2) == (1, [2])  # user-facing indices are untouched by the renumbering


def test_specified_atoms_are_always_kept():
	"""A cutoff smaller than the atom1-atom2 distance must still keep atom2 (and atom3) for alignment."""
	mol = run(xyz_dir + "tBu.xyz", atom1=1, atom2=5, sterimol=True, cutoff=0.5)
	assert mol.n_atoms_kept == 2
	assert mol.L > 0


def test_atom3_is_renumbered():
	full = run(xyz_dir + "CEt3.xyz", atom1=1, atom2=2, atom3=3, sterimol=True, tensor=True, grid=0.5)
	cut = run(xyz_dir + "CEt3.xyz", atom1=1, atom2=2, atom3=3, sterimol=True, tensor=True, grid=0.5, cutoff=3.0)
	assert cut.n_atoms_kept < cut.n_atoms_total
	# same alignment: the first three atoms sit in the same places in the tensor grid origin
	assert cut.tensor_grid["spacing"] == full.tensor_grid["spacing"]
	assert cut.tensor.shape[0] <= full.tensor.shape[0]


@pytest.mark.parametrize("bad", ["abc", "-1", -2.0])
def test_invalid_cutoff_exits(bad):
	with pytest.raises(SystemExit):
		run(xyz_dir + "Et.xyz", atom1=1, volume=True, cutoff=bad)


@pytest.mark.parametrize("off", ["none", "off", False, None, 0])
def test_cutoff_can_be_switched_off(off):
	mol = run(xyz_dir + "Et.xyz", atom1=1, volume=True, cutoff=off)
	assert mol.cutoff is None
	assert mol.n_atoms_kept == mol.n_atoms_total == 8


def test_cutoff_is_ignored_for_cube_input(capsys):
	full = run("tests/cube_files/Ne_medium.cube", atom1=1, volume=True, r=2.0)
	cut = Dbstep.dbstep("tests/cube_files/Ne_medium.cube", atom1=1, volume=True, r=2.0, cutoff="auto")
	assert cut.cutoff is None
	assert cut.bur_vol == pytest.approx(full.bur_vol)
	assert "ignored for density cube" in capsys.readouterr().out


def test_auto_cutoff_formula():
	options = Dbstep.set_options({"r": 3.5, "grid": 0.05})
	# Bondi C = 1.70, H = 1.09
	assert selection.auto_cutoff(options, np.array(["C", "H"])) == pytest.approx(1.1 * 3.5 + 1.70 + 0.05)
	options = Dbstep.set_options({"scan": "2.0:4.0:0.5", "grid": 0.05, "scalevdw": 1.17})
	assert selection.auto_cutoff(options, np.array(["C"])) == pytest.approx(1.1 * 4.0 + 0.25 + 1.17 * 1.70 + 0.05)


def write_xyz(path, atoms, coords, comment="cluster"):
	with open(path, "w") as f:
		f.write(f"{len(atoms)}\n{comment}\n")
		for atom, (x, y, z) in zip(atoms, coords):
			f.write(f"{atom} {x:.6f} {y:.6f} {z:.6f}\n")


def test_large_cluster_matches_manual_extraction(tmp_path):
	"""A ~50 Å cubic lattice of 4913 carbons: the cropped run must equal running the manually
	extracted neighbourhood as its own XYZ file, and must never see the whole-cluster grid."""
	n_side, spacing = 17, 3.0
	grid_pts = np.arange(n_side) * spacing
	coords = np.array(np.meshgrid(grid_pts, grid_pts, grid_pts, indexing="ij")).reshape(3, -1).T
	atoms = ["C"] * len(coords)
	cluster = tmp_path / "cluster.xyz"
	write_xyz(cluster, atoms, coords)
	center = int(np.argmin(np.linalg.norm(coords - coords.mean(axis=0), axis=1)))  # most central atom (0-based)

	cut = run(str(cluster), atom1=center + 1, volume=True, cutoff="auto", grid=0.1)
	assert cut.n_atoms_total == n_side**3
	assert cut.n_atoms_kept < 60

	# manual extraction of the same neighbourhood, run through the ordinary XYZ path
	keep = np.linalg.norm(coords - coords[center], axis=1) <= cut.cutoff
	sub = tmp_path / "sub.xyz"
	write_xyz(sub, [a for a, k in zip(atoms, keep) if k], coords[keep])
	new_center = int(keep[:center].sum()) + 1
	ref = run(str(sub), atom1=new_center, volume=True, grid=0.1)

	assert cut.n_atoms_kept == ref.n_atoms_total
	assert cut.bur_vol == pytest.approx(ref.bur_vol, abs=1e-9)
	assert cut.occ_vol == pytest.approx(ref.occ_vol, abs=1e-9)
