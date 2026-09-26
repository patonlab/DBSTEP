"""Regressions for issues raised in code review of the 2.x PRs."""

import csv
import os
import shutil
import subprocess
import sys

import pytest

from dbstep import Dbstep, ensemble, parse_data, trajectory

pdb_dir = "tests/pdb_files/"
ALA5 = pdb_dir + "ala5.pdb"
TRAJ = pdb_dir + "ala5_traj.pdb"
ETHER = "tests/sdf_files/ether_conformers.sdf"


def run(file, **kwargs):
	return Dbstep.dbstep(file, quiet=True, **kwargs)


# --- integer energies in SDF data fields (PR 51) -------------------------------------------------------


def test_integer_energy_fields_are_accepted(tmp_path):
	text = open(ETHER).read()
	for old, new in (("1.6241422178985285", "0"), ("3.14336556931666", "2"), ("4.656221082683384", "4.5")):
		text = text.replace(old, new)
	sdf = tmp_path / "ints.sdf"
	sdf.write_text(text)
	runs = Dbstep.all_frames(str(sdf), atom1=3, volume=True, quiet=True)
	ensemble.boltzmann_average(runs)
	assert [r.energy for r in runs] == [0.0, 2.0, 4.5]
	assert runs[0].population > 0.9
	assert ensemble.field_number(" 12 ") == 12.0 and ensemble.field_number("E = -5.25 kcal") == -5.25 and ensemble.field_number("n/a") is None
	# the integer guard still protects xyz comment lines
	assert ensemble.number_in_text("ether 44") is None


# --- --nometals through the ghost/renumber path (PR 53) -------------------------------------------------


def test_nometals_keeps_metadata_aligned_for_decompose():
	mol = run(ALA5, residue="A:3", volume=True, decompose=True, nometals=True, cutoff="none")
	assert "Na" not in mol.atoms
	assert len(mol.metadata["resname"]) == len(mol.atoms) == len(mol.coords)
	assert not any("NA" in label for label in mol.contributions)
	assert sum(mol.contributions.values()) == pytest.approx(mol.bur_vol, abs=1e-5)


def test_nometals_renumbers_spec_atoms(tmp_path):
	"""A metal listed before atom1 used to shift the spec atom indices after removal."""
	src = parse_data.read_input("dbstep/data/Et.xyz", ".xyz", Dbstep.set_options({"atom1": 1, "atom2": [2]}))
	xyz = tmp_path / "na_et.xyz"
	with open(xyz, "w") as f:
		f.write(f"{len(src.ATOMTYPES) + 1}\nsodium far away, then ethane\nNa 9.0 9.0 9.0\n")
		for atom, (x, y, z) in zip(src.ATOMTYPES, src.CARTESIANS):
			f.write(f"{atom} {x:.5f} {y:.5f} {z:.5f}\n")
	ref = run("dbstep/data/Et.xyz", atom1=2, atom2=5, sterimol=True, volume=True)
	with_metal = run(str(xyz), atom1=3, atom2=6, sterimol=True, volume=True, nometals=True)
	assert with_metal.n_atoms_kept == with_metal.n_atoms_total == 9
	assert (with_metal.L, with_metal.Bmin, with_metal.Bmax) == pytest.approx((ref.L, ref.Bmin, ref.Bmax), abs=1e-9)
	assert with_metal.bur_vol == pytest.approx(ref.bur_vol, abs=1e-9)
	assert with_metal.spec_atoms == [2, 5]


def test_metal_as_atom1_becomes_ghost_with_nometals(tmp_path):
	xyz = tmp_path / "fe_center.xyz"
	xyz.write_text("3\nFe with two carbons\nFe 0 0 0\nC 2.0 0 0\nC 0 2.0 0\n")
	mol = run(str(xyz), atom1=1, atom2=2, sterimol=True, volume=True, nometals=True)
	assert list(mol.atoms) == ["Bq", "C", "C"]
	assert mol.spec_atoms == [1, 2]
	assert mol.L == pytest.approx(2.0 + 1.70, abs=1e-6)  # C along the axis plus its Bondi radius


# --- --decompose with a scan starting at R = 0 (PR 53) -------------------------------------------------


def test_decompose_scan_from_zero_stays_aligned():
	mol = run(ALA5, residue="A:3", volume=True, decompose=True, scan="0.0:4.0:2.0")
	assert len(mol.contributions) == len(mol.results) == 3
	assert mol.contributions[0] == {} and mol.bur_vol[0] == 0.0
	for contributions, vbur in zip(mol.contributions[1:], mol.bur_vol[1:]):
		assert sum(contributions.values()) == pytest.approx(vbur, abs=1e-5)
	rows = Dbstep.contribution_rows(mol)
	assert {row["radius"] for row in rows} == {2.0, 4.0}


# --- tensor files per frame and residue (PR 48) ----------------------------------------------------------


def test_tensor_save_writes_one_file_per_frame(tmp_path):
	traj = tmp_path / "traj.pdb"
	shutil.copy(TRAJ, traj)
	Dbstep.all_frames(str(traj), frames="0,1", atom1=2, atom2=5, atom3=1, tensor=True, save=True, grid=1.0, quiet=True)
	files = sorted(p.name for p in tmp_path.glob("*_tensor.npy"))
	assert files == ["traj_frame0_tensor.npy", "traj_frame1_tensor.npy"]
	single = tmp_path / "ala5.pdb"
	shutil.copy(ALA5, single)
	run(str(single), residue="A:3", atom3="N", tensor=True, save=True, grid=1.0)
	assert (tmp_path / "ala5_A3_ALA_tensor.npy").exists()


# --- frames handling (PR 48) ----------------------------------------------------------------------------


def test_all_frames_restores_options_and_accepts_frame_zero():
	options = Dbstep.set_options({"atom1": 3, "volume": True, "quiet": True})
	Dbstep.all_frames(ETHER, frames="2", options=options)
	assert options.frames is False
	runs = Dbstep.all_frames(ETHER, frames=0, atom1=3, volume=True, quiet=True)
	assert [r.results[0]["frame"] for r in runs] == [0]
	assert trajectory.parse_frames(0, 10) == [0]
	assert trajectory.parse_frames([0], 10) == [0]
	options.frames = 0
	assert trajectory.frame_indices("dbstep/data/Et.xyz", options) == [0]


# --- all-residue mode honours a named atom1 (PR 47) ------------------------------------------------------


def test_all_residues_uses_named_atom1():
	runs = Dbstep.all_residues(ALA5, atom1="N", volume=True, grid=0.2, quiet=True)
	assert len(runs) == 5 and all(r.atom1 == "N" for r in runs)
	with pytest.raises(SystemExit, match="atom name"):
		Dbstep.all_residues(ALA5, atom1=2, volume=True, quiet=True)


# --- atom2 must differ from atom1 (PR 46) ---------------------------------------------------------------


def test_default_atom2_never_equals_atom1():
	mol = run(ALA5, residue="A:3", atom="CB", sterimol=True)
	assert mol.atom1 == "CB" and mol.atom2 != ["CB"]
	assert mol.spec_atoms[0] != mol.spec_atoms[1]
	assert mol.L > 0
	with pytest.raises(SystemExit, match="same atom as atom1"):
		run(ALA5, residue="A:3", atom="CA", atom2="CA", sterimol=True)


# --- path column (PR 47) ------------------------------------------------------------------------------


def test_path_column_distinguishes_same_named_inputs(tmp_path):
	a, b = tmp_path / "a", tmp_path / "b"
	a.mkdir(), b.mkdir()
	shutil.copy(ALA5, a / "sample.pdb")
	shutil.copy(ALA5, b / "sample.pdb")
	out = tmp_path / "out.csv"
	cmd = [sys.executable, "-m", "dbstep", str(a / "sample.pdb"), str(b / "sample.pdb"), "--residue", "A:3", "-b", "--decompose", "--csv", str(out)]
	result = subprocess.run(cmd, capture_output=True, text=True)
	assert result.returncode == 0, result.stdout + result.stderr
	with open(out, newline="") as f:
		rows = list(csv.DictReader(f))
	assert [row["file"] for row in rows] == ["sample.pdb", "sample.pdb"]
	assert [row["path"] for row in rows] == [str(a / "sample.pdb"), str(b / "sample.pdb")]
	with open(tmp_path / "out_contributions.csv", newline="") as f:
		contributions = list(csv.DictReader(f))
	assert {row["path"] for row in contributions} == {str(a / "sample.pdb"), str(b / "sample.pdb")}
	mol = run(ALA5, residue="A:3", volume=True)
	assert mol.results[0]["path"] == ALA5 and os.path.basename(mol.results[0]["path"]) == mol.results[0]["file"]
