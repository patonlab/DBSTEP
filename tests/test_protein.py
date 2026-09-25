"""Residue selection in PDB input: --residue/--atom, water/hetero/self filters, and equivalence with the XYZ path."""

import itertools
import re
import subprocess
import sys

import numpy as np
import pytest

from dbstep import Dbstep, parse_data, selection

pdb_dir = "tests/pdb_files/"
A8O = pdb_dir + "1a8o.pdb"
ALA5 = pdb_dir + "ala5.pdb"


def run(file, **kwargs):
	return Dbstep.dbstep(file, quiet=True, **kwargs)


def write_xyz(path, atoms, coords):
	with open(path, "w") as f:
		f.write(f"{len(atoms)}\nextracted\n")
		for atom, (x, y, z) in zip(atoms, coords):
			f.write(f"{atom} {x:.6f} {y:.6f} {z:.6f}\n")


def rerun_as_xyz(mol, path, **kwargs):
	"""Write the structure a run actually measured to XYZ and run it through the ordinary numeric path."""
	write_xyz(path, mol.atoms, mol.coords)
	return run(str(path), atom1=mol.spec_atoms[0], atom2=mol.spec_atoms[1:], cutoff="none", **kwargs)


def results(mol):
	return (mol.L, mol.Bmin, mol.Bmax, mol.bur_vol, mol.bur_shell, mol.occ_vol)


# --- exact equivalence with the XYZ path ------------------------------------------------------------

FLAG_SETS = [
	{},
	{"nowater": True},
	{"nowater": True, "exclude_self": True},
	{"self_only": True},
	{"nohet": True, "noH": True},
	{"nowater": True, "nohet": True, "exclude_self": True},
]


@pytest.mark.parametrize("resid, flags", list(itertools.product(["A:160", "A:186", "A:200"], FLAG_SETS)))
def test_residue_run_equals_extracted_xyz(tmp_path, resid, flags):
	"""Same atoms, same spec atoms, same grid: the residue run and the XYZ of what it kept must match exactly."""
	from_pdb = run(A8O, residue=resid, volume=True, sterimol=True, grid=0.1, **flags)
	from_xyz = rerun_as_xyz(from_pdb, tmp_path / "kept.xyz", volume=True, sterimol=True, grid=0.1)
	assert from_pdb.n_atoms_kept == len(from_pdb.atoms) == from_xyz.n_atoms_total
	assert results(from_pdb) == pytest.approx(results(from_xyz), abs=1e-9)


def test_residue_run_with_scan_equals_extracted_xyz(tmp_path):
	from_pdb = run(A8O, residue="A:186", volume=True, sterimol=True, scan="2.0:4.0:1.0", nowater=True, grid=0.1)
	from_xyz = rerun_as_xyz(from_pdb, tmp_path / "kept.xyz", volume=True, sterimol=True, scan="2.0:4.0:1.0", grid=0.1)
	assert from_pdb.bur_vol == pytest.approx(from_xyz.bur_vol, abs=1e-9)
	assert from_pdb.Bmax == pytest.approx(from_xyz.Bmax, abs=1e-9)
	assert from_pdb.L == pytest.approx(from_xyz.L, abs=1e-9)


# --- crop invariance on a peptide small enough for a whole-molecule grid ----------------------------


def ala5_index(name, resseq):
	mol = parse_data.read_input(ALA5, ".pdb", Dbstep.set_options({"atom1": 1, "atom2": [2]}))
	return int(np.where((mol.METADATA["resseq"] == resseq) & (mol.METADATA["name"] == name))[0][0]) + 1


def test_crop_invariance_against_whole_peptide(tmp_path):
	whole = run(ALA5, atom1=ala5_index("CA", 3), atom2=ala5_index("CB", 3), volume=True, sterimol=True)
	cropped = run(ALA5, residue="A:3", volume=True, sterimol=True)
	uncropped = run(ALA5, residue="A:3", volume=True, sterimol=True, cutoff="none")
	assert cropped.n_atoms_kept < cropped.n_atoms_total == 56
	assert cropped.bur_vol == pytest.approx(whole.bur_vol, abs=1e-9)
	assert results(uncropped) == pytest.approx(results(whole), abs=1e-9)


# --- semantics --------------------------------------------------------------------------------------


def test_self_only_equals_residue_extracted_to_its_own_file(tmp_path):
	mol = parse_data.read_input(ALA5, ".pdb", Dbstep.set_options({"atom1": 1, "atom2": [2]}))
	in_res = mol.METADATA["resseq"] == 3
	xyz = tmp_path / "res3.xyz"
	write_xyz(xyz, mol.ATOMTYPES[in_res], mol.CARTESIANS[in_res])
	names = list(mol.METADATA["name"][in_res])
	extracted = run(str(xyz), atom1=names.index("CA") + 1, atom2=names.index("CB") + 1, volume=True, sterimol=True)
	self_only = run(ALA5, residue="A:3", self_only=True, volume=True, sterimol=True, cutoff="none")
	assert self_only.n_atoms_kept == int(in_res.sum())
	assert results(self_only) == pytest.approx(results(extracted), abs=1e-9)


def test_exclude_self_lowers_buried_volume_and_keeps_alignment():
	full = run(ALA5, residue="A:3", volume=True, sterimol=True)
	env = run(ALA5, residue="A:3", exclude_self=True, volume=True, sterimol=True)
	assert env.bur_vol < full.bur_vol
	# the residue's atoms remain only as the ghost spec atoms
	assert set(env.atoms[[i - 1 for i in env.spec_atoms]]) == {"Bq"}
	assert env.L > 0


def test_nowater_only_matters_when_a_water_is_within_reach():
	# ala5: one water 3.2 A from CA of residue 3 (inside the sphere), one 9 A away (outside)
	full = run(ALA5, residue="A:3", volume=True)
	dry = run(ALA5, residue="A:3", volume=True, nowater=True)
	assert dry.bur_vol < full.bur_vol
	assert "O" in full.metadata["element"][full.metadata["resname"] == "HOH"]
	assert not (dry.metadata["resname"] == "HOH").any()
	# residue 1 is far from both waters: nothing changes
	far = run(ALA5, residue="A:1", volume=True)
	far_dry = run(ALA5, residue="A:1", volume=True, nowater=True)
	assert far.bur_vol == pytest.approx(far_dry.bur_vol)


def test_nohet_drops_ions_but_keeps_waters_and_modified_residues():
	full = run(ALA5, residue="A:3", volume=True, cutoff="none")
	nohet = run(ALA5, residue="A:3", volume=True, nohet=True, cutoff="none")
	assert "Na" in full.metadata["element"] and "Na" not in nohet.metadata["element"]
	assert (nohet.metadata["resname"] == "HOH").any()
	assert nohet.bur_vol <= full.bur_vol
	# 1A8O: selenomethionine is HETATM but has a backbone, so it survives --nohet
	mse_neighbour = run(A8O, residue="A:152", volume=True, nohet=True)  # A:151 is MSE, well inside the auto cutoff
	assert "MSE" in set(mse_neighbour.metadata["resname"])


def test_chain_filter(tmp_path):
	lines = [line for line in open(ALA5).read().splitlines() if line.startswith("ATOM")]
	chain_b = [line[:21] + "B" + line[22:30] + f"{float(line[30:38]) + 4.0:8.3f}" + line[38:] for line in lines]
	pdb = tmp_path / "two_chains.pdb"
	pdb.write_text("\n".join(lines + ["TER"] + chain_b + ["END"]) + "\n")
	both = run(str(pdb), residue="A:3", volume=True)
	only_a = run(str(pdb), residue="A:3", volume=True, chain="A")
	assert set(both.metadata["chain"]) == {"A", "B"}
	assert set(only_a.metadata["chain"]) == {"A"}
	assert only_a.bur_vol < both.bur_vol
	with pytest.raises(SystemExit, match="chains A, B"):
		run(str(pdb), residue="3", volume=True)


def test_atom_names_and_defaults():
	mol = run(ALA5, residue="A:3", volume=True, sterimol=True)
	assert (mol.atom1, mol.atom2) == ("CA", ["CB"])
	assert mol.residue_label == "A:3 ALA"
	custom = run(ALA5, residue="A:3", atom="N", atom2="C", sterimol=True)
	assert (custom.atom1, custom.atom2) == ("N", ["C"])
	# numeric indices still work in residue mode
	numeric = run(ALA5, residue="A:3", atom=ala5_index("N", 3), atom2=ala5_index("C", 3), sterimol=True)
	assert results(numeric) == pytest.approx(results(custom))


def test_glycine_falls_back_to_backbone_nitrogen(capsys):
	mol = Dbstep.dbstep(A8O, residue="A:156", sterimol=True, quiet=True, verbose=True)
	assert mol.atom2 == ["N"]
	assert "using N as atom2" in capsys.readouterr().out


def test_insertion_code_and_lower_case_selection(tmp_path):
	lines = open(ALA5).read().splitlines()
	patched = [line[:26] + "A" + line[27:] if line.startswith("ATOM") and int(line[22:26]) == 3 else line for line in lines]
	pdb = tmp_path / "icode.pdb"
	pdb.write_text("\n".join(patched) + "\n")
	mol = run(str(pdb), residue="a:3a", volume=True)
	assert mol.residue_label == "a:3a ALA"
	assert mol.n_atoms_kept > 0


@pytest.mark.parametrize(
	"kwargs, message",
	[
		({"residue": "A:999"}, "not found"),
		({"residue": "A:3", "atom": "XX"}, "No atom named XX"),
		({"residue": "A:3", "self_only": True, "exclude_self": True}, "cannot be combined"),
		({"residue": "all"}, "all_residues"),
	],
)
def test_selection_errors(kwargs, message):
	with pytest.raises(SystemExit, match=message):
		run(ALA5, volume=True, **kwargs)


def test_residue_requires_pdb_input():
	with pytest.raises(SystemExit, match="requires PDB input"):
		run("dbstep/data/Et.xyz", residue="A:1", volume=True)


def test_residue_mask_helpers():
	mol = parse_data.read_input(ALA5, ".pdb", Dbstep.set_options({"atom1": 1, "atom2": [2]}))
	assert selection.residue_mask(mol, "A:3").sum() == int((mol.METADATA["resseq"] == 3).sum())
	assert selection.residue_mask(mol, "A:1,A:2").sum() == int((mol.METADATA["resseq"] <= 2).sum())
	assert selection.water_mask(mol).sum() == 2
	assert selection.find_atom_by_name(mol, selection.residue_mask(mol, "A:3"), "cb") == ala5_index("CB", 3)


# --- command line -------------------------------------------------------------------------------------


def test_cli_residue_mode():
	cmd = [sys.executable, "-m", "dbstep", A8O, "--residue", "A:186", "-b", "-s", "--nowater", "--nohet", "-v"]
	out = subprocess.run(cmd, capture_output=True, text=True)
	assert out.returncode == 0, out.stdout + out.stderr
	text = out.stdout
	assert "PDB residue mode: measuring residue A:186, excluding waters and hetero groups" in text
	assert re.search(r"1a8o\.pdb A:186 \w{3}\s+CA\s+CB\s+3\.50\s+[\d.]+\s+[\d.]+\s+0\.00\s+[\d.]+\s+[\d.]+\s+[\d.]+", text)
	assert re.search(r"Cutoff [\d.]+ Ang around atom1: keeping \d+ of \d+ atoms", text)
