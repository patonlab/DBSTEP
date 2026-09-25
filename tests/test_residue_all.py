"""--residue all (one run per residue), per-run result records and --csv output."""

import csv
import re
import subprocess
import sys

import pytest

from dbstep import Dbstep

pdb_dir = "tests/pdb_files/"
A8O = pdb_dir + "1a8o.pdb"
ALA5 = pdb_dir + "ala5.pdb"


def run(file, **kwargs):
	return Dbstep.dbstep(file, quiet=True, **kwargs)


def values(mol):
	return (mol.L, mol.Bmin, mol.Bmax, mol.bur_vol, mol.occ_vol)


def run_cli(*args):
	result = subprocess.run([sys.executable, "-m", "dbstep", *args], capture_output=True, text=True)
	assert result.returncode == 0, result.stdout + result.stderr
	return result.stdout


# --- all residues -----------------------------------------------------------------------------------


def test_all_residues_matches_individual_runs():
	runs = Dbstep.all_residues(ALA5, volume=True, sterimol=True, quiet=True)
	assert [r.residue_label for r in runs] == [f"A:{i} ALA" for i in range(1, 6)]
	for mol in runs:
		single = run(ALA5, residue=mol.residue_label.split()[0], volume=True, sterimol=True)
		assert values(mol) == pytest.approx(values(single))


def test_all_residues_skips_waters_and_hetero_groups_but_keeps_modified_residues():
	runs = Dbstep.all_residues(A8O, volume=True, grid=0.25, quiet=True)
	labels = [r.residue_label for r in runs]
	assert len(runs) == 70
	assert "A:151 MSE" in labels
	assert not any("HOH" in label for label in labels)
	assert labels[0] == "A:151 MSE" and labels[-1] == "A:220 GLY"


def test_all_residues_with_chain_filter(tmp_path):
	lines = [line for line in open(ALA5).read().splitlines() if line.startswith("ATOM")]
	chain_b = [line[:21] + "B" + line[22:30] + f"{float(line[30:38]) + 4.0:8.3f}" + line[38:] for line in lines]
	pdb = tmp_path / "two_chains.pdb"
	pdb.write_text("\n".join(lines + ["TER"] + chain_b + ["END"]) + "\n")
	both = Dbstep.all_residues(str(pdb), volume=True, grid=0.2, quiet=True)
	only_b = Dbstep.all_residues(str(pdb), volume=True, grid=0.2, quiet=True, chain="B")
	assert len(both) == 10 and len(only_b) == 5
	assert all(r.residue_label.startswith("B:") for r in only_b)


def test_all_residues_uses_atom_name_and_rejects_indices():
	runs = Dbstep.all_residues(ALA5, atom="N", volume=True, grid=0.2, quiet=True)
	assert len(runs) == 5 and all(r.atom1 == "N" for r in runs)
	with pytest.raises(SystemExit, match="must be an atom name"):
		Dbstep.all_residues(ALA5, atom=2, volume=True, quiet=True)
	with pytest.raises(SystemExit, match="requires PDB input"):
		Dbstep.all_residues("dbstep/data/Et.xyz", volume=True, quiet=True)


def test_dbstep_class_refuses_all():
	with pytest.raises(SystemExit, match="all_residues"):
		run(ALA5, residue="all", volume=True)


# --- result records ------------------------------------------------------------------------------------


def test_results_records_for_a_scan():
	mol = run(ALA5, residue="A:3", volume=True, sterimol=True, scan="2.0:4.0:1.0")
	assert [row["radius"] for row in mol.results] == [2.0, 3.0, 4.0]
	row = mol.results[0]
	assert row["file"] == "ala5.pdb" and row["frame"] == "" and row["structure"] == "" and row["residue"] == "A:3 ALA"
	assert (row["atom1"], row["atom2"]) == ("CA", "CB")
	assert row["percent_vbur"] == mol.bur_vol[0]
	assert row["bmin"] == mol.Bmin[0]
	# rows carry the per-slice L of a scan; mol.L is the overall length recomputed afterwards
	assert all(isinstance(r["L"], float) and r["L"] <= mol.L + 1e-9 for r in mol.results)
	assert mol.results[-1]["percent_vbur"] == mol.bur_vol[-1]


def test_results_records_volume_only_leave_sterimol_columns_empty():
	mol = run("dbstep/data/Et.xyz", atom1=2, volume=True)
	assert len(mol.results) == 1
	row = mol.results[0]
	assert (row["file"], row["residue"], row["atom1"], row["atom2"]) == ("Et.xyz", "", 2, "")
	assert row["percent_vbur"] == mol.bur_vol and row["bmin"] == "" and row["L"] == ""


# --- CSV output ---------------------------------------------------------------------------------------


def read_csv(path):
	with open(path, newline="") as f:
		return list(csv.DictReader(f))


def test_csv_all_residues_matches_table(tmp_path):
	out = tmp_path / "out.csv"
	text = run_cli(ALA5, "--residue", "all", "-b", "-s", "--csv", str(out))
	rows = read_csv(out)
	assert len(rows) == 5
	assert list(rows[0].keys()) == ["file", "frame", "structure", "residue", "atom1", "atom2", "radius", "mol_vol", "percent_vbur", "percent_sbur", "bmin", "bmax", "L"]
	assert f"Results written to {out}" in text
	for row in rows:
		printed = re.search(rf"ala5\.pdb {re.escape(row['residue'])}\s+CA\s+CB\s+3\.50\s+([\d.]+)\s+([\d.]+)\s+[\d.]+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", text)
		assert printed, row
		assert float(row["percent_vbur"]) == pytest.approx(float(printed.group(2)), abs=0.006)
		assert float(row["L"]) == pytest.approx(float(printed.group(5)), abs=0.006)


def test_csv_multi_model_all_residues(tmp_path):
	atom_lines = [line for line in open(ALA5).read().splitlines() if line.startswith(("ATOM", "HETATM"))]
	shifted = [line[:30] + f"{float(line[30:38]) + 1.0:8.3f}" + line[38:] for line in atom_lines]
	pdb = tmp_path / "models.pdb"
	pdb.write_text("\n".join(["MODEL        1", *atom_lines, "ENDMDL", "MODEL        2", *shifted, "ENDMDL", "END"]) + "\n")
	out = tmp_path / "models.csv"
	text = run_cli(str(pdb), "--residue", "all", "-b", "--grid", "0.2", "--csv", str(out))
	rows = read_csv(out)
	assert len(rows) == 10
	assert [row["structure"] for row in rows] == ["model1"] * 5 + ["model2"] * 5
	assert rows[0]["residue"] == "A:1 ALA" and rows[9]["residue"] == "A:5 ALA"
	assert "model2 A:3 ALA" in text


def test_csv_for_plain_xyz_input(tmp_path):
	out = tmp_path / "et.csv"
	run_cli("dbstep/data/Et.xyz", "--sterimol", "--atom1", "2", "--atom2", "5", "--csv", str(out))
	rows = read_csv(out)
	assert len(rows) == 1
	assert (rows[0]["file"], rows[0]["structure"], rows[0]["residue"], rows[0]["radius"]) == ("Et.xyz", "", "", "")
	assert (float(rows[0]["bmin"]), float(rows[0]["bmax"]), float(rows[0]["L"])) == pytest.approx((1.99, 2.13, 3.24), abs=0.006)
