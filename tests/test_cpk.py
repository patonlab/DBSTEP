"""--radii cpk: Sterimol CPK radii from coordination-number atom types, validated against Verloop's Fortran program."""

import csv
import re
import subprocess
import sys

import numpy as np
import pytest

from dbstep import Dbstep, radii, selection
from dbstep.constants import cpk, bondi

# Sterimol_Benchmark.csv of patonlab/sterimol: "Original Fortran" columns (L, B1, B5) for the substituents in dbstep/data
FORTRAN = {
	"H": (2.06, 1.00, 1.00),
	"Me": (3.00, 1.52, 2.04),
	"Et": (4.11, 1.52, 3.17),
	"iPr": (4.11, 1.90, 3.17),
	"nBu": (6.17, 1.52, 4.54),
	"CH2iPr": (5.05, 1.52, 4.45),
	"cHex": (6.17, 1.91, 3.49),
	"nPr": (5.05, 1.52, 3.49),
	"Ad": (6.17, 3.16, 3.49),
	"tBu": (4.11, 2.77, 3.17),
	"CH2tBu": (5.05, 1.52, 4.45),
	"CHEt2": (5.05, 1.90, 4.45),
	"CHiPr2": (5.05, 2.05, 4.45),
	"CHPr2": (6.17, 1.90, 5.67),
	"CEt3": (5.05, 2.77, 4.45),
	"Ph": (6.28, 1.71, 3.11),
	"Bn": (4.62, 1.52, 6.02),
	"4ClPh": (7.74, 1.80, 3.11),
	"4MePh": (7.22, 1.71, 3.11),
	"4MeOPh": (8.20, 1.78, 3.11),
	"35diMePh": (6.28, 1.71, 4.30),
	"1Nap": (6.28, 1.71, 5.50),
}

METHYL_H = "H 1.03 0.0 -0.36\nH -0.51 0.89 -0.36\nH -0.51 -0.89 -0.36\n"


def run(file, **kwargs):
	return Dbstep.dbstep(file, quiet=True, **kwargs)


def write_xyz(tmp_path, name, body):
	lines = [line for line in body.strip().splitlines()]
	path = tmp_path / f"{name}.xyz"
	path.write_text(f"{len(lines)}\n{name}\n" + "\n".join(lines) + "\n")
	return str(path)


# --- validation against the original Fortran program -------------------------------------------------


@pytest.mark.parametrize("name, expected", FORTRAN.items())
def test_classic_sterimol_with_cpk_radii_matches_fortran(name, expected):
	mol = run(f"dbstep/data/{name}.xyz", sterimol=True, measure="classic", radii="cpk")
	# Verloop's L includes the 0.40 Ang correction for the H atom at the origin (see test_dbstep.py)
	got = (round(mol.L + 0.4, 2), round(mol.Bmin, 2), round(mol.Bmax, 2))
	assert got == pytest.approx(expected, abs=0.011), (name, got, expected)


# --- atom typing ------------------------------------------------------------------------------------


def test_coordination_numbers_of_methane_like_carbon():
	mol = run("dbstep/data/Me.xyz", sterimol=True, radii="cpk")
	cn = radii.coordination_numbers(mol.atoms, mol.coords)
	assert cn[list(mol.atoms).index("C")] == pytest.approx(4.0, abs=0.1)
	assert all(c == pytest.approx(1.0, abs=0.1) for c, atom in zip(cn, mol.atoms) if atom == "H")
	assert list(mol.metadata["cpk_type"]) == ["H", "C", "H", "H", "H"]


@pytest.mark.parametrize(
	"name, body, expected",
	[
		("acetonitrile", "C 0 0 0\nC 0 0 1.46\nN 0 0 2.62\n" + METHYL_H, {"C": {"C", "C3"}, "N": {"C6/N6"}}),
		("acetone", "C 0 0 0\nO 0 0 1.22\nC 1.29 0 -0.78\nC -1.29 0 -0.78\nH 2.2 0 -0.2\nH 1.4 0.9 -1.4\nH 1.4 -0.9 -1.4\nH -2.2 0 -0.2\nH -1.4 0.9 -1.4\nH -1.4 -0.9 -1.4\n", {"O": {"O2"}, "C": {"C6/N6", "C"}}),
		("methanol", "C 0 0 0\nO 0 0 1.43\nH 0.9 0 1.75\n" + METHYL_H, {"O": {"O"}, "C": {"C"}}),
		("methylamine", "C 0 0 0\nN 0 0 1.47\nH 0.85 0 1.95\nH -0.85 0 1.95\n" + METHYL_H, {"N": {"N"}}),
		("dimethylsulfone", "S 0 0 0\nO 0 1.02 1.02\nO 0 -1.02 1.02\nC 1.46 0 -1.02\nC -1.46 0 -1.02\n", {"S": {"S4"}, "O": {"O2"}}),
		("hydrogen_sulfide", "S 0 0 0\nH 1.34 0 0\nH -0.39 1.28 0\n", {"S": {"S"}}),
		("chlorobromoiodide", "C 0 0 0\nCl 1.77 0 0\nBr -0.95 1.65 0\nI -0.95 -1.85 0\nF 0 0 1.35\n", {"Cl": {"C1"}, "Br": {"B1"}, "I": {"I"}, "F": {"F"}}),
	],
)
def test_sterimol_atom_types(tmp_path, name, body, expected):
	xyz = write_xyz(tmp_path, name, body)
	mol = run(xyz, sterimol=True, radii="cpk")
	types = {}
	for element, cpk_type in zip(mol.atoms, mol.metadata["cpk_type"]):
		types.setdefault(element, set()).add(cpk_type)
	for element, wanted in expected.items():
		assert types[element] == wanted, (element, types[element])
	assert all(cpk_type in cpk for cpk_type in mol.metadata["cpk_type"])


def test_types_are_assigned_before_hydrogens_are_removed():
	"""With --noH the methyl carbon keeps its sp3 type (CN counted with the hydrogens present)."""
	mol = run("dbstep/data/Me.xyz", sterimol=True, measure="classic", radii="cpk", noH=True)
	assert list(mol.atoms) == ["Bq", "C"]  # atom1 (H) is kept as a ghost
	assert list(mol.metadata["cpk_type"]) == ["H", "C"]
	assert mol.L == pytest.approx(1.10 + cpk["C"], abs=1e-6)
	assert mol.Bmax == pytest.approx(cpk["C"], abs=1e-6)


def test_elements_without_a_cpk_type_fall_back_to_bondi(tmp_path):
	xyz = write_xyz(tmp_path, "silyl", "C 0 0 0\nSi 0 0 1.87\n")
	mol = run(xyz, sterimol=True, measure="classic", radii="cpk", atom1=1, atom2=2)
	assert list(mol.metadata["cpk_type"]) == ["C3", ""]
	assert mol.L == pytest.approx(1.87 + bondi["Si"], abs=1e-6)


def test_bondi_results_are_unchanged():
	cpk_mol = run("dbstep/data/Ph.xyz", sterimol=True, measure="classic", radii="cpk")
	bondi_mol = run("dbstep/data/Ph.xyz", sterimol=True, measure="classic")
	assert "cpk_type" not in bondi_mol.metadata
	assert bondi_mol.Bmax == pytest.approx(3.20 - 0.0, abs=0.01) and cpk_mol.Bmax == pytest.approx(3.11, abs=0.01)


# --- interplay with cropping, residues and the CLI -----------------------------------------------------


def test_cutoff_bound_covers_every_cpk_radius():
	mol = run("dbstep/data/4ClPh.xyz", sterimol=True, radii="cpk")
	options = Dbstep.set_options({"radii": "cpk"})
	assigned = np.array([cpk[t] for t in mol.metadata["cpk_type"]])
	assert selection.max_vdw_radius(mol.atoms, options) >= assigned.max()
	assert selection.max_vdw_radius(mol.atoms, options) == pytest.approx(1.80)


def test_residue_buried_volume_with_cpk_is_crop_invariant():
	auto = run("tests/pdb_files/ala5.pdb", residue="A:3", volume=True, radii="cpk")
	full = run("tests/pdb_files/ala5.pdb", residue="A:3", volume=True, radii="cpk", cutoff="none")
	assert auto.bur_vol == pytest.approx(full.bur_vol, abs=1e-9)
	assert auto.n_atoms_kept < auto.n_atoms_total
	assert len(auto.metadata["cpk_type"]) == len(auto.atoms)
	# backbone: carbonyl O is double-bonded, N carries three neighbours (Verloop's rule types it tetrahedral), CA is sp3;
	# water oxygens have two neighbours
	names, types, resnames = auto.metadata["name"], auto.metadata["cpk_type"], auto.metadata["resname"]
	backbone = resnames == "ALA"
	assert set(types[backbone & (names == "O")]) == {"O2"} and set(types[backbone & (names == "N")]) == {"N"}
	assert set(types[backbone & (names == "CA")]) == {"C"}


def test_cli_radii_cpk(tmp_path):
	out = tmp_path / "cpk.csv"
	cmd = [sys.executable, "-m", "dbstep", "dbstep/data/Et.xyz", "--sterimol", "--radii", "cpk", "--csv", str(out)]
	result = subprocess.run(cmd, capture_output=True, text=True)
	assert result.returncode == 0, result.stdout + result.stderr
	assert "CPK (Sterimol atom types) atomic radii will be scaled by 1.0" in result.stdout
	assert re.search(r"Et\.xyz\s+1\s+2\s+1\.52\s+3\.17\s+3\.71", result.stdout)
	with open(out, newline="") as f:
		row = list(csv.DictReader(f))[0]
	assert (float(row["bmin"]), float(row["bmax"]), float(row["L"])) == pytest.approx((1.52, 3.17, 3.71), abs=0.006)
