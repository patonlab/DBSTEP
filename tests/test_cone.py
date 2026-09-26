"""--cone: Tolman cone angles, metal-centroid distances and ligand Sterimol parameters of metal complexes."""

import csv
import math
import re
import subprocess
import sys

import numpy as np
import pytest

from dbstep import Dbstep, cone
from dbstep.constants import bondi

RH = "tests/metal_files/RhCpMe5Cl2PMe3.xyz"  # patonlab/sterimol example: [RhCp*Cl2(PMe3)]
CP_RING = [3, 4, 5, 24, 25]


def run(file, **kwargs):
	return Dbstep.dbstep(file, quiet=True, **kwargs)


def write_xyz(tmp_path, name, atoms, coords, comment=""):
	path = tmp_path / f"{name}.xyz"
	lines = [f"{a} {x:.6f} {y:.6f} {z:.6f}" for a, (x, y, z) in zip(atoms, coords)]
	path.write_text(f"{len(atoms)}\n{comment or name}\n" + "\n".join(lines) + "\n")
	return str(path)


def read_xyz(path):
	lines = open(path).read().splitlines()
	atoms = [line.split()[0] for line in lines[2:] if line.strip()]
	coords = np.array([[float(v) for v in line.split()[1:4]] for line in lines[2:] if line.strip()])
	return atoms, coords


# --- validation against patonlab/sterimol (CPK radii) --------------------------------------------------


def test_half_sandwich_matches_legacy_sterimol():
	"""sterimol.py examples/RhCpMe5Cl2PMe3.log: Tolman_CA 173.97, MC_dist 1.833, B1 3.902, B5 4.304."""
	mol = run(RH, cone=True, radii="cpk")
	assert mol.cone_angle == pytest.approx(173.97, abs=0.01)
	assert mol.metal_centroid == pytest.approx(1.833, abs=0.001)
	assert (mol.atom1, mol.atom2) == (1, CP_RING)
	assert len(mol.ligand_atoms) == 25 and mol.ligand_atoms[:3] == [3, 4, 5]
	assert mol.Bmin == pytest.approx(3.902, abs=0.02) and mol.Bmax == pytest.approx(4.304, abs=0.01)
	assert len(mol.cone_sectors) == 5
	# only the ligand is measured: the metal stays as a ghost at the origin, Cl and PMe3 are gone
	assert mol.n_atoms_kept == 26 and list(mol.atoms).count("Bq") == 1 and "Cl" not in mol.atoms and "P" not in mol.atoms
	assert mol.spec_atoms == [1, 2, 3, 4, 16, 17]  # renumbered to the kept atoms: ghost metal, then the ring carbons


def test_auto_detection_equals_explicit_atoms():
	auto = run(RH, cone=True, radii="cpk")
	explicit = run(RH, cone=True, radii="cpk", atom1=1, atom2="3,4,5,24,25")
	assert (auto.cone_angle, auto.metal_centroid, auto.Bmin, auto.Bmax, auto.L) == pytest.approx((explicit.cone_angle, explicit.metal_centroid, explicit.Bmin, explicit.Bmax, explicit.L))


def test_phosphine_by_donor_atom():
	mol = run(RH, cone=True, atom2=17, radii="cpk")
	assert len(mol.ligand_atoms) == 13 and len(mol.cone_sectors) == 3  # PMe3: one sector per methyl
	assert 110 < mol.cone_angle < 120  # Tolman's CPK-model value for PMe3 is 118
	assert mol.metal_centroid == pytest.approx(2.29, abs=0.01)  # the Rh-P bond length
	assert mol.atom2 == [17]


# --- definition checks on synthetic ligands -------------------------------------------------------------


def test_single_atom_ligand_is_the_angle_of_its_sphere(tmp_path):
	xyz = write_xyz(tmp_path, "pdcl", ["Pd", "Cl", "Cl"], [[0, 0, 0], [0, 0, 2.3], [0, 0, -2.3]])
	mol = run(xyz, cone=True, atom1=1, atom2=2)
	assert mol.cone_angle == pytest.approx(math.degrees(2 * math.asin(bondi["Cl"] / 2.3)), abs=1e-6)
	assert mol.metal_centroid == pytest.approx(2.3) and mol.ligand_atoms == [2] and len(mol.cone_sectors) == 1


def test_ph3_matches_tolman_construction(tmp_path):
	"""P at the origin, M on +z at 2.28 A, three H at 1.42 A with H-P-H = 93.5 degrees: theta/2 = alpha + asin(r/d)."""
	hph = math.radians(93.5)
	gamma = math.acos(math.sqrt((math.cos(hph) + 0.5) / 1.5))
	hs = [1.42 * np.array([math.sin(gamma) * math.cos(a), math.sin(gamma) * math.sin(a), -math.cos(gamma)]) for a in (0, 2 * math.pi / 3, 4 * math.pi / 3)]
	metal = np.array([0.0, 0.0, 2.28])
	xyz = write_xyz(tmp_path, "ph3", ["Ni", "P", "H", "H", "H"], [metal, [0, 0, 0], *hs])
	mol = run(xyz, cone=True, radii="cpk")  # auto: one metal, donor P
	axis = -metal / np.linalg.norm(metal)
	halves = []
	for h in hs:
		v = h - metal
		d = np.linalg.norm(v)
		halves.append(math.acos(np.dot(v, axis) / d) + math.asin(1.0 / d))
	assert mol.cone_angle == pytest.approx(math.degrees(2 * np.mean(halves)), abs=1e-4)  # xyz written with 6 decimals
	assert mol.atom2 == [2] and len(mol.cone_sectors) == 3


def test_invariant_to_rigid_motion_and_atom_order(tmp_path):
	atoms, coords = read_xyz(RH)
	ref = run(RH, cone=True, atom1=1, atom2="3,4,5,24,25")
	# random rotation + translation
	rng = np.random.default_rng(7)
	q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
	if np.linalg.det(q) < 0:
		q[:, 0] *= -1
	moved = write_xyz(tmp_path, "moved", atoms, coords @ q.T + np.array([5.0, -3.0, 12.0]))
	mol = run(moved, cone=True, atom1=1, atom2="3,4,5,24,25")
	assert (mol.cone_angle, mol.metal_centroid) == pytest.approx((ref.cone_angle, ref.metal_centroid), abs=1e-4)
	assert (mol.Bmin, mol.Bmax, mol.L) == pytest.approx((ref.Bmin, ref.Bmax, ref.L), abs=5e-3)  # classic B1 sweeps a finite set of angles
	# reversed atom order, auto-detected
	reversed_xyz = write_xyz(tmp_path, "reversed", atoms[::-1], coords[::-1])
	rev = run(reversed_xyz, cone=True)
	assert (rev.cone_angle, rev.metal_centroid) == pytest.approx((ref.cone_angle, ref.metal_centroid), abs=1e-4)
	assert rev.atom1 == 41 and sorted(rev.atom2) == [42 - i for i in CP_RING][::-1]


def test_ligand_buried_volume_from_the_metal():
	mol = run(RH, cone=True, volume=True, radius=3.5)
	assert 0 < mol.bur_vol < 100 and mol.cone_angle > 0
	assert mol.results[0]["cone_angle"] == mol.cone_angle and mol.results[0]["radius"] == 3.5


def test_module_helpers():
	atoms, coords = read_xyz(RH)
	assert cone.find_metal(atoms) == 0
	adjacency = cone.neighbours(atoms, coords, skip={0})
	assert adjacency[0] == set() and len(adjacency[2]) == 3  # ring carbon: two ring neighbours and its methyl carbon
	assert cone.find_ligand_atoms(atoms, coords, 0, adjacency) == [2, 3, 4, 23, 24]
	assert len(cone.connected(adjacency, [16])) == 13  # PMe3 through P17


# --- errors ----------------------------------------------------------------------------------------------


def test_errors(tmp_path):
	with pytest.raises(SystemExit, match="no metal"):
		run("dbstep/data/Et.xyz", cone=True)
	xyz = write_xyz(tmp_path, "two_metals", ["Pd", "Pd", "Cl"], [[0, 0, 0], [3, 0, 0], [0, 0, 2.3]])
	with pytest.raises(SystemExit, match="several metal"):
		run(xyz, cone=True)
	with pytest.raises(SystemExit, match="must not include"):
		run(RH, cone=True, atom1=1, atom2=1)
	with pytest.raises(SystemExit, match="--residue"):
		run("tests/pdb_files/ala5.pdb", cone=True, residue="A:3")


# --- CLI, CSV and Boltzmann averaging -----------------------------------------------------------------------


def test_cli_cone_table_and_csv(tmp_path):
	out = tmp_path / "cone.csv"
	cmd = [sys.executable, "-m", "dbstep", RH, "--cone", "--radii", "cpk", "--csv", str(out)]
	result = subprocess.run(cmd, capture_output=True, text=True)
	assert result.returncode == 0, result.stdout + result.stderr
	assert re.search(r"Bmin\s+Bmax\s+L\s+Cone/°\s+M-Cent/Å", result.stdout)
	assert re.search(r"RhCpMe5Cl2PMe3\.xyz\s+1\s+3,4,5,24,25\s+3\.9\d\s+4\.30\s+4\.0\d\s+173\.97\s+1\.83", result.stdout)
	assert "Cone angle of RhCpMe5Cl2PMe3.xyz: apex atom 1, ligand of 25 atoms (axis atoms 3,4,5,24,25)" in result.stdout
	with open(out, newline="") as f:
		rows = list(csv.DictReader(f))
	assert "cone_angle" in rows[0] and "metal_centroid" in rows[0]
	assert float(rows[0]["cone_angle"]) == pytest.approx(173.97, abs=0.01) and float(rows[0]["metal_centroid"]) == pytest.approx(1.833, abs=0.001)


def test_boltzmann_average_of_cone_angles(tmp_path):
	atoms, coords = read_xyz(RH)
	# two "conformers": the structure and a copy with the methyl hydrogens of one ring carbon pushed outwards
	other = coords.copy()
	for i in (6, 7, 8):  # H on C6 (0-based indices)
		other[i] += 0.4 * (other[i] - coords[5]) / np.linalg.norm(other[i] - coords[5])
	frames = tmp_path / "rh_frames.xyz"
	text = ""
	for name, xyz, energy in (("a", coords, 0.0), ("b", other, 1.0)):
		text += f"{len(atoms)}\n{name} E = {energy:.1f}\n" + "\n".join(f"{a} {x:.6f} {y:.6f} {z:.6f}" for a, (x, y, z) in zip(atoms, xyz)) + "\n"
	frames.write_text(text)
	out = tmp_path / "rh.csv"
	cmd = [sys.executable, "-m", "dbstep", str(frames), "--cone", "--atom1", "1", "--atom2", "3,4,5,24,25", "--boltzmann", "--csv", str(out)]
	result = subprocess.run(cmd, capture_output=True, text=True)
	assert result.returncode == 0, result.stdout + result.stderr
	with open(out, newline="") as f:
		rows = list(csv.DictReader(f))
	assert [row["structure"] for row in rows] == ["a", "b", "boltzmann"]
	a, b, avg = (float(row["cone_angle"]) for row in rows)
	pa, pb = float(rows[0]["population"]), float(rows[1]["population"])
	assert abs(b - a) > 0.1 and avg == pytest.approx(pa * a + pb * b, abs=1e-6)
	assert "boltzmann" in result.stdout and re.search(r"boltzmann\s+1\s+3,4,5,24,25\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+1\.83", result.stdout)
