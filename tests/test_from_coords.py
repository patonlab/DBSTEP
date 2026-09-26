"""dbstep on in-memory structures (from_coords / Structure) and the alignment transform."""

import numpy as np
import pytest

from dbstep import Dbstep, calculator, parse_data


def read_xyz(path):
	lines = open(path).read().splitlines()
	atoms = [line.split()[0] for line in lines[2:] if line.strip()]
	coords = np.array([[float(v) for v in line.split()[1:4]] for line in lines[2:] if line.strip()])
	return atoms, coords


def test_from_coords_matches_file_run():
	atoms, coords = read_xyz("dbstep/data/Et.xyz")
	ref = Dbstep.dbstep("dbstep/data/Et.xyz", atom1=2, atom2=5, sterimol=True, volume=True, quiet=True)
	mem = Dbstep.from_coords(atoms, coords, name="ethane", atom1=2, atom2=5, sterimol=True, volume=True, quiet=True)
	assert (mem.L, mem.Bmin, mem.Bmax, mem.bur_vol) == pytest.approx((ref.L, ref.Bmin, ref.Bmax, ref.bur_vol))
	assert mem.results[0]["file"] == "ethane" and mem.results[0]["path"] == ""
	assert mem.n_atoms_total == len(atoms)


def test_structure_with_metadata_supports_decompose():
	mol = parse_data.PDBParser("tests/pdb_files/ala5.pdb", "pdb", False, False, 1, [1])
	structure = parse_data.Structure(mol.ATOMTYPES, mol.CARTESIANS, "ala5", mol.METADATA)
	ca3 = int(np.flatnonzero((mol.METADATA["resid"] == "A:3") & (mol.METADATA["name"] == "CA"))[0]) + 1
	mem = Dbstep.dbstep(structure, atom1=ca3, volume=True, decompose=True, cutoff="auto", quiet=True)
	ref = Dbstep.dbstep("tests/pdb_files/ala5.pdb", residue="A:3", volume=True, decompose=True, quiet=True)
	assert mem.bur_vol == pytest.approx(ref.bur_vol, abs=1e-9)
	assert mem.contributions == pytest.approx(ref.contributions, abs=1e-9)


def test_structure_validation_and_ghosts():
	with pytest.raises(ValueError):
		parse_data.Structure(["C", "H"], [[0, 0, 0]])
	# a ghost atom1 occupies nothing but still defines the origin
	mol = Dbstep.from_coords(["Bq", "C", "C"], [[0, 0, 0], [0, 0, 1.5], [0, 0, 3.0]], atom1=1, atom2=2, sterimol=True, quiet=True)
	assert mol.L == pytest.approx(3.0 + 1.70, abs=1e-6)


def test_rigid_transform_recovers_the_alignment():
	atoms, coords = read_xyz("tests/metal_files/RhCpMe5Cl2PMe3.xyz")
	run = Dbstep.from_coords(atoms, coords, atom1=17, atom2=[18, 20, 37], atom3=18, sterimol=True, quiet=True)
	rotation, translation = calculator.rigid_transform(run.coords, run.aligned_coords)
	assert np.allclose(run.coords @ rotation.T + translation, run.aligned_coords, atol=1e-8)
	assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-10) and np.linalg.det(rotation) == pytest.approx(1.0)
	# the aligned origin is atom1 and the aligned z axis points from atom1 towards atom2
	back = (np.array([[0, 0, 0], [0, 0, 1.0]]) - translation) @ rotation
	assert np.allclose(back[0], coords[16], atol=1e-8)
	direction = coords[[17, 19, 36]].mean(axis=0) - coords[16]
	assert np.dot(back[1] - back[0], direction / np.linalg.norm(direction)) == pytest.approx(1.0, abs=1e-8)
	vectors = run.sterimol_vectors()
	assert set(vectors) == {"bmax", "bmin"}
	assert np.hypot(*vectors["bmax"][:2]) == pytest.approx(run.Bmax, abs=2e-3) and np.hypot(*vectors["bmin"][:2]) == pytest.approx(run.Bmin, abs=2e-3)
