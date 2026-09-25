"""Python API helpers that do not fit elsewhere."""

import pytest

from dbstep import Dbstep


def test_from_rdkit_matches_file_input():
	Chem = pytest.importorskip("rdkit.Chem")
	from rdkit.Chem import AllChem

	mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
	AllChem.EmbedMolecule(mol, randomSeed=1)
	result = Dbstep.from_rdkit(mol, atom1=1, atom2=2, sterimol=True, volume=True, quiet=True)
	assert result.L > 2.5 and 0 < result.bur_vol < 100
	assert result.results[0]["file"] in ("", "rdkit_mol") or isinstance(result.results[0]["file"], str)


def test_from_rdkit_rejects_molecules_without_conformer():
	Chem = pytest.importorskip("rdkit.Chem")
	with pytest.raises(SystemExit, match="no 3D conformer"):
		Dbstep.from_rdkit(Chem.MolFromSmiles("CC"), atom1=1, quiet=True)


def test_from_rdkit_rejects_file_names():
	with pytest.raises(SystemExit, match="expects an RDKit Mol"):
		Dbstep.from_rdkit("dbstep/data/Et.xyz", atom1=1, quiet=True)
