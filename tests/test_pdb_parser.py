"""PDB input: fixed-column parsing, element inference, residue metadata, MODEL blocks."""

import numpy as np
import pytest

from dbstep import Dbstep, parse_data

pdb_dir = "tests/pdb_files/"


def get_options(noH=False, exclude=False, atom1=1, atom2=2, structure=None):
	return Dbstep.set_options({"noH": noH, "exclude": exclude, "atom1": atom1, "atom2": [atom2], "structure": structure})


def parse(path, **kwargs):
	return parse_data.read_input(path, ".pdb", get_options(**kwargs))


# --- element inference --------------------------------------------------------------------------


@pytest.mark.parametrize(
	"name_field, element_field, expected",
	[
		(" CA ", " C", "C"),  # element column wins when present
		(" CA ", "", "C"),  # alpha carbon: one-letter element starts in column 14
		("CA  ", "", "Ca"),  # calcium: two-letter element starts in column 13
		(" HG ", "", "H"),  # gamma hydrogen
		("HG  ", "", "Hg"),  # mercury
		("FE  ", "", "Fe"),
		("SE  ", "", "Se"),
		("NA  ", "", "Na"),
		("CL  ", "", "Cl"),
		(" N  ", "", "N"),
		(" OXT", "", "O"),
		("1HB ", "", "H"),  # hydrogen names may start with a digit
		("HD21", "", "H"),  # ... or with H followed by non-element letters
		("HH12", "", "H"),
		(" CD ", "", "C"),  # delta carbon, not cadmium
		("SE  ", "SE", "Se"),  # upper-case element column is normalised
		(" ZN ", "ZN", "Zn"),
	],
)
def test_pdb_element_inference(name_field, element_field, expected):
	assert parse_data._pdb_element(name_field, element_field) == expected


# --- real structure: 1A8O (HIV capsid C-terminal domain) ------------------------------------------


def test_1a8o_atoms_and_metadata():
	mol = parse(pdb_dir + "1a8o.pdb")
	assert len(mol.ATOMTYPES) == len(mol.CARTESIANS) == 644
	meta = mol.METADATA
	assert all(len(values) == 644 for values in meta.values())
	assert set(meta["chain"]) == {"A"}
	assert np.count_nonzero(meta["resname"] == "HOH") == 88
	assert np.count_nonzero(meta["record"] == "HETATM") == 120
	# selenomethionine selenium atoms are HETATM with a two-letter element
	assert list(mol.ATOMTYPES[meta["element"] == "Se"]) == ["Se"] * 4
	assert set(meta["resname"][meta["element"] == "Se"]) == {"MSE"}
	assert meta["resid"][0] == "A:151"
	assert meta["name"][1] == "CA"
	assert mol.structure_name is None
	assert mol.n_altloc_dropped == 0


def test_1a8o_has_a_single_model():
	assert len(parse_data.get_pdb_models(pdb_dir + "1a8o.pdb")) == 1


def test_1a8o_runs_with_auto_cutoff():
	mol = Dbstep.dbstep(pdb_dir + "1a8o.pdb", atom1=20, volume=True, cutoff="auto", quiet=True)
	assert mol.n_atoms_total == 644
	assert mol.n_atoms_kept < 120
	assert 0 < mol.bur_vol < 100


# --- generated penta-alanine with hydrogens, waters and an ion -------------------------------------


def test_ala5_metadata():
	mol = parse(pdb_dir + "ala5.pdb")
	meta = mol.METADATA
	assert len(mol.ATOMTYPES) == 56
	assert np.count_nonzero(mol.ATOMTYPES == "H") == 27
	assert sorted(set(meta["resseq"][meta["resname"] == "ALA"])) == [1, 2, 3, 4, 5]
	assert list(meta["resname"][meta["record"] == "HETATM"]) == ["HOH", "HOH", "NA"]
	assert list(mol.ATOMTYPES[meta["record"] == "HETATM"]) == ["O", "O", "Na"]


def test_noH_keeps_metadata_aligned():
	mol = parse(pdb_dir + "ala5.pdb", noH=True)
	assert "H" not in mol.ATOMTYPES
	assert len(mol.ATOMTYPES) == 56 - 27
	assert all(len(values) == len(mol.ATOMTYPES) for values in mol.METADATA.values())
	assert "H" not in set(mol.METADATA["element"])
	assert list(mol.METADATA["name"][:5]) == ["N", "CA", "C", "O", "CB"]


def test_exclude_keeps_metadata_aligned_and_ghosts_spec_atoms():
	# exclude atom 2 (the CA that is also atom2): it must stay as a Bq ghost, atom 3 must go
	mol = parse(pdb_dir + "ala5.pdb", exclude="2,3", atom1=1, atom2=2)
	assert len(mol.ATOMTYPES) == 55
	assert mol.ATOMTYPES[1] == "Bq"
	assert mol.METADATA["name"][1] == "CA"
	assert mol.METADATA["name"][2] == "O"
	assert (mol.spec_atom_1, mol.spec_atom_2) == (1, [2])


def test_exclude_accepts_a_list():
	from_string = parse(pdb_dir + "ala5.pdb", exclude="5,6,7")
	from_list = parse(pdb_dir + "ala5.pdb", exclude=[5, 6, 7])
	assert list(from_string.ATOMTYPES) == list(from_list.ATOMTYPES)


def test_pdb_matches_same_atoms_from_xyz(tmp_path):
	"""Coordinates and elements from the PDB path give identical results to the same atoms as XYZ."""
	mol = parse(pdb_dir + "ala5.pdb")
	xyz = tmp_path / "ala5.xyz"
	with open(xyz, "w") as f:
		f.write(f"{len(mol.ATOMTYPES)}\nala5\n")
		for atom, (x, y, z) in zip(mol.ATOMTYPES, mol.CARTESIANS):
			f.write(f"{atom} {x:.3f} {y:.3f} {z:.3f}\n")
	ca3 = int(np.where((mol.METADATA["resseq"] == 3) & (mol.METADATA["name"] == "CA"))[0][0]) + 1
	cb3 = int(np.where((mol.METADATA["resseq"] == 3) & (mol.METADATA["name"] == "CB"))[0][0]) + 1
	from_pdb = Dbstep.dbstep(pdb_dir + "ala5.pdb", atom1=ca3, atom2=cb3, volume=True, sterimol=True, quiet=True)
	from_xyz = Dbstep.dbstep(str(xyz), atom1=ca3, atom2=cb3, volume=True, sterimol=True, quiet=True)
	assert (from_pdb.L, from_pdb.Bmin, from_pdb.Bmax) == pytest.approx((from_xyz.L, from_xyz.Bmin, from_xyz.Bmax))
	assert from_pdb.bur_vol == pytest.approx(from_xyz.bur_vol)
	assert from_pdb.occ_vol == pytest.approx(from_xyz.occ_vol)


# --- synthetic edge cases -----------------------------------------------------------------------


def write_pdb(path, records):
	path.write_text("\n".join(records) + "\n")


def atom_line(serial, name, resname, chain, resseq, x, y, z, record="ATOM", altloc=" ", icode=" ", element=""):
	return f"{record:<6}{serial:5d} {name:4}{altloc}{resname:>3} {chain}{resseq:4d}{icode}   {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}"


def test_altloc_policy(tmp_path):
	pdb = tmp_path / "altloc.pdb"
	write_pdb(pdb, [
		atom_line(1, " N  ", "SER", "A", 1, 0, 0, 0),
		atom_line(2, " OG ", "SER", "A", 1, 1, 0, 0, altloc="A"),
		atom_line(3, " OG ", "SER", "A", 1, 1.5, 0, 0, altloc="B"),
		atom_line(4, " C  ", "SER", "A", 1, 2, 0, 0),
	])
	mol = parse(str(pdb))
	assert list(mol.METADATA["name"]) == ["N", "OG", "C"]
	assert mol.CARTESIANS[1][0] == pytest.approx(1.0)
	assert mol.n_altloc_dropped == 1


def test_insertion_code_and_missing_element_column(tmp_path):
	pdb = tmp_path / "icode.pdb"
	write_pdb(pdb, [
		atom_line(1, " CA ", "GLY", "B", 52, 0, 0, 0),
		atom_line(2, " CA ", "GLY", "B", 52, 3, 0, 0, icode="A"),
		atom_line(3, "CA  ", " CA", "B", 300, 6, 0, 0, record="HETATM"),
	])
	mol = parse(str(pdb))
	assert list(mol.METADATA["resid"]) == ["B:52", "B:52A", "B:300"]
	assert list(mol.ATOMTYPES) == ["C", "C", "Ca"]


def test_multi_model_selection(tmp_path):
	base = parse(pdb_dir + "ala5.pdb")
	lines = open(pdb_dir + "ala5.pdb").read().splitlines()
	atom_lines = [line for line in lines if line.startswith(("ATOM", "HETATM"))]
	shifted = [line[:30] + f"{float(line[30:38]) + 10.0:8.3f}" + line[38:] for line in atom_lines]
	pdb = tmp_path / "models.pdb"
	write_pdb(pdb, ["MODEL        1", *atom_lines, "ENDMDL", "MODEL        2", *shifted, "ENDMDL", "END"])

	assert [name for name, _, _ in parse_data.get_pdb_models(str(pdb))] == ["model1", "model2"]
	first = parse(str(pdb), structure=0)
	second = parse(str(pdb), structure=1)
	assert first.structure_name == "model1" and second.structure_name == "model2"
	assert np.allclose(first.CARTESIANS, base.CARTESIANS)
	assert np.allclose(second.CARTESIANS[:, 0], base.CARTESIANS[:, 0] + 10.0)
	with pytest.raises(SystemExit):
		parse(str(pdb), structure=2)


def test_file_without_atoms_exits(tmp_path):
	pdb = tmp_path / "empty.pdb"
	write_pdb(pdb, ["HEADER    NOTHING", "END"])
	with pytest.raises(SystemExit):
		parse(str(pdb))
