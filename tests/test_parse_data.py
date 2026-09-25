import numpy as np
import pytest

from dbstep import parse_data, Dbstep


def get_options(noH=False, structure=None):
	"""Makes a mini options object."""
	return Dbstep.set_options({"noH": noH, "atom1": 1, "atom2": [2], "structure": structure})


cube_dir = "tests/cube_files/"
xyz_dir = "dbstep/data/"


@pytest.mark.parametrize(
	"molecule, ext, expected_len, options",
	[
		(xyz_dir + "Et.xyz", ".xyz", 8, get_options()),
		(cube_dir + "CH2CMe3_100.cube", ".cube", 17, get_options()),
		(xyz_dir + "Et.xyz", ".xyz", 3, get_options(True)),
		(cube_dir + "CH2CMe3_100.cube", ".cube", 17, get_options(True)),
		(xyz_dir + "all.sdf", ".sdf", 18, get_options()),
		(xyz_dir + "all.sdf", ".sdf", 11, get_options(True)),  # atom 1 is H but a spec atom, so it is kept as Bq
	],
)
def test_read_input_len(molecule, ext, expected_len, options):
	parser = parse_data.read_input(molecule, ext, options)
	assert len(parser.ATOMTYPES) == len(parser.CARTESIANS) == expected_len


def test_multi_structure_boundaries_agree_between_xyz_and_sdf():
	xyz_structures = parse_data.get_xyz_structures(xyz_dir + "all.xyz")
	sdf_structures = parse_data.get_sdf_structures(xyz_dir + "all.sdf")
	assert len(xyz_structures) == len(sdf_structures) > 1
	assert [s[0].split()[0] for s in xyz_structures] == [s[0] for s in sdf_structures]


def test_single_structure_xyz_has_one_boundary():
	assert len(parse_data.get_xyz_structures(xyz_dir + "Et.xyz")) == 1


def test_structure_index_selects_matching_structure():
	"""Structure i of all.xyz and all.sdf is the same molecule, and differs from structure 0."""
	structures = parse_data.get_xyz_structures(xyz_dir + "all.xyz")
	first = parse_data.read_input(xyz_dir + "all.xyz", ".xyz", get_options(structure=0))
	for idx, (comment, _, n_atoms) in enumerate(structures):
		from_xyz = parse_data.read_input(xyz_dir + "all.xyz", ".xyz", get_options(structure=idx))
		from_sdf = parse_data.read_input(xyz_dir + "all.sdf", ".sdf", get_options(structure=idx))
		assert from_xyz.structure_name == comment
		assert from_sdf.structure_name == comment.split()[0]
		assert len(from_xyz.ATOMTYPES) == len(from_sdf.ATOMTYPES) == n_atoms
		assert list(from_xyz.ATOMTYPES) == list(from_sdf.ATOMTYPES)
		assert np.allclose(from_xyz.CARTESIANS, from_sdf.CARTESIANS, atol=1e-4)
		if idx > 0:
			assert from_xyz.CARTESIANS.shape != first.CARTESIANS.shape or not np.allclose(from_xyz.CARTESIANS, first.CARTESIANS)


def test_structure_index_out_of_range_exits():
	with pytest.raises(SystemExit):
		parse_data.read_input(xyz_dir + "all.sdf", ".sdf", get_options(structure=999))


def test_sdf_counts_line_with_100_or_more_atoms(tmp_path):
	"""V2000 count fields are 3 characters wide, so 100 atoms/100 bonds reads '100100' with no separator."""
	n = 100
	lines = ["big", "  test", "", f"{n:3d}{n:3d}  0  0  0  0  0  0  0  0999 V2000"]
	lines += [f"{i * 0.1:10.4f}{0.0:10.4f}{0.0:10.4f} C   0  0  0  0  0  0  0  0  0  0  0  0" for i in range(n)]
	lines += [f"{i + 1:3d}{(i + 1) % n + 1:3d}  1  0  0  0  0" for i in range(n)]
	lines += ["M  END", "$$$$"]
	sdf = tmp_path / "big.sdf"
	sdf.write_text("\n".join(lines) + "\n")

	parser = parse_data.read_input(str(sdf), ".sdf", get_options())
	assert len(parser.ATOMTYPES) == n
	assert set(parser.ATOMTYPES) == {"C"}


def _write_cube(path, natoms_field, extra_header=None):
	"""Write a minimal 2x2x2 cube file with a single H atom at the origin."""
	lines = [
		" title",
		" comment",
		f"{natoms_field:5d} {-1.0:12.6f} {-1.0:12.6f} {-1.0:12.6f}",
		f"{2:5d} {1.0:12.6f} {0.0:12.6f} {0.0:12.6f}",
		f"{2:5d} {0.0:12.6f} {1.0:12.6f} {0.0:12.6f}",
		f"{2:5d} {0.0:12.6f} {0.0:12.6f} {1.0:12.6f}",
		f"{1:5d} {1.0:12.6f} {0.0:12.6f} {0.0:12.6f} {0.0:12.6f}",
	]
	if extra_header is not None:
		lines.append(extra_header)
	lines += ["  0.1 0.2 0.3 0.4", "  0.5 0.6 0.7 0.8"]
	path.write_text("\n".join(lines) + "\n")


def test_cube_parser_density_file(tmp_path):
	cube = tmp_path / "dens.cube"
	_write_cube(cube, 1)
	parser = parse_data.read_input(str(cube), ".cube", get_options())
	assert list(parser.ATOMTYPES) == ["H"]
	assert parser.DATA.shape == (2, 2, 2)
	assert np.isclose(parser.DENSITY.sum(), 3.6)


def test_cube_parser_orbital_file_with_negative_atom_count(tmp_path):
	"""Orbital cubes use a negative atom count and one extra header line listing the orbitals."""
	cube = tmp_path / "mo.cube"
	_write_cube(cube, -1, extra_header="    1    5")
	parser = parse_data.read_input(str(cube), ".cube", get_options())
	assert list(parser.ATOMTYPES) == ["H"]
	assert parser.DATA.shape == (2, 2, 2)
	assert np.isclose(parser.DENSITY.sum(), 3.6)


def test_cube_parser_rejects_wrong_number_of_values(tmp_path):
	cube = tmp_path / "bad.cube"
	_write_cube(cube, 1, extra_header="  9.0 9.0")
	with pytest.raises(SystemExit):
		parse_data.read_input(str(cube), ".cube", get_options())
