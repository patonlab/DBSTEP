import pytest

from dbstep import parse_data, Dbstep


def get_options(noH=False):
	"""Makes a mini options object."""
	return Dbstep.set_options({'noH': noH, 'atom1': 1, 'atom2': [2]})


@pytest.mark.parametrize("molecule, ext, expected_len, options", [
		("dbstep/examples/Et.xyz", ".xyz", 8, get_options()),
		("dbstep/examples/CH2CMe3_100.cube", ".cube", 17, get_options()),
		("dbstep/examples/Et.xyz", ".xyz", 2, get_options(True)),
		("dbstep/examples/CH2CMe3_100.cube", ".cube", 17, get_options(True))
	])
def test_read_input_len(molecule, ext, expected_len, options):
	parser = parse_data.read_input(molecule, ext, options)
	assert len(parser.ATOMTYPES) == len(parser.CARTESIANS) == expected_len
