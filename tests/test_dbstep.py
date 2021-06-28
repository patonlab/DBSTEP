import pytest

from dbstep import Dbstep


class DbstepShell:
	"""A reference to the dbstep class so its helper methods can be accessed without running its __init__"""
	def __init__(self):
		self.__class__ = Dbstep.dbstep


class TestDbstep:
	"""Class which contains tests for the dbstep class."""

	@pytest.mark.parametrize("input_atom_1, input_atom_2, expected_atom_1, expected_atom_2", [
		(None, None, 1, [2]),
		(1, "4,6,1,4", 1, [4, 6, 1, 4]),
		(3, [3, 4, "5"], 3, [3, 4, 5])
	])
	def test_get_spec_atoms(self, input_atom_1, input_atom_2, expected_atom_1, expected_atom_2):
		options = Dbstep.set_options({'atom1': input_atom_1, 'atom2': input_atom_2})
		db = DbstepShell()
		db._get_spec_atoms(options)
		assert options.spec_atom_1 == expected_atom_1
		assert options.spec_atom_2 == expected_atom_2

	@pytest.mark.parametrize("input_atom_1, input_atom_2, expected_exception", [
		(1, "aa", ValueError),
		("aa", [1], ValueError),
		(3, [1, 3, "4a"], ValueError),
		(3, [3, 4, [1, 2, 3]], TypeError),
	])
	def test_get_spec_atoms_exception(self, input_atom_1, input_atom_2, expected_exception):
		options = Dbstep.set_options({'atom1': input_atom_1, 'atom2': input_atom_2})
		db = DbstepShell()
		with pytest.raises(expected_exception):
			db._get_spec_atoms(options)
