# -*- coding: UTF-8 -*-
import os, sys
import numpy as np
import cclib
from abc import ABC, abstractmethod
from dbstep.constants import BOHR_TO_ANG, periodic_table


"""
parse_data

Parses data from files
Currently supporting: 
	.cube Gaussian volumetric files 
	all filetypes parsed by the cclib python package (see https://cclib.github.io/)
"""


def get_xyz_structures(file_path):
	"""Count and locate structures in a multi-XYZ file.

	Args:
		file_path (str): path to XYZ file

	Returns:
		list of (comment, atom_start_line, n_atoms) tuples
	"""
	with open(file_path) as f:
		lines = f.readlines()
	return _parse_xyz_boundaries(lines)


def _parse_xyz_boundaries(file_lines):
	"""Identify structure boundaries in a (multi-)XYZ file.

	Returns:
		list of (comment, atom_start_line, n_atoms) tuples.
		Empty list if no standard XYZ headers found.
	"""
	structures = []
	i = 0
	while i < len(file_lines):
		line = file_lines[i].strip()
		if not line:
			i += 1
			continue
		parts = line.split()
		if len(parts) == 1:
			try:
				n_atoms = int(parts[0])
			except ValueError:
				i += 1
				continue
			if n_atoms <= 0:
				i += 1
				continue
			comment = file_lines[i + 1].strip() if i + 1 < len(file_lines) else ""
			structures.append((comment, i + 2, n_atoms))
			i = i + 2 + n_atoms
		else:
			i += 1
	return structures


def get_sdf_structures(file_path):
	"""Count and locate structures in a multi-SDF file.

	Args:
		file_path (str): path to SDF file

	Returns:
		list of (name, record_start_line, record_end_line) tuples
	"""
	with open(file_path) as f:
		lines = f.readlines()
	return _parse_sdf_boundaries(lines)


def _parse_sdf_boundaries(file_lines):
	"""Identify structure boundaries in a (multi-)SDF file.

	Each record ends with a $$$$ delimiter line.

	Returns:
		list of (name, record_start_line, record_end_line) tuples.
	"""
	structures = []
	record_start = 0
	for i, line in enumerate(file_lines):
		if line.strip() == "$$$$":
			name = file_lines[record_start].strip() if record_start < len(file_lines) else ""
			structures.append((name, record_start, i))
			record_start = i + 1
	# Handle file without trailing $$$$
	if record_start < len(file_lines) and any(line.strip() for line in file_lines[record_start:]):
		name = file_lines[record_start].strip()
		structures.append((name, record_start, len(file_lines)))
	return structures


def read_input(molecule, ext, options):
	"""Chooses a Parser based on input molecule format.

	Args:
		molecule (str or mol object): path to file if molecule represented as one, or RDKit mol object
		ext (str): file extension used
		options (dict): options for DBSTEP program

	Returns:
		DataParser object with parsed molecule data to be used by the rest of the program
	"""
	if ext == ".cube":
		mol = CubeParser(molecule, "cube")
	else:
		structure = getattr(options, 'structure', None)
		if ext in [".xyz", ".com", ".gjf"]:
			mol = XYZParser(molecule, ext[1:], options.noH, options.exclude, options.spec_atom_1, options.spec_atom_2, structure=structure)
		elif ext in [".sdf", ".mol"]:
			mol = SDFParser(molecule, ext[1:], options.noH, options.exclude, options.spec_atom_1, options.spec_atom_2, structure=structure)
		elif ext == "rdkit":
			mol = RDKitParser(molecule, options.noH, options.exclude, options.spec_atom_1, options.spec_atom_2)
		else:
			mol = cclibParser(molecule, ext[1:], options.noH, options.exclude, options.spec_atom_1, options.spec_atom_2)
		if options.noH or options.exclude:
			options.spec_atom_1 = mol.spec_atom_1
			options.spec_atom_2 = mol.spec_atom_2
	return mol


class DataParser(ABC):
	"""Abstract base class made to be inherited by parsers for different molecule formats.

	Attributes:
		_input (str or RDKit mol object): the input molecule
		FORMAT (str): format of the input molecule
		ATOMTYPES (numpy array of char): the atoms in the molecule, starts as a list
		CARTESIANS (numpy array of tuples): xyz coordinates for each atom in the molecule, starts as a list
		noH (bool): true if hydrogens should be removed false otherwise.
		exclude (str): atoms to exclude from steric measurements (1-indexed)
		spec_atom_1 (int): specifies atom1
		spec_atom_2 (list of int): specifies atom2(s)
		file_lines (list of str, optional): each line of the file
	"""

	def __init__(self, _input, input_format, noH=False, exclude=False, spec_atom_1=None, spec_atom_2=None, manual_file_lines=False):
		"""Initializes basic member variables and lays out the ordering of method calls.

		Args:
			_input (str or RDKit mol object): the input molecule
			input_format (str): input_format of the input molecule
			noH (bool, optional): boolean which specifies whether hydrogens should be removed
			exclude (str, optional): string listing atom indices to remove. 1-indexed.
			spec_atom_1 (int, optional): specifies atom1
			spec_atom_2 (list of int, optional): contains atom2(s)
			manual_file_lines (bool, optional): to parse _input line by line manually using get_file_lines or not
		"""
		self._input, self.FORMAT = _input, input_format
		self.ATOMTYPES, self.CARTESIANS = [], []
		self.noH = noH
		self.exclude = exclude
		self.spec_atom_1, self.spec_atom_2 = spec_atom_1, spec_atom_2
		if manual_file_lines:
			self.file_lines = DataParser.get_file_lines(_input)
		self.parse_input()
		self.ATOMTYPES, self.CARTESIANS = np.array(self.ATOMTYPES), np.array(self.CARTESIANS)
		if (self.noH or self.exclude) and self.FORMAT != "cube":
			self.exclude_atoms()

	@abstractmethod
	def parse_input(self):
		"""Parse the input, filling ATOMTYPES with the atoms of the input molecule and CARTESIANS with the atoms xyz coordinates."""
		pass

	def exclude_atoms(self):
		"""Remove requested atoms - hydrogens or manually specified atoms"""
		atoms_to_remove = [False for i in range(len(self.ATOMTYPES))]
		if self.noH:
			atoms_to_remove = [True if self.ATOMTYPES[i] == "H" else atoms_to_remove[i] for i in range(len(atoms_to_remove))]
		if self.exclude:
			del_atom_list = [int(atom) - 1 for atom in self.exclude.split(",")]
			atoms_to_remove = [True if i in del_atom_list else atoms_to_remove[i] for i in range(len(atoms_to_remove))]

		spec_atoms = [self.spec_atom_1 - 1]
		[spec_atoms.append(atom - 1) for atom in self.spec_atom_2]

		# if removed atom is one of the spec atoms, replace its atom type with Bq (radii=0)
		self.ATOMTYPES = np.array(["Bq" if i in spec_atoms and atoms_to_remove[i] else self.ATOMTYPES[i] for i in range(len(atoms_to_remove))])
		atoms_to_remove = [False if i in spec_atoms and atoms_to_remove[i] else atoms_to_remove[i] for i in range(len(atoms_to_remove))]

		spec_atoms = [spec_atom - np.count_nonzero(atoms_to_remove[:spec_atom]) for spec_atom in spec_atoms]
		self.spec_atom_1 = spec_atoms[0] + 1
		self.spec_atom_2 = [atom + 1 for atom in spec_atoms[1:]]
		self.ATOMTYPES = self.ATOMTYPES[np.invert(atoms_to_remove)]
		self.CARTESIANS = self.CARTESIANS[np.invert(atoms_to_remove)]

	@staticmethod
	def get_file_lines(file):
		""" "Reads file and returns the lines using readlines()

		Args:
		file (str): the path to the file

		Returns:
			list with lines of the file
		"""
		with open(file, "r") as f:
			return f.readlines()


class CubeParser(DataParser):
	"""Read data from cube file, obtian XYZ Cartesians, dimensions, and volumetric data."""

	def __init__(self, file, input_format):
		super().__init__(file, input_format, manual_file_lines=True)
		self.INCREMENTS = np.asarray([self.x_inc, self.y_inc, self.z_inc])
		self.DENSITY = np.asarray(self.DENSITY)
		self.DATA = np.reshape(self.DENSITY, (self.xdim, self.ydim, self.zdim))

	def parse_input(self):
		"""Parses input from a cube file.

		http://paulbourke.net/dataformats/cube/ was used to determine general format of a cube file.

		"""
		self.num_atoms = None
		self.ATOMNUM, self.DENSITY, self.DENSITY_LINE = [], [], []
		file_lines = self.file_lines
		start_of_atoms = 6

		# first two lines skipped as they do not have useful information for this program
		for i in range(2, len(file_lines)):
			try:
				curr_line = file_lines[i]
				coord = [float(c) for c in curr_line.split()]
				if i == 2:
					self.num_atoms = coord[0]
					self.ORIGIN = [coord[1] * BOHR_TO_ANG, coord[2] * BOHR_TO_ANG, coord[3] * BOHR_TO_ANG]
				elif i == 3:
					self.xdim = int(coord[0])
					self.SPACING = coord[1] * BOHR_TO_ANG
					self.x_inc = [coord[1] * BOHR_TO_ANG, coord[2] * BOHR_TO_ANG, coord[3] * BOHR_TO_ANG]
				elif i == 4:
					self.ydim = int(coord[0])
					self.y_inc = [coord[1] * BOHR_TO_ANG, coord[2] * BOHR_TO_ANG, coord[3] * BOHR_TO_ANG]
				elif i == 5:
					self.zdim = int(coord[0])
					self.z_inc = [coord[1] * BOHR_TO_ANG, coord[2] * BOHR_TO_ANG, coord[3] * BOHR_TO_ANG]
				elif self.num_atoms and start_of_atoms <= i < start_of_atoms + self.num_atoms:
					self._parse_atom_line(coord)
				else:
					self._parse_density_line(coord, curr_line)
			except ValueError as e:
				# TODO: make a custom cube file exception to chain ValueError with this error
				# TODO: handle potentially invalid atom errors
				sys.exit(f'  Unable to parse "{self._input}", a value on line {i + 1} could not be read in.')

	def _parse_atom_line(self, split_line):
		"""Parses a line in the cube file containing atom number and coordinates."""
		atom_num = int(split_line[0])
		atom = periodic_table[atom_num]
		x, y, z = float(split_line[2]) * BOHR_TO_ANG, float(split_line[3]) * BOHR_TO_ANG, float(split_line[4]) * BOHR_TO_ANG
		self.ATOMNUM.append(atom_num)
		self.ATOMTYPES.append(atom)
		self.CARTESIANS.append([x, y, z])

	def _parse_density_line(self, split_line, curr_line):
		"""Appends density values from a line in the cube file to the DENSITY member array."""
		for val in split_line:
			self.DENSITY.append(float(val))
		self.DENSITY_LINE.append(curr_line)


class XYZParser(DataParser):
	"""Read XYZ Cartesians from an xyz file or chem files similar to xyz."""

	def __init__(self, file, input_format, noH, exclude, spec_atom_1, spec_atom_2, structure=None):
		self._structure = structure
		self.structure_name = None
		super().__init__(file, input_format, noH, exclude, spec_atom_1, spec_atom_2, manual_file_lines=True)

	def parse_input(self):
		"""Parses input from either xyz file or com/gif file."""
		file_lines = self.file_lines
		if self.FORMAT == "xyz":
			structures = _parse_xyz_boundaries(file_lines)
			if structures:
				idx = self._structure if self._structure is not None else 0
				if idx >= len(structures):
					sys.exit(f"  Structure index {idx} out of range (file has {len(structures)} structures)")
				comment, atom_start, n_atoms = structures[idx]
				self.structure_name = comment
				self._parse_atom_lines(file_lines, atom_start, atom_start + n_atoms)
			else:
				# Fallback for non-standard XYZ files: parse all lines
				self._parse_atom_lines(file_lines, 0, len(file_lines))
		elif self.FORMAT == "com" or self.FORMAT == "gjf":
			for i in range(0, len(file_lines)):
				if file_lines[i].find("#") > -1:
					if len(file_lines[i + 1].split()) == 0:
						start = i + 5
					if len(file_lines[i + 2].split()) == 0:
						start = i + 6
					break
			self._parse_atom_lines(file_lines, start, len(file_lines))

	def _parse_atom_lines(self, file_lines, start, end):
		"""Parse atom coordinate lines from file_lines[start:end]."""
		for i in range(start, min(end, len(file_lines))):
			try:
				coord = file_lines[i].split()
				for j in range(len(coord)):
					try:
						coord[j] = float(coord[j])
					except ValueError:
						pass
				if len(coord) >= 4:
					if isinstance(coord[1], float) and isinstance(coord[2], float) and isinstance(coord[3], float):
						self.ATOMTYPES.append(coord[0])
						self.CARTESIANS.append([coord[1], coord[2], coord[3]])
			except Exception:
				pass


class SDFParser(DataParser):
	"""Read XYZ Cartesians from an SDF/MOL file, with multi-structure support."""

	def __init__(self, file, input_format, noH, exclude, spec_atom_1, spec_atom_2, structure=None):
		self._structure = structure
		self.structure_name = None
		super().__init__(file, input_format, noH, exclude, spec_atom_1, spec_atom_2, manual_file_lines=True)

	def parse_input(self):
		"""Parses input from an SDF file using structure boundaries."""
		file_lines = self.file_lines
		structures = _parse_sdf_boundaries(file_lines)
		if not structures:
			sys.exit(f"  Unable to parse any structures from {self._input}")
		idx = self._structure if self._structure is not None else 0
		if idx >= len(structures):
			sys.exit(f"  Structure index {idx} out of range (file has {len(structures)} structures)")
		name, start, end = structures[idx]
		self.structure_name = name

		# Parse the counts line (4th line of the record) for number of atoms
		counts_line = file_lines[start + 3]
		n_atoms = int(counts_line.split()[0])

		# Atom block starts at line start+4
		atom_start = start + 4
		for i in range(atom_start, atom_start + n_atoms):
			parts = file_lines[i].split()
			x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
			atom_type = parts[3]
			self.ATOMTYPES.append(atom_type)
			self.CARTESIANS.append([x, y, z])


class cclibParser(DataParser):
	"""Use the cclib package to extract data from generic computational chemistry output files."""

	def __init__(self, file, input_format, noH, exclude, spec_atom_1, spec_atom_2):
		super().__init__(file, input_format, noH, exclude, spec_atom_1, spec_atom_2)

	def parse_input(self):
		"""Parses input file uses cclib file parser."""
		cclib_parsed = cclib.io.ccread(self._input)
		self.CARTESIANS = np.array(cclib_parsed.atomcoords[-1])
		for i in cclib_parsed.atomnos:
			self.ATOMTYPES.append(periodic_table[i])


class RDKitParser(DataParser):
	"""Extract coordinates and atom types from rdkit mol object

	Attributes:
		ATOMTYPES (numpy array): List of elements present in molecular file
		CARTESIANS (numpy array): List of Cartesian (x,y,z) coordinates for each atom
	"""

	def __init__(self, mol, noH, exclude, spec_atom_1, spec_atom_2):
		super().__init__(mol, "RDKit", noH, exclude, spec_atom_1, spec_atom_2)

	def parse_input(self):
		"""Store cartesians and symbols from mol object"""
		try:
			self.ATOMTYPES, self.CARTESIANS = [], []
			for i in range(self._input.GetNumAtoms()):
				self.ATOMTYPES.append(self._input.GetAtoms()[i].GetSymbol())
				pos = self._input.GetConformer().GetAtomPosition(i)
				self.CARTESIANS.append([pos.x, pos.y, pos.z])
		except ValueError:
			self.ATOMTYPES, self.CARTESIANS = [], []
			print("Mol object does not have 3D coordinates!")
