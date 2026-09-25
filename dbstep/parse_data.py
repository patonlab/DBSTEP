# -*- coding: UTF-8 -*-
import sys
import numpy as np
import cclib
from abc import ABC, abstractmethod
from dbstep.constants import BOHR_TO_ANG, periodic_table


"""
parse_data

Parses data from files
Currently supporting:
	.xyz (single and multi-structure), .com/.gjf Gaussian inputs
	.sdf/.mol V2000 (single and multi-structure)
	.pdb/.ent Protein Data Bank files (single and multi-MODEL), with per-atom residue metadata
	.cube Gaussian volumetric files
	all filetypes parsed by the cclib python package (see https://cclib.github.io/)
	RDKit mol objects (Python API)
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


def get_pdb_models(file_path):
	"""Count and locate MODEL records in a (multi-model) PDB file.

	Args:
		file_path (str): path to PDB file

	Returns:
		list of (name, start_line, end_line) tuples, one per MODEL; a single entry covering the
		whole file when it has no MODEL records
	"""
	with open(file_path) as f:
		lines = f.readlines()
	return _parse_pdb_boundaries(lines)


def _parse_pdb_boundaries(file_lines):
	"""Identify MODEL/ENDMDL blocks in a PDB file. Files without MODEL records are one structure."""
	structures = []
	start = None
	for i, line in enumerate(file_lines):
		record = line[:6].strip()
		if record == "MODEL":
			start = i
			name = "model" + line[6:].strip()
		elif record == "ENDMDL" and start is not None:
			structures.append((name, start, i))
			start = None
	if start is not None:  # unterminated final MODEL
		structures.append((name, start, len(file_lines)))
	if not structures:
		structures.append(("", 0, len(file_lines)))
	return structures


def _pdb_element(name_field, element_field=""):
	"""Element symbol for a PDB atom from columns 77-78, falling back to the atom name (columns 13-16).

	PDB atom names are right-justified so that a two-letter element occupies columns 13-14
	("FE  ", "CA  " calcium) while a one-letter element starts in column 14 (" CA " alpha carbon,
	" HG " gamma hydrogen). Four-character hydrogen names may start with a digit ("1HB ") or
	with H ("HD21").
	"""
	# note: periodic_table[0] is "" so membership tests must exclude the empty string
	element = element_field.strip().capitalize()
	if element and element in periodic_table:
		return element
	name = name_field.ljust(4)
	if name[0].isalpha():
		candidate = name[0:2].strip().capitalize()
		if candidate and candidate in periodic_table:
			return candidate
	for char in name:
		if char.isalpha():
			return char.upper()
	return name.strip()


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
		elif ext in [".pdb", ".ent"]:
			mol = PDBParser(molecule, ext[1:], options.noH, options.exclude, options.spec_atom_1, options.spec_atom_2, structure=structure)
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
		"""Remove requested atoms - hydrogens or manually specified atoms.

		A specified atom (atom1/atom2) that would be removed is kept as a zero-radius ghost ("Bq")
		so that translation and alignment still work; the remaining atoms are renumbered.
		"""
		n_atoms = len(self.ATOMTYPES)
		atoms_to_remove = np.zeros(n_atoms, dtype=bool)
		if self.noH:
			atoms_to_remove |= self.ATOMTYPES == "H"
		if self.exclude:
			indices = self.exclude.split(",") if isinstance(self.exclude, str) else self.exclude
			atoms_to_remove[[int(atom) - 1 for atom in indices]] = True

		spec_atoms = [self.spec_atom_1 - 1] + [atom - 1 for atom in self.spec_atom_2]

		# if removed atom is one of the spec atoms, replace its atom type with Bq (radii=0)
		self.ATOMTYPES = np.array(["Bq" if i in spec_atoms and atoms_to_remove[i] else self.ATOMTYPES[i] for i in range(n_atoms)])
		atoms_to_remove[spec_atoms] = False

		removed_before = np.cumsum(atoms_to_remove) - atoms_to_remove
		self.spec_atom_1 = int(self.spec_atom_1 - removed_before[self.spec_atom_1 - 1])
		self.spec_atom_2 = [int(atom - removed_before[atom - 1]) for atom in self.spec_atom_2]
		self.keep(~atoms_to_remove)

	def keep(self, mask):
		"""Keep only the atoms where `mask` is True, in all per-atom arrays (ATOMTYPES, CARTESIANS and any METADATA)."""
		mask = np.asarray(mask, dtype=bool)
		self.ATOMTYPES = self.ATOMTYPES[mask]
		self.CARTESIANS = self.CARTESIANS[mask]
		for key, values in getattr(self, "METADATA", {}).items():
			self.METADATA[key] = np.asarray(values)[mask]

	@staticmethod
	def get_file_lines(file):
		""" "Reads file and returns the lines using readlines()

		Args:
		file (str): the path to the file

		Returns:
			list with lines of the file
		"""
		with open(file) as f:
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
		# orbital cube files flag themselves with a negative atom count and carry one extra header line after the atom block
		mo_header_line = None

		# first two lines skipped as they do not have useful information for this program
		for i in range(2, len(file_lines)):
			try:
				curr_line = file_lines[i]
				coord = [float(c) for c in curr_line.split()]
				if i == 2:
					self.num_atoms = int(abs(coord[0]))
					if coord[0] < 0:
						mo_header_line = start_of_atoms + self.num_atoms
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
				elif i == mo_header_line:
					continue
				else:
					self._parse_density_line(coord, curr_line)
			except ValueError:
				# TODO: make a custom cube file exception to chain ValueError with this error
				# TODO: handle potentially invalid atom errors
				sys.exit(f'  Unable to parse "{self._input}", a value on line {i + 1} could not be read in.')

		n_expected = self.xdim * self.ydim * self.zdim
		if len(self.DENSITY) != n_expected:
			sys.exit(f'  Unable to parse "{self._input}": expected {n_expected} volumetric values but read {len(self.DENSITY)}. Only single-dataset cube files are supported.')

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

		# Parse the counts line (4th line of the record) for number of atoms.
		# V2000 counts fields are fixed-width (3 chars), so >99 atoms and bonds run together ("100100  0 ...")
		counts_line = file_lines[start + 3]
		try:
			n_atoms = int(counts_line[0:3])
		except ValueError:
			n_atoms = int(counts_line.split()[0])

		# Atom block starts at line start+4
		atom_start = start + 4
		for i in range(atom_start, atom_start + n_atoms):
			parts = file_lines[i].split()
			x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
			atom_type = parts[3]
			self.ATOMTYPES.append(atom_type)
			self.CARTESIANS.append([x, y, z])


class PDBParser(DataParser):
	"""Read Cartesians and per-atom residue metadata from a PDB file (ATOM/HETATM records, fixed columns).

	Attributes:
		METADATA (dict of numpy arrays): record ("ATOM"/"HETATM"), name, resname, chain, resseq (int),
			icode, element, and resid ("A:45", "A:45A" with an insertion code), all aligned with ATOMTYPES
		structure_name (str or None): "modelN" for multi-MODEL files, otherwise None
		n_altloc_dropped (int): atoms skipped because they carry an alternate location other than "A"
	"""

	def __init__(self, file, input_format, noH, exclude, spec_atom_1, spec_atom_2, structure=None):
		self._structure = structure
		self.structure_name = None
		self.METADATA = {}
		self.n_altloc_dropped = 0
		super().__init__(file, input_format, noH, exclude, spec_atom_1, spec_atom_2, manual_file_lines=True)

	def parse_input(self):
		"""Parses ATOM/HETATM records of the selected MODEL (default: the first)."""
		file_lines = self.file_lines
		structures = _parse_pdb_boundaries(file_lines)
		idx = self._structure if self._structure is not None else 0
		if idx >= len(structures):
			sys.exit(f"  Structure index {idx} out of range (file has {len(structures)} models)")
		name, start, end = structures[idx]
		if len(structures) > 1:
			self.structure_name = name

		meta = {key: [] for key in ("record", "name", "resname", "chain", "resseq", "icode", "element")}
		for line_number in range(start, end):
			line = file_lines[line_number]
			record = line[:6].strip()
			if record not in ("ATOM", "HETATM"):
				continue
			line = line.rstrip("\n").ljust(80)
			altloc = line[16]
			if altloc not in (" ", "A"):
				self.n_altloc_dropped += 1
				continue
			try:
				x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
				resseq = int(line[22:26])
			except ValueError:
				sys.exit(f'  Unable to parse "{self._input}", line {line_number + 1} is not a valid ATOM/HETATM record.')
			element = _pdb_element(line[12:16], line[76:78])
			self.ATOMTYPES.append(element)
			self.CARTESIANS.append([x, y, z])
			meta["record"].append(record)
			meta["name"].append(line[12:16].strip())
			meta["resname"].append(line[17:20].strip())
			meta["chain"].append(line[21].strip())
			meta["resseq"].append(resseq)
			meta["icode"].append(line[26].strip())
			meta["element"].append(element)

		if not self.ATOMTYPES:
			sys.exit(f"  No ATOM/HETATM records found in {self._input}")
		self.METADATA = {key: np.array(values) for key, values in meta.items()}
		self.METADATA["resid"] = np.array([f"{chain}:{resseq}{icode}" for chain, resseq, icode in zip(meta["chain"], meta["resseq"], meta["icode"])])


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
