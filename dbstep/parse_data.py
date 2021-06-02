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


def element_id(massno, num=False):
	"""Return element id number"""
	try:
		if num:
			return periodic_table.index(massno)
		else: return periodic_table[massno]
	except IndexError:
		return "XX"


def read_input(molecule, ext, options):
	"""Chooses a Parser based on input molecule format.

	Args:
		molecule (str or mol object): path to file if molecule represented as one, or RDKit mol object
		ext (str): file extension used
		options (dict): options for DBSTEP program

	Returns:
		DataParser object with parsed molecule data to be used by the rest of the program
	"""
	if ext == '.cube':
		options.surface = 'density'
		mol = CubeParser(molecule, "cube")
	else:
		if ext in [".xyz", '.com', '.gjf']:
			mol = XYZParser(molecule, ext[1:], options.noH, options.spec_atom_1, options.spec_atom_2)
		elif ext == 'rdkit':
			mol = RDKitParser(molecule, options.noH, options.spec_atom_1, options.spec_atom_2)
		else:
			mol = cclibParser(molecule, ext[1:], options.noH, options.spec_atom_1, options.spec_atom_2)
		if options.noH:
			options.spec_atom_1 = mol.spec_atom_1
			options.spec_atom_2 = mol.spec_atom_2
	return mol


class DataParser(ABC):
	"""Abstract base class made to be inherited by parsers for different molecule formats.

	Attributes:
		FORMAT (str): format of the input molecule
		ATOMTYPES (numpy array of char): the atoms in the molecule, starts as a list
		CARTESIANS (numpy array of tuples): xyz coordinates for each atom in the molecule, starts as a list
		noH (bool): true if hydrogens should be removed false otherwise.
		spec_atom_1 (int): specifies atom1
		spec_atom_2 (list of int): specifies atom2(s)
	"""

	def __init__(self, input_format, noH=False, spec_atom_1=None, spec_atom_2=None):
		"""
		Basic member variable initialization

		Args:
			input_format (str): input_format of the input molecule
			noH (bool, optional): boolean which specifies whether
			spec_atom_1 (int, optional): specifies atom1
			spec_atom_2 (list of int, optional): contains atom2(s)
		"""
		self.FORMAT = input_format
		self.ATOMTYPES, self.CARTESIANS = [], []
		self.noH = noH
		self.spec_atom_1, self.spec_atom_2 = spec_atom_1, spec_atom_2

	@abstractmethod
	def parse_input(self, _input):
		"""Parse the input, filling ATOMTYPES with the atoms of the input molecule and CARTESIANS with the atoms xyz coordinates.
		"""
		pass

	def remove_hydrogens(self):
		"""Remove hydrogens if requested, update spec_atom numbering if necessary."""
		if self.noH:
			is_atom_type_h = self.ATOMTYPES == 'H'
			spec_atoms = [self.spec_atom_1] + self.spec_atom_2
			spec_atoms = [
				spec_atom - np.count_nonzero(is_atom_type_h[:spec_atom])
				for spec_atom in spec_atoms]
			self.spec_atom_1 = spec_atoms[0]
			self.spec_atom_2 = spec_atoms[1:]
			self.ATOMTYPES = self.ATOMTYPES[np.invert(is_atom_type_h)]
			self.CARTESIANS = self.CARTESIANS[np.invert(is_atom_type_h)]

	@staticmethod
	def get_file_lines(file):
		""""Reads file and returns the lines using readlines()

		Args:
		file (str): the path to the file

		Returns:
			list with lines of the file
		"""
		with open(file, 'r') as f:
			return f.readlines()


class CubeParser(DataParser):
	"""Read data from cube file, obtian XYZ Cartesians, dimensions, and volumetric data."""

	def __init__(self, file, input_format):
		super().__init__(input_format)
		self.parse_input(file)
		self.INCREMENTS = np.asarray([self.x_inc, self.y_inc, self.z_inc])
		self.DENSITY = np.asarray(self.DENSITY)
		self.DATA = np.reshape(self.DENSITY, (self.xdim, self.ydim, self.zdim))

	def parse_input(self, _input):
		"""Parses input from a cube file."""
		file_lines = DataParser.get_file_lines(_input)
		self.ATOMTYPES, self.ATOMNUM, self.CARTESIANS, self.DENSITY, self.DENSITY_LINE = [], [], [], [], []
		for i in range(2, len(file_lines)):
			try:
				coord = file_lines[i].split()
				for j in range(len(coord)):
					try:
						coord[j] = float(coord[j])
					except ValueError: pass
				if i == 2:
					self.ORIGIN = [coord[1]*BOHR_TO_ANG, coord[2]*BOHR_TO_ANG, coord[3]*BOHR_TO_ANG]
				elif i == 3:
					self.xdim = int(coord[0])
					self.SPACING = coord[1]*BOHR_TO_ANG
					self.x_inc = [coord[1]*BOHR_TO_ANG, coord[2]*BOHR_TO_ANG, coord[3]*BOHR_TO_ANG]
				elif i == 4:
					self.ydim = int(coord[0])
					self.y_inc = [coord[1]*BOHR_TO_ANG, coord[2]*BOHR_TO_ANG, coord[3]*BOHR_TO_ANG]
				elif i == 5:
					self.zdim = int(coord[0])
					self.z_inc = [coord[1]*BOHR_TO_ANG, coord[2]*BOHR_TO_ANG,coord[3]*BOHR_TO_ANG]
				elif len(coord) == 5:
					if coord[0] == int(coord[0]) and isinstance(coord[2], float) and isinstance(coord[3], float) and isinstance(coord[4], float):
						[atom, x,y,z] = [periodic_table[int(coord[0])], float(coord[2])*BOHR_TO_ANG, float(coord[3])*BOHR_TO_ANG, float(coord[4])*BOHR_TO_ANG]
						self.ATOMNUM.append(int(coord[0]))
						self.ATOMTYPES.append(atom)
						self.CARTESIANS.append([x,y,z])
				if coord[0] != int(coord[0]):
					for val in coord:
						self.DENSITY.append(val)
					self.DENSITY_LINE.append(file_lines[i])
			except: pass


class XYZParser(DataParser):
	"""Read XYZ Cartesians from an xyz file or chem files similar to xyz."""

	def __init__(self, file, input_format, noH, spec_atom_1, spec_atom_2):
		super().__init__(input_format, noH, spec_atom_1, spec_atom_2)
		self.parse_input(file)
		self.ATOMTYPES, self.CARTESIANS = np.array(self.ATOMTYPES), np.array(self.CARTESIANS)
		self.remove_hydrogens()

	def parse_input(self, _input):
		"""Parses input from either xyz file or com/gif file."""
		file_lines = DataParser.get_file_lines(_input)
		if self.FORMAT == 'xyz':
			for i in range(0,len(file_lines)):
				try:
					coord = file_lines[i].split()
					for i in range(len(coord)):
						try:
							coord[i] = float(coord[i])
						except ValueError: pass
					if len(coord) == 4:
						if isinstance(coord[1], float) and isinstance(coord[2], float) and isinstance(coord[3], float):
							[atom, x,y,z] = [coord[0], coord[1], coord[2], coord[3]]
							self.ATOMTYPES.append(atom)
							self.CARTESIANS.append([x,y,z])
				except: pass
		elif self.FORMAT == 'com' or self.FORMAT == 'gjf':
			for i in range(0,len(file_lines)):
				if file_lines[i].find("#") > -1:
					if len(file_lines[i+1].split()) == 0:
						start = i+5
					if len(file_lines[i+2].split()) == 0:
						start = i+6
					break
			for i in range(start, len(file_lines)):
				try:
					coord = file_lines[i].split()
					for i in range(len(coord)):
						try:
							coord[i] = float(coord[i])
						except ValueError: pass
					if len(coord) == 4:
						if isinstance(coord[1], float) and isinstance(coord[2], float) and isinstance(coord[3],float):
							[atom, x,y,z] = [coord[0], coord[1], coord[2], coord[3]]
							self.ATOMTYPES.append(atom)
							self.CARTESIANS.append([x,y,z])
				except: pass

			
class cclibParser(DataParser):
	"""Use the cclib package to extract data from generic computational chemistry output files."""
	def __init__(self, file, input_format, noH, spec_atom_1, spec_atom_2):
		super().__init__(input_format, noH, spec_atom_1, spec_atom_2)
		self.parse_input(file)
		self.ATOMTYPES = np.array(self.ATOMTYPES)
		self.remove_hydrogens()

	def parse_input(self, _input):
		"""Parses input file uses cclib file parser."""
		cclib_parsed = cclib.io.ccread(_input)
		self.CARTESIANS = np.array(cclib_parsed.atomcoords[-1])
		for i in cclib_parsed.atomnos:
			self.ATOMTYPES.append(periodic_table[i])


class RDKitParser(DataParser):
	"""Extract coordinates and atom types from rdkit mol object
	
	Attributes:
		ATOMTYPES (numpy array): List of elements present in molecular file
		CARTESIANS (numpy array): List of Cartesian (x,y,z) coordinates for each atom
	"""
	def __init__(self, mol, noH, spec_atom_1, spec_atom_2):
		super().__init__('RDKit-', noH, spec_atom_1, spec_atom_2)
		self.parse_input(mol)
		self.CARTESIANS = np.array(self.CARTESIANS)
		self.ATOMTYPES = np.array(self.ATOMTYPES)
		self.remove_hydrogens()

	def parse_input(self, _input):
		"""Store cartesians and symbols from mol object"""
		try:
			self.ATOMTYPES, self.CARTESIANS = [], []
			for i in range(_input.GetNumAtoms()):
				self.ATOMTYPES.append(_input.GetAtoms()[i].GetSymbol())
				pos = _input.GetConformer().GetAtomPosition(i)
				self.CARTESIANS.append([pos.x, pos.y, pos.z])
		except ValueError:
			self.ATOMTYPES, self.CARTESIANS = [], []
			print("Mol object does not have 3D coordinates!")
