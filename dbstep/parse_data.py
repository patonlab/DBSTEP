# -*- coding: UTF-8 -*-
import os,sys
import numpy as np
import cclib


"""
parse_data

Parses data from files
Currently supporting: 
	.cube Gaussian volumetric files 
	all filetypes parsed by the cclib python package (see https://cclib.github.io/)
"""


BOHR_TO_ANG = 0.529177249

periodictable = ["","H","He","Li","Be","B","C","N","O","F","Ne",
	"Na","Mg","Al","Si","P","S","Cl","Ar",
	"K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
	"Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
	"Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
	"Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"]


def element_id(massno, num=False):
	"""Return element id number"""
	try:
		if num:
			return periodictable.index(massno)
		else: return periodictable[massno]
	except IndexError:
		return "XX"


class GetCubeData:
	""" Read data from cube file, obtian XYZ Cartesians, dimensions, and volumetric data """
	def __init__(self, file):
		if not os.path.exists(file+".cube"): print("\nFATAL ERROR: cube file [ %s ] does not exist"%file)
		def getATOMTYPES(self, outlines, format):
			self.ATOMTYPES, self.ATOMNUM, self.CARTESIANS, self.DENSITY, self.DENSITY_LINE = [], [], [], [], []
			if format == 'cube':
				for i in range(2,len(outlines)):
					try:
						coord = outlines[i].split()
						for j in range(len(coord)):
							try:
								coord[j] = float(coord[j])
							except ValueError: pass
						if i == 2:
							self.ORIGIN = [coord[1]*BOHR_TO_ANG, coord[2]*BOHR_TO_ANG,coord[3]*BOHR_TO_ANG]
						elif i == 3:
							self.xdim = int(coord[0])
							self.SPACING = coord[1]*BOHR_TO_ANG
							self.x_inc = [coord[1]*BOHR_TO_ANG,coord[2]*BOHR_TO_ANG,coord[3]*BOHR_TO_ANG]
						elif i == 4:
							self.ydim = int(coord[0])
							self.y_inc = [coord[1]*BOHR_TO_ANG,coord[2]*BOHR_TO_ANG,coord[3]*BOHR_TO_ANG]
						elif i == 5:
							self.zdim = int(coord[0])
							self.z_inc = [coord[1]*BOHR_TO_ANG,coord[2]*BOHR_TO_ANG,coord[3]*BOHR_TO_ANG]
						elif len(coord) == 5:
							if coord[0] == int(coord[0]) and isinstance(coord[2],float) == True and isinstance(coord[3],float) and isinstance(coord[4],float):
								[atom, x,y,z] = [periodictable[int(coord[0])], float(coord[2])*BOHR_TO_ANG, float(coord[3])*BOHR_TO_ANG, float(coord[4])*BOHR_TO_ANG]
								self.ATOMNUM.append(int(coord[0]))
								self.ATOMTYPES.append(atom)
								self.CARTESIANS.append([x,y,z])
						if coord[0] != int(coord[0]):
							for val in coord:
								self.DENSITY.append(val)
							self.DENSITY_LINE.append(outlines[i])
					except: pass
		self.FORMAT = 'cube'
		molfile = open(file+"."+self.FORMAT,"r")
		mollines = molfile.readlines()
		getATOMTYPES(self, mollines, self.FORMAT)
		self.INCREMENTS=np.asarray([self.x_inc,self.y_inc,self.z_inc])
		cube_data = np.zeros([self.xdim,self.ydim,self.zdim])
		self.DENSITY = np.asarray(self.DENSITY)
		self.DATA = np.reshape(self.DENSITY,(self.xdim,self.ydim,self.zdim))
		vol_x = []
		vol_y = []
		vol_z = []
		for i in range(self.xdim):
			for j in range(self.ydim):
				for k in range(self.zdim):
					if self.DATA[i][j][k] > 0.05:
						vol_x.append(self.ORIGIN[0]+(i-1)*self.x_inc[0] + (j-1)*self.x_inc[1] + (k-1)*self.x_inc[2])
						vol_y.append(self.ORIGIN[1]+(i-1)*self.y_inc[0] + (j-1)*self.y_inc[1] + (k-1)*self.y_inc[2])
						vol_z.append(self.ORIGIN[2]+(i-1)*self.z_inc[0] + (j-1)*self.z_inc[1] + (k-1)*self.z_inc[2])
		self.ATOMTYPES = np.array(self.ATOMTYPES)
		self.CARTESIANS = np.array(self.CARTESIANS)


class GetXYZData:
	""" Read XYZ Cartesians from file """
	def __init__(self, file, ext, noH, spec_atom_1, spec_atom_2):
		if ext == '.xyz':
			if not os.path.exists(file+".xyz"):
				sys.exit("\nFATAL ERROR: XYZ file [ %s ] does not exist"%file)
		elif ext == '.log':
			if not os.path.exists(file+".log"):
				print(("\nFATAL ERROR: log file [ %s ] does not exist"%file))
		self.FORMAT = ext
		molfile = open(file+self.FORMAT,"r")
		outlines = molfile.readlines()

		self.ATOMTYPES, self.CARTESIANS = [], []
		if self.FORMAT == '.xyz':
			for i in range(0,len(outlines)):
				try:
					coord = outlines[i].split()
					for i in range(len(coord)):
						try:
							coord[i] = float(coord[i])
						except ValueError: pass
					if len(coord) == 4:
						if isinstance(coord[1],float) and isinstance(coord[2],float) and isinstance(coord[3],float):
							[atom, x,y,z] = [coord[0], coord[1], coord[2], coord[3]]
							self.ATOMTYPES.append(atom)
							self.CARTESIANS.append([x,y,z])
				except: pass
		elif self.FORMAT == '.com' or self.FORMAT == '.gjf':
			for i in range(0,len(outlines)):
				if outlines[i].find("#") > -1:
					if len(outlines[i+1].split()) == 0: 
						start = i+5
					if len(outlines[i+2].split()) == 0: 
						start = i+6
					break
			for i in range(start,len(outlines)):
				try:
					coord = outlines[i].split()
					for i in range(len(coord)):
						try:
							coord[i] = float(coord[i])
						except ValueError: pass
					if len(coord) == 4:
						if isinstance(coord[1],float) and isinstance(coord[2],float) and isinstance(coord[3],float):
							[atom, x,y,z] = [coord[0], coord[1], coord[2], coord[3]]
							self.ATOMTYPES.append(atom)
							self.CARTESIANS.append([x,y,z])
				except: pass
		self.ATOMTYPES = np.array(self.ATOMTYPES)
		self.CARTESIANS = np.array(self.CARTESIANS)
		#remove hydrogens if requested, update spec_atom numbering if necessary
		if noH:
			atom1_id,atom2_id = -1,-1
			for n,atom in enumerate(self.ATOMTYPES):
				if atom+str(n+1) == spec_atom_1:
					atom1_id = n
				elif atom+str(n+1) == spec_atom_2:
					atom2_id = n
			hs=[]
			for n,atom in enumerate(self.ATOMTYPES):
				if atom == "H":
					hs.append(n)
			center_sub,lig_sub = 0,0
			for n in range(len(hs)):
				if hs[n] < atom1_id:
					center_sub -= 1
				if hs[n] < atom2_id:
					lig_sub -= 1
			atom1_id += center_sub
			atom2_id += lig_sub
			
			self.ATOMTYPES = np.delete(self.ATOMTYPES,hs)
			self.CARTESIANS = np.delete(self.CARTESIANS,hs,0)
			
			if atom1_id >= 0: 
				self.spec_atom_1 = self.ATOMTYPES[atom1_id]+str(atom1_id+1)
			else:
				self.spec_atom_1 = False
			if atom2_id >= 0: 
				self.spec_atom_2 = self.ATOMTYPES[atom2_id]+str(atom2_id+1)
			else:
				self.spec_atom_2 = False

			
class GetData_cclib:
	"""
	Use the cclib package to extract data from generic computational chemistry output files.
	
	Attributes:
		ATOMTYPES (numpy array): List of elements present in molecular file
		CARTESIANS (numpy array): List of Cartesian (x,y,z) coordinates for each atom
		FORMAT (str): file format
	"""
	def __init__(self, file, ext, noH, spec_atom_1, spec_atom_2):
		self.ATOMTYPES, self.CARTESIANS = [],[]
		#parse coordinate from file
		filename = file+ext
		parsed = cclib.io.ccread(filename)
		
		self.FORMAT = ext 
		
		#store cartesians and symbols
		self.CARTESIANS = np.array(parsed.atomcoords[-1])
		[self.ATOMTYPES.append(periodictable[i]) for i in parsed.atomnos]
		self.ATOMTYPES = np.array(self.ATOMTYPES)
		
		#remove hydrogens if requested, update spec_atom numbering if necessary
		if noH:
			atom1_id, atom2_id = spec_atom_1, spec_atom_2
			hs=[]
			for n,atom in enumerate(self.ATOMTYPES):
				if atom == "H":
					hs.append(n)
			center_sub,lig_sub = 0,0
			for n in range(len(hs)):
				if hs[n] < atom1_id:
					center_sub -= 1
				if hs[n] < atom2_id:
					lig_sub -= 1
			atom1_id += center_sub
			atom2_id += lig_sub
			
			self.ATOMTYPES = np.delete(self.ATOMTYPES,hs)
			self.CARTESIANS = np.delete(self.CARTESIANS,hs,0)
			
			if atom1_id >= 0: 
				self.spec_atom_1 = atom1_id
			else:
				self.spec_atom_1 = False
			if atom2_id >= 0: 
				self.spec_atom_2 = atom2_id
			else:
				self.spec_atom_2 = False
	
	
class GetData_RDKit:
	"""
	Extract coordinates and atom types from rdkit mol object
	
	Attributes:
		ATOMTYPES (numpy array): List of elements present in molecular file
		CARTESIANS (numpy array): List of Cartesian (x,y,z) coordinates for each atom
		FORMAT (str): file format
	"""
	def __init__(self, mol, noH, spec_atom_1, spec_atom_2):
		self.FORMAT = 'RDKit-'
		#store cartesians and symbols from mol object
		try:
			self.ATOMTYPES, self.CARTESIANS = [],[]
			for i in range(mol.GetNumAtoms()):
				self.ATOMTYPES.append(mol.GetAtoms()[i].GetSymbol())
				pos = mol.GetConformer().GetAtomPosition(i)
				self.CARTESIANS.append([pos.x, pos.y, pos.z])
		except ValueError:
			self.ATOMTYPES, self.CARTESIANS = [],[]
			print("Mol object does not have 3D coordinates!")
			# self.ATOMTYPES, self.CARTESIANS = [],[]
			# AllChem.EmbedMolecule(mol,randomSeed=42) #currently not importing any rdkit so this will fails
			# for i in range(mol.GetNumAtoms()):
			# 	self.ATOMTYPES.append(mol.GetAtoms()[i].GetSymbol())
			# 	pos = mol.GetConformer().GetAtomPosition(i)
			# 	self.CARTESIANS.append([pos.x, pos.y, pos.z])

		self.CARTESIANS = np.array(self.CARTESIANS)
		self.ATOMTYPES = np.array(self.ATOMTYPES)
		
		#remove hydrogens if requested, update spec_atom numbering if necessary
		if noH:
			atom1_id, atom2_id = spec_atom_1, spec_atom_2
			hs=[]
			for n,atom in enumerate(self.ATOMTYPES):
				if atom == "H":
					hs.append(n)
			center_sub,lig_sub = 0,0
			for n in range(len(hs)):
				if hs[n] < atom1_id:
					center_sub -= 1
				if hs[n] < atom2_id:
					lig_sub -= 1
			atom1_id += center_sub
			atom2_id += lig_sub
			
			self.ATOMTYPES = np.delete(self.ATOMTYPES,hs)
			self.CARTESIANS = np.delete(self.CARTESIANS,hs,0)
			
			if atom1_id >= 0: 
				self.spec_atom_1 = atom1_id
			else:
				self.spec_atom_1 = False
			if atom2_id >= 0: 
				self.spec_atom_2 = atom2_id
			else:
				self.spec_atom_2 = False
	