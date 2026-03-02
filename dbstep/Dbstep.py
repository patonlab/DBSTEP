# -*- coding: UTF-8 -*-

# Python Libraries
import os, sys
from glob import glob
import numpy as np
from optparse import OptionParser

from dbstep import sterics, parse_data, calculator, writer
from dbstep.constants import periodic_table, bondi, charry_tkatchenko, metals

class dbstep:
	"""
	dbstep object that contains coordinates, steric data

	Objects that can currently be referenced are:
			L, Bmax, Bmin,
			occ_vol, bur_vol, bur_shell

	If steric scan is requested, Bmin and Bmax variables
	contain lists of params along scan
	"""

	_verbose_header_printed = False
	_column_header_printed = False
	_column_width = 0
	_file_col_width = 20

	def __init__(self, *args, **kwargs):
		self.file = args[0]
		# Sterimol Parameters
		self.L, self.Bmin, self.Bmax = False, False, False
		# Volume Parameters
		self.occ_vol, self.bur_vol, self.bur_shell = False, False, False

		if "options" in kwargs:
			self.options = kwargs["options"]
		else:
			self.options = set_options(kwargs)
		# SambVca mode: Bondi radii scaled by 1.17, exclude H atoms
		if hasattr(self.options, 'sambvca') and self.options.sambvca:
			self.options.radii = "bondi"
			self.options.SCALE_VDW = 1.17
			self.options.noH = True

		file = self.file
		options = self.options

		if isinstance(file, str):
			name, ext = os.path.splitext(file)
		else:
			name = file
			ext = "rdkit"

		# Auto-detect density surface from cube file input
		if ext == ".cube":
			options.surface = "density"

		# flag volume if buried shell requested
		if options.vshell:
			options.volume = True
		# sterimol scan requires grid-based measurement for per-radius slicing
		if options.sterimol and options.scan and options.measure == "classic":
			options.measure = "grid"

		origin = np.array([0, 0, 0])
		self._get_spec_atoms(options)
		mol = parse_data.read_input(file, ext, options)
		self._check_num_atoms(mol, file)

		# Assign radii / parse density and set up grid bounds
		x_min, x_max, y_min, y_max, z_min, z_max = self._assign_surface(mol, file, options, origin)

		# Rotate molecule to align atom1-atom2 bond along Z-axis
		self._orient_molecule(mol, options)

		# Recompute grid bounds after rotation so the grid covers the full rotated molecule
		if options.sterimol and options.surface == "vdw":
			[x_min, x_max, y_min, y_max, z_min, z_max, _] = sterics.max_dim(mol.CARTESIANS, mol.RADII, options)

		# Parse scan range
		r_min, r_max, r_intervals, strip_width = self._parse_scan(options)

		# Build occupancy grid
		occ_grid, occ_vol, point_tree, grid_axes = self._build_grid(
			mol, name, options, origin, x_min, x_max, y_min, y_max, z_min, z_max)

		# Print column headers (once across multi-file runs)
		self._print_column_header(options)

		# Compute steric parameters over radius range
		spheres, cylinders = self._compute(
			mol, file, options, origin, occ_grid, occ_vol, point_tree, grid_axes,
			r_min, r_max, r_intervals, strip_width)

		# Recompute L if a scan has been performed to get an overall L
		if options.measure == "grid" and r_intervals > 1 and options.sterimol:
			L, Bmax, Bmin, cyl = sterics.get_cube_sterimol(occ_grid, r_max, options.grid, 0.0)
			self.L = L
			if not options.quiet:
				print("\n   L parameter is {:5.2f} Ang".format(L))

		# Write PyMOL visualization files
		if options.pymol and ext != "rdkit":
			if options.sterimol:
				cylinders.append("   CYLINDER, 0., 0., 0., 0., 0., {:5.3f}, 0.1, 1.0, 1.0, 1.0, 0., 0.0, 1.0,".format(self.L))
			writer.xyz_export(file, mol)
			writer.pymol_export(file, mol, spheres, cylinders, options.isoval, options.visv, options.viss)

	def _assign_surface(self, mol, file, options, origin):
		"""Assign VDW radii or parse density cube, translate molecule, and remove metals.
		Returns grid bounds (x_min, x_max, y_min, y_max, z_min, z_max)."""
		x_min = x_max = y_min = y_max = z_min = z_max = 0.0

		if options.surface == "vdw":
			# Select radii set based on options
			radii_dict = charry_tkatchenko if options.radii == "charry-tkatchenko" else bondi
			for atom in mol.ATOMTYPES:
				if atom not in periodic_table and atom not in radii_dict:
					print("\n   UNABLE TO GENERATE VDW RADII FOR ATOM: ", atom)
					exit()
			mol.RADII = [radii_dict.get(atom, 2.0) for atom in mol.ATOMTYPES]
			mol.RADII = np.array(mol.RADII) * options.SCALE_VDW

			# Translate molecule to place atom1 at the origin
			if options.sterimol or options.volume:
				mol.CARTESIANS = calculator.translate_mol(mol, options, origin)

			# Remove metals when --nometals is specified (iterate in reverse to avoid index shifting)
			if options.no_metals:
				for i in range(len(mol.ATOMTYPES) - 1, -1, -1):
					if mol.ATOMTYPES[i] in metals:
						mol.ATOMTYPES = np.delete(mol.ATOMTYPES, i)
						mol.CARTESIANS = np.delete(mol.CARTESIANS, i, axis=0)
						mol.RADII = np.delete(mol.RADII, i)

			# Determine grid bounds from molecule extent
			[x_min, x_max, y_min, y_max, z_min, z_max, xyz_max] = sterics.max_dim(mol.CARTESIANS, mol.RADII, options)
			if options.gridsize and options.verbose:
				print("   Grid sizing requested: " + str(options.gridsize))

		elif options.surface == "density":
			if not hasattr(mol, "DENSITY"):
				print("   UNABLE TO READ DENSITY CUBE")
				exit()
			mol.DENSITY = np.array(mol.DENSITY)
			if options.verbose:
				print("\n   Read cube file {} containing {} points".format(file, mol.xdim * mol.ydim * mol.zdim))
			[x_min, y_min, z_min] = np.array(mol.ORIGIN)
			[x_max, y_max, z_max] = np.array(mol.ORIGIN) + np.array([(mol.xdim - 1) * mol.SPACING, (mol.ydim - 1) * mol.SPACING, (mol.zdim - 1) * mol.SPACING])
			options.grid = mol.SPACING

			# Translate density cube
			[mol.CARTESIANS, mol.ORIGIN, x_min, x_max, y_min, y_max, z_min, z_max, xyz_max] = calculator.translate_dens(
				mol, options, x_min, x_max, y_min, y_max, z_min, z_max,
				max(x_max, y_max, z_max, abs(x_min), abs(y_min), abs(z_min)), origin)

		else:
			print("   Requested surface {} is not currently implemented. Try either vdw or density".format(options.surface))
			exit()

		return x_min, x_max, y_min, y_max, z_min, z_max

	def _orient_molecule(self, mol, options):
		"""Rotate molecule to align atom1-atom2 bond along the Z-axis.
		Stores rotation angles in options.rotation for later use."""
		options.rotation = None
		if not options.sterimol:
			return
		point = calculator.point_vec(mol.CARTESIANS, options.spec_atom_2)
		if len(mol.CARTESIANS) > 1 and not options.norot:
			# Compute rotation angles
			options.rotation = calculator.get_rotation_angles(mol.CARTESIANS, options.spec_atom_1, point, options.atom3)
			if options.surface == "vdw":
				mol.CARTESIANS = calculator.rotate_mol(mol.CARTESIANS, options.spec_atom_1, point, options.verbose, options.atom3)
			elif options.surface == "density":
				mol.CARTESIANS, mol.ORIGIN = calculator.rotate_mol(mol.CARTESIANS, options.spec_atom_1, point, options.verbose, options.atom3, cube_origin=mol.ORIGIN)

	def _parse_scan(self, options):
		"""Parse radius or scan range. Returns (r_min, r_max, r_intervals, strip_width)."""
		r_intervals = 1
		if not options.scan:
			return options.radius, options.radius, 1, 0.0
		try:
			[r_min, r_max, strip_width] = [float(s) for s in options.scan.split(":")]
			r_intervals += int((r_max - r_min) / strip_width)
			return r_min, r_max, r_intervals, strip_width
		except (ValueError, AttributeError):
			print("   Can't read your scan request. Try something like --scan 3:5:0.25")
			exit()

	def _build_grid(self, mol, name, options, origin, x_min, x_max, y_min, y_max, z_min, z_max):
		"""Construct occupancy grid. Returns (occ_grid, occ_vol, point_tree, grid_axes)."""
		grid_axes = None
		occ_grid = occ_vol = point_tree = None

		if options.surface == "vdw":
			# User can override grid dimensions
			if options.gridsize:
				gs = [float(val) for val in options.gridsize.replace(":", ",").split(",")]
				if gs[1] < x_max or gs[0] > x_min or gs[3] < y_max or gs[2] > y_min or gs[5] < z_max or gs[4] > z_min:
					sys.exit("ERROR: Your molecule is larger than the gridsize you selected,\n       please try again with a larger gridsize")
				x_min, x_max, y_min, y_max, z_min, z_max = gs

			x_vals = np.linspace(x_min, x_max, int(1 + round((x_max - x_min) / options.grid)))
			y_vals = np.linspace(y_min, y_max, int(1 + round((y_max - y_min) / options.grid)))
			z_vals = np.linspace(z_min, z_max, int(1 + round((z_max - z_min) / options.grid)))

			if options.volume or options.measure == "grid":
				if options.volume and not (options.sterimol and options.measure == "grid"):
					# Fast path: skip full grid construction when grid sterimol not needed
					occ_grid, occ_vol = sterics.occupied_direct(mol.CARTESIANS, mol.RADII, origin, x_vals, y_vals, z_vals, options)
					point_tree = None
					grid_axes = (x_vals, y_vals, z_vals)
				else:
					# Standard path: full grid needed for grid-based sterimol
					grid = np.array(np.meshgrid(x_vals, y_vals, z_vals)).T.reshape(-1, 3)
					occ_grid, point_tree, occ_vol = sterics.occupied(grid, mol.CARTESIANS, mol.RADII, origin, options)

		elif options.surface == "density":
			x_vals = np.linspace(x_min, x_max, mol.xdim)
			y_vals = np.linspace(y_min, y_max, mol.ydim)
			z_vals = np.linspace(z_min, z_max, mol.zdim)
			if options.pymol:
				writer.WriteCubeData(name, mol)
			# Use 'ij' indexing so grid order matches cube file density order (x-slow, y-mid, z-fast)
			grid = np.array(np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')).reshape(3, -1).T
			occ_grid, occ_vol, point_tree = sterics.occupied_dens(grid, mol.DENSITY, options)
			# Rotate occupied grid to match the molecular orientation (bond along Z)
			if options.rotation is not None:
				occ_grid = calculator.apply_rotation(occ_grid, options.rotation)
			if options.volume:
				grid, point_tree = sterics.resize_grid(x_max, y_max, z_max, x_min, y_min, z_min, options, mol)

		return occ_grid, occ_vol, point_tree, grid_axes

	def _print_column_header(self, options):
		"""Print column headers once across multi-file runs."""
		if options.quiet or dbstep._column_header_printed:
			return
		fw = dbstep._file_col_width
		if options.volume and options.sterimol:
			header = "   {:>{fw}} {:>6} {:>6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format("File", "Atom1", "Atom2", "R/Å", "Mol_Vol", "%V_Bur", "%S_Bur", "Bmin", "Bmax", "L", fw=fw)
		elif options.sterimol:
			header = "   {:>{fw}} {:>6} {:>6} {:>10} {:>10} {:>10}".format("File", "Atom1", "Atom2", "Bmin", "Bmax", "L", fw=fw)
		elif options.volume:
			header = "   {:>{fw}} {:>6} {:>6} {:>10} {:>10} {:>10}".format("File", "Atom", "R/Å", "Mol_Vol", "%V_Bur", "%S_Bur", fw=fw)
		else:
			header = None
		if header:
			dbstep._column_width = len(header)
			print(header)
			print("   " + "-" * (dbstep._column_width - 3))
		dbstep._column_header_printed = True

	def _compute(self, mol, file, options, origin, occ_grid, occ_vol, point_tree, grid_axes,
				 r_min, r_max, r_intervals, strip_width):
		"""Run volume and/or sterimol calculations. Returns (spheres, cylinders) for PyMOL."""
		spheres, cylinders = [], []
		Bmin_list, Bmax_list, bur_vol_list, bur_shell_list = [], [], [], []

		# Precompute squared distances from origin for occupied grid points
		if options.volume:
			occ_dist2 = np.sum((occ_grid - origin) ** 2, axis=1)
		else:
			occ_dist2 = None

		fname = os.path.basename(file)
		fw = dbstep._file_col_width

		for rad in np.linspace(r_min, r_max, r_intervals):
			if options.volume:
				if rad == 0:
					bur_vol, bur_shell = 0.0, 0.0
				else:
					if options.vshell:
						strip_width = options.vshell
					bur_vol, bur_shell = sterics.buried_vol(occ_grid, point_tree, origin, rad, strip_width, options, occ_dist2=occ_dist2, grid_axes=grid_axes)
				bur_vol_list.append(bur_vol)
				bur_shell_list.append(bur_shell)

			if options.sterimol:
				if options.measure == "grid":
					L, Bmax, Bmin, cyl = sterics.get_cube_sterimol(occ_grid, rad, options.grid, strip_width, options.pos)
				elif options.surface == "vdw":
					L, Bmax, Bmin, cyl = sterics.get_classic_sterimol(mol.CARTESIANS, mol.RADII, mol.ATOMTYPES)
				else:
					print("   Can't use classic Sterimol with the isodensity surface. Use --measure grid or --surface vdw")
					exit()
				Bmin_list.append(Bmin)
				Bmax_list.append(Bmax)
				if options.pymol:
					cylinders.extend(cyl)

			# Tabulate result
			if options.volume and options.sterimol:
				if options.pymol:
					spheres.append("   SPHERE, 0.000, 0.000, 0.000, {:5.3f},".format(rad))
				if not options.quiet:
					atom2_str = ",".join(str(a) for a in options.spec_atom_2)
					print("   {:>{fw}} {:>6} {:>6} {:6.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f}".format(fname, options.spec_atom_1, atom2_str, rad, occ_vol, bur_vol, bur_shell, Bmin, Bmax, L, fw=fw))
			elif options.volume:
				if options.pymol:
					spheres.append("   SPHERE, 0.000, 0.000, 0.000, {:5.3f},".format(rad))
				if not options.quiet:
					print("   {:>{fw}} {:>6} {:6.2f} {:10.2f} {:10.2f} {:10.2f}".format(fname, options.spec_atom_1, rad, occ_vol, bur_vol, bur_shell, fw=fw))
			elif options.sterimol:
				if not options.quiet:
					atom2_str = ",".join(str(a) for a in options.spec_atom_2)
					print("   {:>{fw}} {:>6} {:>6} {:10.2f} {:10.2f} {:10.2f}".format(fname, options.spec_atom_1, atom2_str, Bmin, Bmax, L, fw=fw))

		# Store results on self
		if options.measure == "grid":
			self.occ_vol = occ_vol
		if options.sterimol:
			self.L = L
		if not options.scan:
			if options.sterimol:
				self.Bmax = Bmax
				self.Bmin = Bmin
			if options.volume:
				self.bur_vol = bur_vol
				self.bur_shell = bur_shell
		else:
			if options.sterimol:
				self.Bmax = Bmax_list
				self.Bmin = Bmin_list
			if options.volume:
				self.bur_vol = bur_vol_list
				self.bur_shell = bur_shell_list

		return spheres, cylinders

	def _get_spec_atoms(self, options):
		"""Gets the specification atoms from input or sets the defaults."""
		# if atoms are not specified upon input, grab first and second atom in file
		# allow for multiple ways to specify atoms (H1 or just 1)
		if not options.spec_atom_1:
			options.spec_atom_1 = 1
		else:
			try:
				options.spec_atom_1 = int(options.spec_atom_1)
			except Exception as atom1_exception:
				raise type(atom1_exception)(f"{options.spec_atom_1} is not a valid input for atom1. Please enter a positive integer index.")
			if options.spec_atom_1 <= 0:
				sys.exit(f"{options.spec_atom_1} is not a valid input for atom1. DBSTEP uses 1-indexed numbers, please enter a positive integer index.")
		# set default for atom 2
		if not options.spec_atom_2:
			options.spec_atom_2 = [2]
		else:
			if isinstance(options.spec_atom_2, str):
				if "," in options.spec_atom_2:
					options.spec_atom_2 = options.spec_atom_2.split(",")
				else:
					options.spec_atom_2 = [options.spec_atom_2]
			elif isinstance(options.spec_atom_2, int):
				options.spec_atom_2 = [options.spec_atom_2]
			try:
				options.spec_atom_2 = [int(atom) for atom in options.spec_atom_2]
			except Exception as atom2_error:
				raise type(atom2_error)(f"{options.spec_atom_2} is not a valid input for atom2. Valid inputs are: \n\tAn int, comma separated ints, or a python list of ints")
			for a2 in options.spec_atom_2:
				if a2 <= 0:
					sys.exit(f"{a2} is not a valid input for atom2. DBSTEP uses 1-indexed numbers, please enter a positive integer index.")

	def _check_num_atoms(self, mol, file):
		"""Checks if there are enough atoms in the input molecule for the type of calculation being performed."""
		if self.options.volume:
			min_atoms = 1
			calculation = "volume"
		else:
			min_atoms = 2
			calculation = "sterimol"
		num_atoms = len(mol.ATOMTYPES)
		if num_atoms < min_atoms:
			if mol.FORMAT == "RDKit":
				sys.exit(f"{num_atoms} atom(s) found in RDKit mol object, should have at least {min_atoms} atom(s) for {calculation} calculation.")
			else:
				sys.exit(f"{num_atoms} atom(s) found in {file}, should have at least {min_atoms} atom(s) for {calculation} calculation.")


class options_add:
	pass


def set_options(kwargs):
	# set default options and options provided
	options = options_add()
	# dictionary containing default values for options
	var_dict = {
		"verbose": ["verbose", False],
		"v": ["verbose", False],
		"grid": ["grid", 0.1],
		"scalevdw": ["SCALE_VDW", 1.0],
		"noH": ["noH", False],
		"nometals": ["no_metals", False],
		"norot": ["norot", False],
		"r": ["radius", 3.5],
		"scan": ["scan", False],
		"atom1": ["spec_atom_1", False],
		"atom2": ["spec_atom_2", False],
		"atom3": ["atom3", False],
		"exclude": ["exclude", False],
		"isoval": ["isoval", 0.0016],
		"s": ["sterimol", False],
		"sterimol": ["sterimol", False],
		"surface": ["surface", "vdw"],
		"debug": ["debug", False],
		"b": ["volume", False],
		"volume": ["volume", False],
		"vshell": ["vshell", False],
		"pymol": ["pymol", False],
		"quiet": ["quiet", False],
		"radii": ["radii", "bondi"],
		"sambvca": ["sambvca", False],
		"gridsize": ["gridsize", False],
		"measure": ["measure", "classic"],
		"pos": ["pos", False],
		"graph": ["graph", False],
		"fg": ["shared_fg", False],
		"shared_fg": ["shared_fg", False],
		"maxpath": ["max_path_length", 9],
		"max_path_length": ["max_path_length", 9],
		"voltype": ["voltype", "crippen"],
		"visv": ["visv", "circle"],
		"viss": ["viss", False],
	}

	for key in var_dict:
		vars(options)[var_dict[key][0]] = var_dict[key][1]
	for key in kwargs:
		if key in var_dict:
			vars(options)[var_dict[key][0]] = kwargs[key]
		else:
			print("Warning! Option: [", key, ":", kwargs[key], "] provided but no option exists, try -h to see available options.")

	return options


def main():
	files = []
	# get command line inputs. Use -h to list all possible arguments and default values
	parser = OptionParser(usage="Usage: %prog [options] <input1>.log <input2>.log ...")
	parser.add_option("--2d", dest="graph", action="store_true", help="[2D sterics] Analyze 2D steric contributions from SMILES input", default=False)
	parser.add_option("--2d-type", dest="voltype", action="store", default="crippen", choices=["crippen", "mcgowan", "degree"], help="[2D sterics] Atomic volume method: crippen, mcgowan, or degree (default: crippen)")
	parser.add_option("--atom1", dest="spec_atom_1", action="store", help="Specify the base atom number (default: 1)", default=False, metavar="spec_atom_1")
	parser.add_option("--atom2", dest="spec_atom_2", action="store", help="Specify the connected atom(s) number(s), e.g. 3 or 3,4 (default: 2)", default=False, metavar="spec_atom_2")
	parser.add_option("--atom3", dest="atom3", action="store", help="Align a third atom to the positive x direction", default=False)
	parser.add_option("--pymol", dest="pymol", action="store_true", help="Write PyMOL visualization and xyz output files", default=False)
	parser.add_option("--debug", dest="debug", action="store_true", help="Debug mode: graph grid points, print extra information", default=False)
	parser.add_option("--exclude", dest="exclude", action="store", help="Atom indices to ignore, comma-separated with no spaces", default=False, metavar="exclude")
	parser.add_option("--fg", dest="shared_fg", action="store", default=False, help="[2D sterics] SMILES pattern of shared functional group to define the origin, e.g. 'C(O)=O'")
	parser.add_option("--grid", dest="grid", action="store", help="Grid point spacing in Angstrom (default: 0.05)", default=0.05, type=float, metavar="grid")
	parser.add_option("--gridsize", dest="gridsize", action="store", help="Manual grid dimensions: xmin,xmax:ymin,ymax:zmin,zmax", default=False)
	parser.add_option("--isoval", dest="isoval", action="store", help="Density isovalue cutoff (default: 0.0016)", type="float", default=0.0016, metavar="isoval")
	parser.add_option("--maxpath", dest="max_path_length", type=int, action="store", default=9, help="[2D sterics] Maximum path length in bonds (default: 9)")
	parser.add_option("--noH", dest="noH", action="store_true", help="Exclude hydrogen atoms from steric measurements", default=False)
	parser.add_option("--nometals", dest="no_metals", action="store_true", help="Exclude metal atoms from steric measurements", default=False)
	parser.add_option("--norot", dest="norot", action="store_true", help="Do not rotate the molecule (use if structures have been pre-aligned)", default=False)
	parser.add_option("--pos", dest="pos", action="store_true", help="Measure Sterimol parameters in positive direction (from atom1 toward atom2)", default=False)
	parser.add_option("--quiet", dest="quiet", action="store_true", help="Suppress all print output", default=False)
	parser.add_option("--radii", dest="radii", action="store", choices=["bondi", "charry-tkatchenko"], help="VDW radii set: bondi or charry-tkatchenko (default: bondi)", default="bondi")
	parser.add_option("-r", dest="radius", action="store", help="Radius of sphere in Angstrom (default: 3.5)", default=3.5, type=float, metavar="radius")
	parser.add_option("--sambvca", dest="sambvca", action="store_true", help="Use SambVca 2.1 defaults: scale VDW radii by 1.17 and exclude H atoms", default=False)
	parser.add_option("--scalevdw", dest="SCALE_VDW", action="store", help="Scaling factor for VDW radii (default: 1.0)", type=float, default=1.0, metavar="SCALE_VDW")
	parser.add_option("--scan", dest="scan", action="store", help="Scan over a range of radii, format: rmin:rmax:interval", default=False, metavar="scan")
	parser.add_option("-s", "--sterimol", dest="sterimol", action="store_true", help="Compute Sterimol parameters (L, Bmin, Bmax)", default=False)
	parser.add_option("--measure", dest="measure", action="store", choices=["classic", "grid"], help="Sterimol method: classic (Verloop, default) or grid-based", default="classic", metavar="measure")
	parser.add_option("--surface", dest="surface", action="store", choices=["vdw", "density"], help="Surface type: Bondi VDW radii or density cube file (default: vdw)", default="vdw", metavar="surface")
	parser.add_option("-b", "--vbur", dest="volume", action="store_true", help="Calculate buried volume of input molecule", default=False)
	parser.add_option("-v", "--verbose", dest="verbose", action="store_true", help="Print verbose output", default=False)
	parser.add_option("--viss", dest="viss", action="store_true", help="Visualize Sterimol Bmin and Bmax in PyMOL as circle outlines", default=False)
	parser.add_option("--visv", dest="visv", action="store", choices=["circle", "sphere"], help="Visualize volume in PyMOL as circle or sphere (default: circle)", default="circle")
	parser.add_option("--vshell", dest="vshell", action="store", help="Calculate buried volume of hollow sphere with given shell width; use -r to set radius", default=False, type=float, metavar="width")
	(options, args) = parser.parse_args()

	# SambVca mode: Bondi radii scaled by 1.17, exclude H atoms
	if options.sambvca:
		options.radii = "bondi"
		options.SCALE_VDW = 1.17
		options.noH = True

	# Sterimol scan requires grid-based measurement for per-radius slicing
	if options.sterimol and options.scan and options.measure == "classic":
		options.measure = "grid"

	# make sure upper/lower case doesn't matter
	options.surface = options.surface.lower()

	# Get input files from commandline
	if len(sys.argv) > 1:
		for elem in sys.argv[1:]:
			try:
				for file in glob(elem):
					files.append(file)
			except IndexError:
				pass

	if len(files) == 0:
		sys.exit("    Please specify a valid input file and try again.")

	# Auto-detect density surface from cube file input
	if any(f.endswith(".cube") for f in files):
		options.surface = "density"

	# Set file column width based on longest filename
	dbstep._file_col_width = max(len(os.path.basename(f)) for f in files) + 2

	if not options.quiet:
		print("\n   \u00b7\u2584\u2584\u2584\u2584  \u2584\u2584\u2584\u2584\u00b7 .\u2584\u2584 \u00b7\u2584\u2584\u2584\u2584\u2584\u2584\u2584\u2584 . \u2584\u2584\u2584\u00b7")
		print("   \u2588\u2588\u258a \u2588\u2588 \u2590\u2588 \u2580\u2588\u258a\u2590\u2588 \u2580.\u2022\u2588\u2588  \u2580\u2584.\u2580\u00b7\u2590\u2588 \u2584\u2588")
		print("   \u2590\u2588\u00b7 \u2590\u2588\u258c\u2590\u2588\u2580\u2580\u2588\u2584\u2584\u2580\u2580\u2580\u2588\u2584\u2590\u2588.\u258a\u2590\u2580\u2580\u258a\u2584 \u2588\u2588\u2580\u00b7")
		print("   \u2588\u2588. \u2588\u2588 \u2588\u2588\u2584\u258a\u2590\u2588\u2590\u2588\u2584\u258a\u2590\u2588\u2590\u2588\u258c\u00b7\u2590\u2588\u2584\u2584\u258c\u2590\u2588\u258a\u00b7\u2022")
		print("   \u2580\u2580\u2580\u2580\u2580\u2022 \u00b7\u2580\u2580\u2580\u2580  \u2580\u2580\u2580\u2580 \u2580\u2580\u2580  \u2580\u2580\u2580 .\u2580   ")
		print("")

		if options.graph:
			voltype_label = "McGowan volumes" if options.voltype.lower() == "mcgowan" else "Crippen molar refractivities"
			print("   2D graph mode: using connectivity and {} for atomic contributions\n".format(voltype_label))
		else:
			if options.volume:
				print("   Buried volume (Vbur) will be computed")
			if options.sterimol:
				print("   Sterimol parameters will be generated using {} mode".format("grid-based" if options.measure == "grid" else "classic"))
			if options.surface == "vdw":
				print("   Using a Cartesian grid-spacing of {:5.4f} Angstrom".format(options.grid))
				radii_label = "Charry-Tkatchenko" if options.radii == "charry-tkatchenko" else "Bondi"
				print("   {} atomic radii will be scaled by {}".format(radii_label, options.SCALE_VDW))
				print("   Hydrogen atoms are {}\n".format("excluded" if options.noH else "included"))
			else:
				print("   Using {} isodensity surface with cutoff value of {:5.4f} au".format(options.surface, options.isoval))
				print("   Cartesian grid-spacing will be determined by cube file(s)\n")

	# loop over all specified output files
	for file in files:
		if options.graph:
			try:
				from dbstep import graph
			except ModuleNotFoundError as e:
				print(e, "\nPlease install necessary modules and try again.")
				sys.exit()
			vec_df = graph.mol_to_vec(file, options.shared_fg, options.voltype, options.max_path_length, options.verbose)
			numeric_cols = vec_df.select_dtypes(include='number').columns
			vec_df[numeric_cols] = vec_df[numeric_cols].round(2)
			vec_df.to_csv(file.split(".")[0] + "_2d_output.csv", index=False)
		else:
			dbstep(file, options=options)

	if dbstep._column_width and not options.quiet:
		print("   " + "-" * (dbstep._column_width - 3))


if __name__ == "__main__":
	main()
