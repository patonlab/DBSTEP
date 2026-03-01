# -*- coding: UTF-8 -*-

# Python Libraries
import os, sys, shutil
from glob import glob
import numpy as np
from optparse import OptionParser

from dbstep import sterics, parse_data, calculator, writer
from dbstep.constants import periodic_table, bondi, metals


class dbstep:
	"""
	dbstep object that contains coordinates, steric data

	Objects that can currently be referenced are:
			grid, onehot_grid, unocc_grid
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
		# QSAR specifications
		self.dimensions, self.qsar_dir, self.interaction_energy = False, False, []
		# Grid Information
		self.grid, self.unocc_grid, self.onehot_grid = False, False, False
		# Sterimol Parameters
		self.L, self.Bmin, self.Bmax = False, False, False
		# Volume Parameters
		self.occ_vol, self.bur_vol, self.bur_shell = False, False, False
		if "options" in kwargs:
			self.options = kwargs["options"]
		else:
			self.options = set_options(kwargs)
		# SambVca mode: scale VDW radii by 1.17 and exclude H atoms
		if hasattr(self.options, 'sambvca') and self.options.sambvca:
			self.options.SCALE_VDW = 1.17
			self.options.noH = True
		if "QSAR" in kwargs:
			QSAR = kwargs["QSAR"]
		else:
			QSAR = False

		file = self.file
		options = self.options

		spheres, cylinders = [], []
		if isinstance(file, str):
			name, ext = os.path.splitext(file)
		else:
			name = file
			ext = "rdkit"

		r_intervals, origin = 1, np.array([0, 0, 0])

		self._get_spec_atoms(options)

		# Parse coordinate/volumetric information
		mol = parse_data.read_input(file, ext, options)

		self._check_num_atoms(mol, file)

		# flag volume if buried shell requested
		if options.vshell:
			options.volume = True
		# if measuring volume, need to measure from grid
		if options.volume:
			options.measure = "grid"

		if options.qsar:
			if options.grid < 0.5:
				options.grid = 0.5
				if options.verbose:
					print("   Adjusting grid spacing to 0.5A for QSAR analysis")

		# if surface = VDW the molecular volume is defined by tabulated radii
		# This is necessary when a density cube is not supplied
		# if surface = Density the molecular volume is defined by an isodensity surface from a .cube file
		# This is the default when a density cube is supplied although it can be over-ridden at the command prompt
		
		# surfaces can either be formed from Van der Waals (bondi) radii (=vdw) or cube densities (=density)
		if options.surface == "vdw":
			# generate Bondi radii from atom types
			try:
				mol.RADII = [bondi[atom] for atom in mol.ATOMTYPES]
				if not options.quiet and not dbstep._verbose_header_printed:
					print("\n   \u00b7\u2584\u2584\u2584\u2584  \u2584\u2584\u2584\u2584\u00b7 .\u2584\u2584 \u00b7\u2584\u2584\u2584\u2584\u2584\u2584\u2584\u2584 . \u2584\u2584\u2584\u00b7")
					print("   \u2588\u2588\u258a \u2588\u2588 \u2590\u2588 \u2580\u2588\u258a\u2590\u2588 \u2580.\u2022\u2588\u2588  \u2580\u2584.\u2580\u00b7\u2590\u2588 \u2584\u2588")
					print("   \u2590\u2588\u00b7 \u2590\u2588\u258c\u2590\u2588\u2580\u2580\u2588\u2584\u2584\u2580\u2580\u2580\u2588\u2584\u2590\u2588.\u258a\u2590\u2580\u2580\u258a\u2584 \u2588\u2588\u2580\u00b7")
					print("   \u2588\u2588. \u2588\u2588 \u2588\u2588\u2584\u258a\u2590\u2588\u2590\u2588\u2584\u258a\u2590\u2588\u2590\u2588\u258c\u00b7\u2590\u2588\u2584\u2584\u258c\u2590\u2588\u258a\u00b7\u2022")
					print("   \u2580\u2580\u2580\u2580\u2580\u2022 \u00b7\u2580\u2580\u2580\u2580  \u2580\u2580\u2580\u2580 \u2580\u2580\u2580  \u2580\u2580\u2580 .\u2580   ")
					print("")
					if options.volume:
						print("   Buried volume (Vbur) will be computed")
					if options.sterimol:
						print("   Sterimol parameters will be generated using {} mode".format("grid-based" if options.measure == "grid" else "classic"))
					print("   Using a Cartesian grid-spacing of {:5.4f} Angstrom".format(options.grid))
					print("   Bondi atomic radii will be scaled by {}".format(options.SCALE_VDW))
					print("   Hydrogen atoms are {}\n".format("excluded" if options.noH else "included"))
					dbstep._verbose_header_printed = True
			except KeyError:
				mol.RADII = []
				for atom in mol.ATOMTYPES:
					if atom not in periodic_table:
						print("\n   UNABLE TO GENERATE VDW RADII FOR ATOM: ", atom)
						exit()
					elif atom not in bondi:
						mol.RADII.append(2.0)
					else:
						mol.RADII.append(bondi[atom])
			# scale radii by a factor
			mol.RADII = np.array(mol.RADII) * options.SCALE_VDW
		elif options.surface == "density":
			if hasattr(mol, "DENSITY"):
				mol.DENSITY = np.array(mol.DENSITY)
				if options.verbose:
					print("\n   Read cube file {} containing {} points".format(file, mol.xdim * mol.ydim * mol.zdim))
				[x_min, y_min, z_min] = np.array(mol.ORIGIN)
				[x_max, y_max, z_max] = np.array(mol.ORIGIN) + np.array([(mol.xdim - 1) * mol.SPACING, (mol.ydim - 1) * mol.SPACING, (mol.zdim - 1) * mol.SPACING])
				xyz_max = max(x_max, y_max, z_max, abs(x_min), abs(y_min), abs(z_min))
				# overrides grid settings
				options.grid = mol.SPACING
			else:
				print("   UNABLE TO READ DENSITY CUBE")
				exit()
		else:
			print("   Requested surface {} is not currently implemented. Try either vdw or density".format(options.surface))
			exit()

		# Translate molecule to place atom1 at the origin
		if options.surface == "vdw" and (options.sterimol or options.volume):
			mol.CARTESIANS = calculator.translate_mol(mol, options, origin)
		elif options.surface == "density":
			[mol.CARTESIANS, mol.ORIGIN, x_min, x_max, y_min, y_max, z_min, z_max, xyz_max] = calculator.translate_dens(mol, options, x_min, x_max, y_min, y_max, z_min, z_max, xyz_max, origin)

		# if computing sterimol parameters: rotate molecule and compute
		if options.sterimol:
			# Check if we want to calculate parameters for mono- bi- or tridentate ligand
			spec_atom_2 = ""
			point = calculator.point_vec(mol.CARTESIANS, options.spec_atom_2)

			# Rotate the molecule about the origin to align the metal-ligand bond along the (positive) Z-axis
			# the x and y directions are arbitrary
			if len(mol.CARTESIANS) > 1 and not options.norot:
				if options.surface == "vdw":
					mol.CARTESIANS = calculator.rotate_mol(mol.CARTESIANS, options.spec_atom_1, point, options.verbose, options.atom3)
				elif options.surface == "density":
					mol.CARTESIANS, mol.ORIGIN = calculator.rotate_mol(mol.CARTESIANS, options.spec_atom_1, point, options.verbose, options.atom3, cube_origin=mol.ORIGIN)

		# Remove metals from the steric analysis when --nometals is specified
		# This can't be done for densities
		if options.surface == "vdw":
			# Find maximum horizontal and vertical directions (coordinates + vdw) in which the molecule is fully contained

			# remove metals
			for i, atom in enumerate(mol.ATOMTYPES):
				if atom in metals and options.no_metals:
					mol.ATOMTYPES = np.delete(mol.ATOMTYPES, i)
					mol.CARTESIANS = np.delete(mol.CARTESIANS, i, axis=0)
					mol.RADII = np.delete(mol.RADII, i)

			# determine grid size based on molecule vdw radii and radius selected for buried vol
			[x_min, x_max, y_min, y_max, z_min, z_max, xyz_max] = sterics.max_dim(mol.CARTESIANS, mol.RADII, options)
			if options.gridsize:
				if options.verbose:
					print("   Grid sizing requested: " + str(options.gridsize))
		if QSAR:
			self.dimensions = [x_min, x_max, y_min, y_max, z_min, z_max]
			return
		# Read the requested radius or range
		if not options.scan:
			r_min, r_max, strip_width = options.radius, options.radius, 0.0
		else:
			try:
				[r_min, r_max, strip_width] = [float(scan) for scan in options.scan.split(":")]
				r_intervals += int((r_max - r_min) / strip_width)
			except (ValueError, AttributeError):
				print("   Can't read your scan request. Try something like --scan 3:5:0.25")
				exit()

		# Iterate over the grid points to see whether this is within VDW radius of any atom(s)
		# Grid point occupancy is either yes/no (1/0)
		# To save time this is currently done using a cuboid rather than cubic shaped-grid
		grid_axes = None
		if options.surface == "vdw":
			# user can choose to increase grid size / use in QSAR studies
			if options.gridsize:
				[x_minus, x_plus, y_minus, y_plus, z_minus, z_plus] = [float(val) for val in options.gridsize.replace(":", ",").split(",")]
				sizeflag = True
				if x_plus < x_max or x_minus > x_min:
					sizeflag = False
				elif y_plus < y_max or y_minus > y_min:
					sizeflag = False
				elif z_plus < z_max or z_minus > z_min:
					sizeflag = False
				if sizeflag:
					n_x_vals = int(1 + round((x_plus - x_minus) / options.grid))
					n_y_vals = int(1 + round((y_plus - y_minus) / options.grid))
					n_z_vals = int(1 + round((z_plus - z_minus) / options.grid))
					x_vals = np.linspace(x_minus, x_plus, n_x_vals)
					y_vals = np.linspace(y_minus, y_plus, n_y_vals)
					z_vals = np.linspace(z_minus, z_plus, n_z_vals)
				else:
					# sys exit
					sys.exit("ERROR: Your molecule is larger than the gridsize you selected,\n       please try again with a larger gridsize")
			else:
				n_x_vals = int(1 + round((x_max - x_min) / options.grid))
				n_y_vals = int(1 + round((y_max - y_min) / options.grid))
				n_z_vals = int(1 + round((z_max - z_min) / options.grid))
				x_vals = np.linspace(x_min, x_max, n_x_vals)
				y_vals = np.linspace(y_min, y_max, n_y_vals)
				z_vals = np.linspace(z_min, z_max, n_z_vals)

			if options.measure == "grid":
				if options.volume and not options.sterimol and not options.qsar:
					# Fast path: skip full grid construction for volume-only calculations
					occ_grid, occ_vol = sterics.occupied_direct(mol.CARTESIANS, mol.RADII, origin, x_vals, y_vals, z_vals, options)
					point_tree = None
					grid_axes = (x_vals, y_vals, z_vals)
				else:
					# Standard path: full grid needed for sterimol/qsar
					grid = np.array(np.meshgrid(x_vals, y_vals, z_vals)).T.reshape(-1, 3)
					# compute which grid points occupy molecule
					if options.qsar:
						occ_grid, unocc_grid, onehot_grid, point_tree, occ_vol = sterics.occupied(grid, mol.CARTESIANS, mol.RADII, origin, options)
					else:
						occ_grid, point_tree, occ_vol = sterics.occupied(grid, mol.CARTESIANS, mol.RADII, origin, options)
					grid_axes = None

			if options.qsar:
				if options.verbose:
					print("\n   Creating interaction energy grid xyz files in 'grid_" + name + "' directory")
				probe = "Ar"
				path = os.getcwd() + "/grid_" + name + "/"
				self.qsar_dir = path
				if os.path.exists(path):
					if options.verbose:
						print("   Overwriting: " + path)
					shutil.rmtree(path)
				os.mkdir(path)

				self.grid = grid
				self.unocc_grid = unocc_grid
				self.onehot_grid = onehot_grid

				for n, gridpoint in enumerate(unocc_grid):
					self.interaction_energy.append(0.0)
					xyzfile = open(path + "GRIDPOINT_" + probe + "_" + str(n) + ".xyz", "w")
					xyzfile.write(str(len(mol.ATOMTYPES) + 1) + "\n")
					xyzfile.write(path + "GRIDPOINT_" + probe + "_" + str(n) + "\n")

					for i, atom in enumerate(mol.ATOMTYPES):
						[x, y, z] = mol.CARTESIANS[i]
						[gx, gy, gz] = gridpoint
						xyzfile.write("{} {:10.5f} {:10.5f} {:10.5f}\n".format(mol.ATOMTYPES[i], x, y, z))
					xyzfile.write("{} {:10.5f} {:10.5f} {:10.5f}\n".format(probe, gx, gy, gz))

				xyzfile = open(path + "REF_" + probe + ".xyz", "w")
				xyzfile.write(str(len(mol.ATOMTYPES) + 1) + "\n")
				xyzfile.write("REF_" + probe + "\n")
				for i, atom in enumerate(mol.ATOMTYPES):
					[x, y, z] = mol.CARTESIANS[i]
					[gx, gy, gz] = gridpoint

					xyzfile.write("{} {:10.5f} {:10.5f} {:10.5f}\n".format(mol.ATOMTYPES[i], x, y, z))
				xyzfile.write("{} {:10.5f} {:10.5f} {:10.5f}\n".format(probe, gx + 100, gy + 100, gz + 100))

		elif options.surface == "density":
			x_vals = np.linspace(x_min, x_max, mol.xdim)
			y_vals = np.linspace(y_min, y_max, mol.ydim)
			z_vals = np.linspace(z_min, z_max, mol.zdim)
			# writes a new grid to cube file
			writer.WriteCubeData(name, mol)
			# define the grid points containing the molecule
			grid = np.array(np.meshgrid(x_vals, y_vals, z_vals)).T.reshape(-1, 3)
			# compute occupancy based on isodensity value applied to cube and remove points where there is no molecule
			occ_grid, occ_vol, point_tree = sterics.occupied_dens(grid, mol.DENSITY, options)

			# adjust sizing of grid to fit sphere if necessary
			if options.volume:
				grid = sterics.resize_grid(x_max, y_max, z_max, x_min, y_min, z_min, options, mol)

		if not options.quiet and not dbstep._column_header_printed:
			fw = dbstep._file_col_width
			if options.volume and options.sterimol:
				header = "   {:>{fw}} {:>6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10}".format("File", "Atom", "R/Å", "%V_Bur", "%S_Bur", "Bmin", "Bmax", "L", fw=fw)
			elif options.volume:
				header = "   {:>{fw}} {:>6} {:>6} {:>10} {:>10}".format("File", "Atom", "R/Å", "%V_Bur", "%S_Bur", fw=fw)
			else:
				header = None
			if header:
				dbstep._column_width = len(header)
				print(header)
				print("   " + "-" * (dbstep._column_width - 3))
			dbstep._column_header_printed = True

		Bmin_list, Bmax_list, bur_vol_list, bur_shell_list = [], [], [], []

		# Precompute squared distances from origin for occupied grid points
		# This replaces the occ_point_tree KDTree entirely — just threshold-count per radius
		if options.volume:
			occ_dist2 = np.sum((occ_grid - origin) ** 2, axis=1)
		else:
			occ_dist2 = None

		# Measure Sterimol or Volume
		for rad in np.linspace(r_min, r_max, r_intervals):
			# The buried volume is defined in terms of occupied voxels.
			# If a scan is requested, radius of sphere = rad
			if options.volume:
				if rad == 0:
					bur_vol, bur_shell = 0.0, 0.0
				else:
					if options.vshell:
						strip_width = options.vshell
					bur_vol, bur_shell = sterics.buried_vol(occ_grid, point_tree, origin, rad, strip_width, options, occ_dist2=occ_dist2, grid_axes=grid_axes)
				bur_vol_list.append(bur_vol)
				bur_shell_list.append(bur_shell)
			# Sterimol parameters: classic (VDW radii, default) or grid (occupied voxels, used when volume is also requested)
			if options.sterimol:
				if options.measure == "grid":
					L, Bmax, Bmin, cyl = sterics.get_cube_sterimol(occ_grid, rad, options.grid, strip_width, options.pos)
				elif options.measure == "classic":
					if options.surface == "vdw":
						L, Bmax, Bmin, cyl = sterics.get_classic_sterimol(mol.CARTESIANS, mol.RADII, mol.ATOMTYPES)
					elif options.surface == "density":
						print("   Can't use classic Sterimol with the isodensity surface. Either use VDW radii (--surface vdw) or use grid Sterimol (--sterimol grid)")
						exit()
				Bmin_list.append(Bmin)
				Bmax_list.append(Bmax)

				# for pymol visualization
				for c in cyl:
					cylinders.append(c)

			# Tabulate result
			fname = os.path.basename(file)
			fw = dbstep._file_col_width
			if options.volume and options.sterimol:
				# for pymol visualization
				spheres.append("   SPHERE, 0.000, 0.000, 0.000, {:5.3f},".format(rad))
				if not options.quiet:
					print("   {:>{fw}} {:>6} {:6.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f}".format(fname, options.spec_atom_1, rad, bur_vol, bur_shell, Bmin, Bmax, L, fw=fw))
			elif options.volume:
				spheres.append("   SPHERE, 0.000, 0.000, 0.000, {:5.3f},".format(rad))
				if not options.quiet:
					print("   {:>{fw}} {:>6} {:6.2f} {:10.2f} {:10.2f}".format(fname, options.spec_atom_1, rad, bur_vol, bur_shell, fw=fw))
			elif options.sterimol:
				if not options.scan:
					if not options.quiet:
						print("   {} / Bmin: {:5.2f} / Bmax: {:5.2f} / L: {:5.2f}".format(file, Bmin, Bmax, L))
				else:
					if not options.quiet:
						print("   {} / R: {:5.2f} / Bmin: {:5.2f} / Bmax: {:5.2f} ".format(file, rad, Bmin, Bmax))

		# for object reference
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

		# recompute L if a scan has been performed to get an overall L
		if options.measure == "grid" and r_intervals > 1 and options.sterimol:
			L, Bmax, Bmin, cyl = sterics.get_cube_sterimol(occ_grid, rad, options.grid, 0.0)
			if not options.quiet:
				print("\n   L parameter is {:5.2f} Ang".format(L))

		if options.sterimol:
			cylinders.append("   CYLINDER, 0., 0., 0., 0., 0., {:5.3f}, 0.1, 1.0, 1.0, 1.0, 0., 0.0, 1.0,".format(L))

		if options.pymol and ext != "rdkit":
			writer.xyz_export(file, mol)
			writer.pymol_export(file, mol, spheres, cylinders, options.isoval, options.visv, options.viss)

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
		"isoval": ["isoval", 0.002],
		"s": ["sterimol", False],
		"sterimol": ["sterimol", False],
		"surface": ["surface", "vdw"],
		"debug": ["debug", False],
		"b": ["volume", False],
		"volume": ["volume", False],
		"vshell": ["vshell", False],
		"pymol": ["pymol", False],
		"quiet": ["quiet", False],
		"sambvca": ["sambvca", False],
		"qsar": ["qsar", False],
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
	parser.add_option("--grid", dest="grid", action="store", help="Grid point spacing in Angstrom (default: 0.1)", default=0.1, type=float, metavar="grid")
	parser.add_option("--gridsize", dest="gridsize", action="store", help="Manual grid dimensions: xmin,xmax:ymin,ymax:zmin,zmax", default=False)
	parser.add_option("--isoval", dest="isoval", action="store", help="Density isovalue cutoff (default: 0.002)", type="float", default=0.002, metavar="isoval")
	parser.add_option("--maxpath", dest="max_path_length", type=int, action="store", default=9, help="[2D sterics] Maximum path length in bonds (default: 9)")
	parser.add_option("--noH", dest="noH", action="store_true", help="Exclude hydrogen atoms from steric measurements", default=False)
	parser.add_option("--nometals", dest="no_metals", action="store_true", help="Exclude metal atoms from steric measurements", default=False)
	parser.add_option("--norot", dest="norot", action="store_true", help="Do not rotate the molecule (use if structures have been pre-aligned)", default=False)
	parser.add_option("--pos", dest="pos", action="store_true", help="Measure Sterimol parameters in positive direction (from atom1 toward atom2)", default=False)
	parser.add_option("--qsar", dest="qsar", action="store_true", help="Generate probe atom grid files for QSAR study", default=False)
	parser.add_option("--quiet", dest="quiet", action="store_true", help="Suppress all print output", default=False)
	parser.add_option("-r", dest="radius", action="store", help="Radius of sphere in Angstrom (default: 3.5)", default=3.5, type=float, metavar="radius")
	parser.add_option("--sambvca", dest="sambvca", action="store_true", help="Use SambVca 2.1 defaults: scale VDW radii by 1.17 and exclude H atoms", default=False)
	parser.add_option("--scalevdw", dest="SCALE_VDW", action="store", help="Scaling factor for VDW radii (default: 1.0)", type=float, default=1.0, metavar="SCALE_VDW")
	parser.add_option("--scan", dest="scan", action="store", help="Scan over a range of radii, format: rmin:rmax:interval", default=False, metavar="scan")
	parser.add_option("-s", "--sterimol", dest="sterimol", action="store_true", help="Compute Sterimol parameters (L, Bmin, Bmax)", default=False)
	parser.add_option("--surface", dest="surface", action="store", choices=["vdw", "density"], help="Surface type: Bondi VDW radii or density cube file (default: vdw)", default="vdw", metavar="surface")
	parser.add_option("-b", "--vbur", dest="volume", action="store_true", help="Calculate buried volume of input molecule", default=False)
	parser.add_option("-v", "--verbose", dest="verbose", action="store_true", help="Print verbose output", default=False)
	parser.add_option("--viss", dest="viss", action="store_true", help="Visualize Sterimol Bmin and Bmax in PyMOL as circle outlines", default=False)
	parser.add_option("--visv", dest="visv", action="store", choices=["circle", "sphere"], help="Visualize volume in PyMOL as circle or sphere (default: circle)", default="circle")
	parser.add_option("--vshell", dest="vshell", action="store", help="Calculate buried volume of hollow sphere with given shell width; use -r to set radius", default=False, type=float, metavar="width")
	(options, args) = parser.parse_args()

	# Sterimol defaults to classic; volume forces grid mode internally
	options.measure = "classic"

	# SambVca mode: scale VDW radii by 1.17 and exclude H atoms
	if options.sambvca:
		options.SCALE_VDW = 1.17
		options.noH = True

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
	# in qsar mode, loop through and get dimensions of molecules to create uniform grid sizing
	# (3A larger than greatest magnitudes in xyz directions)
	if options.qsar:
		mols = []
		for file in files:
			mols.append(dbstep(file, options=options, QSAR=True))
		xmin, xmax, ymin, ymax, zmin, zmax = 0, 0, 0, 0, 0, 0
		for mol in mols:
			[x_minus, x_plus, y_minus, y_plus, z_minus, z_plus] = mol.dimensions
			if x_plus >= xmax:
				xmax = x_plus
			if y_plus >= ymax:
				ymax = y_plus
			if z_plus >= zmax:
				zmax = z_plus
			if x_minus <= xmin:
				xmin = x_minus
			if y_minus <= ymin:
				ymin = y_minus
			if z_minus <= zmin:
				zmin = z_minus
		dim = [xmin, xmax, ymin, ymax, zmin, zmax]
		for i in range(len(dim)):
			if i % 2:
				dim[i] += 3
			else:
				dim[i] -= 3
		options.gridsize = str(dim[0]) + "," + str(dim[1]) + ":" + str(dim[2]) + "," + str(dim[3]) + ":" + str(dim[4]) + "," + str(dim[5])
		if options.verbose:
			print("   Grid size for QSAR mode is: " + options.gridsize)

	# Set file column width based on longest filename
	dbstep._file_col_width = max(len(os.path.basename(f)) for f in files) + 2

	# loop over all specified output files
	for file in files:
		if options.graph:
			try:
				from dbstep import graph
			except ModuleNotFoundError as e:
				print(e, "\nPlease install necessary modules and try again.")
				sys.exit()
			vec_df = graph.mol_to_vec(file, options.shared_fg, options.voltype, options.max_path_length, options.verbose)
			vec_df.to_csv(file.split(".")[0] + "_2d_output.csv", index=False)
		else:
			dbstep(file, options=options)

	if dbstep._column_width and not options.quiet:
		print("   " + "-" * (dbstep._column_width - 3))


if __name__ == "__main__":
	main()
