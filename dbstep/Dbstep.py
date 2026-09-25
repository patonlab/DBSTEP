# -*- coding: UTF-8 -*-

# Python Libraries
import copy
import os, sys
from glob import glob
import numpy as np
from optparse import OptionParser

from dbstep import sterics, parse_data, calculator, writer, selection
from dbstep.constants import periodic_table, bondi, charry_tkatchenko, metals

class dbstep:
	"""
	dbstep object that contains coordinates, steric data

	Objects that can currently be referenced are:
			L, Bmax, Bmin,
			occ_vol, bur_vol, bur_shell,
			atom1, atom2 (the reference atom indices as given in the input file),
			cutoff, n_atoms_total, n_atoms_kept (radial crop, see --cutoff),
			residue_label (PDB residue mode, e.g. "A:45 LEU"),
			results (list of dicts, one per radius: file, structure, residue, atom1, atom2, radius,
				mol_vol, percent_vbur, percent_sbur, bmin, bmax, L; also what --csv writes),
			atoms, coords, spec_atoms (the structure actually measured, after selection and crop,
				in input orientation; spec_atoms are 1-indexed into it)

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
		# Tensor Parameters
		self.tensor, self.tensor_grid = False, False

		if "options" in kwargs:
			# Work on a copy: the calculation renumbers spec atoms (--noH/--exclude) and
			# toggles flags, which must not leak into the next file of a multi-file run
			self.options = copy.copy(kwargs["options"])
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
		# tensor mode: force volume (for grid) and sterimol (for alignment)
		if options.tensor:
			if not options.atom3:
				sys.exit("ERROR: --tensor requires --atom3 to fully define the molecular orientation.")
			# default to coarser grid if user didn't explicitly set grid spacing
			if options.grid == 0.05:
				options.grid = 1.0
			options._tensor_only_sterimol = not options.sterimol
			options._tensor_only_volume = not options.volume
			options.volume = True
			options.sterimol = True
		# sterimol scan requires grid-based measurement for per-radius slicing
		if options.sterimol and options.scan and options.measure == "classic":
			options.measure = "grid"

		origin = np.array([0, 0, 0])
		self.residue_label = None
		if options.residue:
			# PDB residue mode: parse everything first, then resolve atom names and apply all filters in one pass
			if ext not in (".pdb", ".ent"):
				sys.exit("   --residue requires PDB input (.pdb/.ent)")
			mol = selection.load_pdb(file, ext, options)
			info = selection.apply_residue_selection(mol, options, verbose=options.verbose)
			self.residue_label = info["label"]
			# user-facing labels are the atom names in residue mode
			self.atom1, self.atom2 = info["atom1_name"], info["atom2_names"]
			if options.cutoff is False:
				options.cutoff = "auto"
			if options.verbose:
				print("   Residue {}: {} atoms; {} atoms removed by filters".format(self.residue_label, info["n_self"], info["n_removed"]))
		else:
			self._get_spec_atoms(options)
			# remember the user-facing (input file) indices: --noH/--exclude renumber options.spec_atom_* internally
			self.atom1, self.atom2 = options.spec_atom_1, list(options.spec_atom_2)
			mol = parse_data.read_input(file, ext, options)
		self._check_num_atoms(mol, file)

		# Radial crop: drop atoms too far from atom1 to influence the measurement,
		# so the grid scales with the sphere rather than with the whole system
		self.cutoff = None
		self.n_atoms_total = self.n_atoms_kept = len(mol.ATOMTYPES)
		if options.cutoff:
			if options.surface == "vdw":
				self.cutoff = selection.resolve_cutoff(options, mol.ATOMTYPES)
				if self.cutoff is not None:
					self.n_atoms_total, self.n_atoms_kept = selection.crop(mol, options, self.cutoff)
					if options.verbose:
						print("   Cutoff {:.2f} Ang around atom1: keeping {} of {} atoms".format(self.cutoff, self.n_atoms_kept, self.n_atoms_total))
			elif not options.quiet:
				print("   Note: --cutoff is ignored for density cube input (the grid comes from the cube file)")
		# the structure actually measured (after selection and crop), before translation/rotation
		self.atoms, self.coords = np.array(mol.ATOMTYPES), np.array(mol.CARTESIANS)
		self.spec_atoms = [options.spec_atom_1] + list(options.spec_atom_2)
		self.metadata = {key: np.array(values) for key, values in getattr(mol, "METADATA", {}).items()}

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
		occ_grid, occ_vol, point_tree, grid_axes, occ_mask = self._build_grid(
			mol, name, options, origin, x_min, x_max, y_min, y_max, z_min, z_max)

		# Store tensor and metadata if requested
		if options.tensor and occ_mask is not None:
			self.tensor = occ_mask.astype(int)
			self.tensor_grid = {
				"origin": [x_min, y_min, z_min],
				"spacing": options.grid,
				"shape": occ_mask.shape,
				"x_vals": grid_axes[0] if grid_axes else None,
				"y_vals": grid_axes[1] if grid_axes else None,
				"z_vals": grid_axes[2] if grid_axes else None,
			}
			if options.save:
				save_base = name if isinstance(name, str) else "tensor"
				np.save(save_base + "_tensor.npy", self.tensor)
				if not options.quiet:
					print("   Tensor saved to {}_tensor.npy (shape: {})".format(save_base, self.tensor.shape))

		# Restore sterimol/volume flags if they were only set for tensor alignment
		if options.tensor:
			if getattr(options, '_tensor_only_sterimol', False):
				options.sterimol = False
			if getattr(options, '_tensor_only_volume', False):
				options.volume = False

		# Print column headers (once across multi-file runs)
		self._print_column_header(options)
		self.results = []

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
			if options.sterimol or options.volume:
				writer.pymol_export(file, mol, spheres, cylinders, options.isoval, options.visv, options.viss)
			if options.tensor and self.tensor is not False:
				writer.tensor_pymol_export(file, mol, self.tensor, self.tensor_grid)

	def _assign_surface(self, mol, file, options, origin):
		"""Assign VDW radii or parse density cube, translate molecule, and remove metals.
		Returns grid bounds (x_min, x_max, y_min, y_max, z_min, z_max)."""
		x_min = x_max = y_min = y_max = z_min = z_max = 0.0

		if options.surface == "vdw":
			# Select radii set based on options
			radii_dict = charry_tkatchenko if options.radii == "charry-tkatchenko" else bondi
			for atom in mol.ATOMTYPES:
				if atom not in periodic_table and atom not in radii_dict:
					sys.exit("\n   UNABLE TO GENERATE VDW RADII FOR ATOM: " + str(atom))
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
				sys.exit("   UNABLE TO READ DENSITY CUBE")
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
			sys.exit("   Requested surface {} is not currently implemented. Try either vdw or density".format(options.surface))

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
			sys.exit("   Can't read your scan request. Try something like --scan 3:5:0.25")

	def _build_grid(self, mol, name, options, origin, x_min, x_max, y_min, y_max, z_min, z_max):
		"""Construct occupancy grid. Returns (occ_grid, occ_vol, point_tree, grid_axes, occ_mask)."""
		grid_axes = None
		occ_grid = occ_vol = point_tree = None
		occ_mask = None

		if options.surface == "vdw":
			# User can override grid dimensions
			if options.gridsize:
				gs = [float(val) for val in options.gridsize.replace(":", ",").split(",")]
				if not options.tensor and (gs[1] < x_max or gs[0] > x_min or gs[3] < y_max or gs[2] > y_min or gs[5] < z_max or gs[4] > z_min):
					sys.exit("ERROR: Your molecule is larger than the gridsize you selected,\n       please try again with a larger gridsize")
				x_min, x_max, y_min, y_max, z_min, z_max = gs

			# Snap lattice coordinates to 8 decimals: linspace endpoints differ between box extents by
			# rounding noise, which would otherwise flip grid points lying exactly on a sphere boundary
			x_vals = np.round(np.linspace(x_min, x_max, int(1 + round((x_max - x_min) / options.grid))), 8)
			y_vals = np.round(np.linspace(y_min, y_max, int(1 + round((y_max - y_min) / options.grid))), 8)
			z_vals = np.round(np.linspace(z_min, z_max, int(1 + round((z_max - z_min) / options.grid))), 8)

			if options.volume or options.measure == "grid":
				if options.volume and not (options.sterimol and options.measure == "grid"):
					# Fast path: skip full grid construction when grid sterimol not needed
					if options.tensor:
						occ_grid, occ_vol, occ_mask = sterics.occupied_direct(mol.CARTESIANS, mol.RADII, origin, x_vals, y_vals, z_vals, options, return_mask=True)
					else:
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

		return occ_grid, occ_vol, point_tree, grid_axes, occ_mask

	def _print_column_header(self, options):
		"""Print column headers once across multi-file runs."""
		if options.quiet or dbstep._column_header_printed:
			return
		fw = dbstep._file_col_width
		# with a cutoff the molecular volume only covers the kept atoms
		vol_label = "MolVol_cut" if self.cutoff is not None else "Mol_Vol"
		if options.volume and options.sterimol:
			header = "   {:>{fw}} {:>6} {:>6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format("File", "Atom1", "Atom2", "R/Å", vol_label, "%V_Bur", "%S_Bur", "Bmin", "Bmax", "L", fw=fw)
		elif options.sterimol:
			header = "   {:>{fw}} {:>6} {:>6} {:>10} {:>10} {:>10}".format("File", "Atom1", "Atom2", "Bmin", "Bmax", "L", fw=fw)
		elif options.volume:
			header = "   {:>{fw}} {:>6} {:>6} {:>10} {:>10} {:>10}".format("File", "Atom", "R/Å", vol_label, "%V_Bur", "%S_Bur", fw=fw)
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

		if isinstance(file, str):
			fname = os.path.basename(file)
		else:
			try:
				fname = file.GetProp("_Name")
			except Exception:
				fname = "rdkit_mol"
		plain_name = fname
		# For multi-structure files, label each structure by its name/comment line
		# (single-structure files keep the filename: comment lines often hold energies etc.)
		structure_idx = getattr(options, 'structure', None)
		structure_label = ""
		if structure_idx is not None:
			if getattr(mol, 'structure_name', None):
				# Extract clean name from comment line (first word, strip extension)
				sname = mol.structure_name.split()[0]
				structure_label = os.path.splitext(sname)[0] if '.' in sname else sname
				fname = structure_label
			else:
				structure_label = str(structure_idx)
				fname = "{}[{}]".format(fname, structure_idx)
		if self.residue_label:
			fname = "{} {}".format(fname, self.residue_label)
		fw = dbstep._file_col_width
		atom2_str = ",".join(str(a) for a in self.atom2)

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
					sys.exit("   Can't use classic Sterimol with the isodensity surface. Use --measure grid or --surface vdw")
				Bmin_list.append(Bmin)
				Bmax_list.append(Bmax)
				if options.pymol:
					cylinders.extend(cyl)

			# Record the result row (also written by --csv)
			self.results.append({
				"file": plain_name,
				"structure": structure_label,
				"residue": self.residue_label or "",
				"atom1": self.atom1,
				"atom2": atom2_str if options.sterimol else "",
				"radius": float(rad) if options.volume else "",
				"mol_vol": occ_vol if options.volume else "",
				"percent_vbur": bur_vol if options.volume else "",
				"percent_sbur": bur_shell if options.volume else "",
				"bmin": Bmin if options.sterimol else "",
				"bmax": Bmax if options.sterimol else "",
				"L": L if options.sterimol else "",
			})

			# Tabulate result
			dp = options.dp
			vfmt = "{{:10.{}f}}".format(dp)
			rfmt = "{{:6.{}f}}".format(dp)
			if options.volume and options.sterimol:
				if options.pymol:
					spheres.append("   SPHERE, 0.000, 0.000, 0.000, {:5.3f},".format(rad))
				if not options.quiet:
					fmt = "   {:>" + str(fw) + "} {:>6} {:>6} " + rfmt + " " + " ".join([vfmt] * 6)
					print(fmt.format(fname, self.atom1, atom2_str, rad, occ_vol, bur_vol, bur_shell, Bmin, Bmax, L))
			elif options.volume:
				if options.pymol:
					spheres.append("   SPHERE, 0.000, 0.000, 0.000, {:5.3f},".format(rad))
				if not options.quiet:
					fmt = "   {:>" + str(fw) + "} {:>6} " + rfmt + " " + " ".join([vfmt] * 3)
					print(fmt.format(fname, self.atom1, rad, occ_vol, bur_vol, bur_shell))
			elif options.sterimol:
				if not options.quiet:
					fmt = "   {:>" + str(fw) + "} {:>6} {:>6} " + " ".join([vfmt] * 3)
					print(fmt.format(fname, self.atom1, atom2_str, Bmin, Bmax, L))

		# Store results on self
		if occ_vol is not None:
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
				raise type(atom1_exception)(f"{options.spec_atom_1} is not a valid input for atom1. Please enter a positive integer index.") from atom1_exception
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
				raise type(atom2_error)(f"{options.spec_atom_2} is not a valid input for atom2. Valid inputs are: \n\tAn int, comma separated ints, or a python list of ints") from atom2_error
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
		"grid": ["grid", 0.05],
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
		"tensor": ["tensor", False],
		"t": ["tensor", False],
		"save": ["save", False],
		"vshell": ["vshell", False],
		"pymol": ["pymol", False],
		"quiet": ["quiet", False],
		"radii": ["radii", "bondi"],
		"sambvca": ["sambvca", False],
		"gridsize": ["gridsize", False],
		"cutoff": ["cutoff", False],
		"residue": ["residue", False],
		"atom": ["atom", False],
		"nowater": ["nowater", False],
		"nohet": ["nohet", False],
		"exclude_self": ["exclude_self", False],
		"self_only": ["self_only", False],
		"chain": ["chain", False],
		"csv": ["csv", False],
		"measure": ["measure", "classic"],
		"pos": ["pos", False],
		"dp": ["dp", 2],
		"graph": ["graph", False],
		"fg": ["shared_fg", False],
		"shared_fg": ["shared_fg", False],
		"maxpath": ["max_path_length", 9],
		"max_path_length": ["max_path_length", 9],
		"voltype": ["voltype", "crippen"],
		"visv": ["visv", "circle"],
		"viss": ["viss", False],
		"structure": ["structure", None],
	}

	for key in var_dict:
		vars(options)[var_dict[key][0]] = var_dict[key][1]
	for key in kwargs:
		if key in var_dict:
			vars(options)[var_dict[key][0]] = kwargs[key]
		else:
			print("Warning! Option: [", key, ":", kwargs[key], "] provided but no option exists, try -h to see available options.")

	return options


def all_residues(file, **kwargs):
	"""Run one calculation per residue of a PDB file (the Python side of --residue all).

	Residues are the polymer residues (ATOM records, plus HETATM residues with a peptide backbone
	such as MSE) that contain the atom1 name (default CA); waters and other hetero groups are
	skipped; --chain restricts the chains. Accepts the same keyword arguments as dbstep(), or
	options=<options object>.

	Returns:
		list of dbstep objects in file order
	"""
	options = kwargs["options"] if "options" in kwargs else set_options(kwargs)
	_, ext = os.path.splitext(file)
	if ext not in (".pdb", ".ent"):
		sys.exit("   --residue all requires PDB input (.pdb/.ent)")
	mol = selection.load_pdb(file, ext, options)
	residues = selection.list_residues(mol, options)
	if not residues:
		sys.exit("   No residues containing atom {} found in {}".format(options.atom or "CA", file))
	previous = options.residue
	runs = []
	try:
		for resid in residues:
			options.residue = resid
			runs.append(dbstep(file, options=options))
	finally:
		options.residue = previous
	return runs


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
	parser.add_option("--residue", dest="residue", action="store", help="[PDB] Residue to measure, e.g. A:45 (chain:number, insertion code appended) or 45; several separated by commas form one selection", default=False, metavar="residue")
	parser.add_option("--atom", dest="atom", action="store", help="[PDB] Name of atom1 within the residue (default: CA); --atom2/--atom3 also accept atom names in residue mode (default atom2: CB)", default=False, metavar="name")
	parser.add_option("--nowater", dest="nowater", action="store_true", help="[PDB] Exclude water molecules", default=False)
	parser.add_option("--nohet", dest="nohet", action="store_true", help="[PDB] Exclude hetero groups (ligands, ions) other than waters, the selected residue and modified polymer residues", default=False)
	parser.add_option("--chain", dest="chain", action="store", help="[PDB] Keep only this chain (plus the selected residue)", default=False, metavar="chain")
	parser.add_option("--exclude-self", dest="exclude_self", action="store_true", help="[PDB] The selected residue occupies no volume: measure only its environment", default=False)
	parser.add_option("--self-only", dest="self_only", action="store_true", help="[PDB] Keep only the selected residue (same as measuring it extracted to its own file)", default=False)
	parser.add_option("--csv", dest="csv", action="store", help="Write all result rows (one per file/structure/residue/radius) to this CSV file", default=False, metavar="file")
	parser.add_option("--cutoff", dest="cutoff", action="store", help="Ignore atoms farther than this distance (Angstrom) from atom1; 'auto' keeps exactly the atoms that can occupy the buried-volume sphere. Keeps the grid small for large systems (default: off)", default=False, metavar="cutoff")
	parser.add_option("--isoval", dest="isoval", action="store", help="Density isovalue cutoff (default: 0.0016)", type="float", default=0.0016, metavar="isoval")
	parser.add_option("--maxpath", dest="max_path_length", type=int, action="store", default=9, help="[2D sterics] Maximum path length in bonds (default: 9)")
	parser.add_option("--noH", dest="noH", action="store_true", help="Exclude hydrogen atoms from steric measurements", default=False)
	parser.add_option("--nometals", dest="no_metals", action="store_true", help="Exclude metal atoms from steric measurements", default=False)
	parser.add_option("--norot", dest="norot", action="store_true", help="Do not rotate the molecule (use if structures have been pre-aligned)", default=False)
	parser.add_option("--dp", dest="dp", action="store", type="int", help="Number of decimal places for output values (default: 2)", default=2, metavar="dp")
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
	parser.add_option("-t", "--tensor", dest="tensor", action="store_true", help="Return 3D binary occupancy tensor (requires --atom1, --atom2, --atom3)", default=False)
	parser.add_option("--save", dest="save", action="store_true", help="Save tensor to .npy file (use with --tensor)", default=False)
	parser.add_option("-b", "--vbur", dest="volume", action="store_true", help="Calculate buried volume of input molecule", default=False)
	parser.add_option("-v", "--verbose", dest="verbose", action="store_true", help="Print verbose output", default=False)
	parser.add_option("--viss", dest="viss", action="store_true", help="Visualize Sterimol Bmin and Bmax in PyMOL as circle outlines", default=False)
	parser.add_option("--visv", dest="visv", action="store", choices=["circle", "sphere"], help="Visualize volume in PyMOL as circle or sphere (default: circle)", default="circle")
	parser.add_option("--vshell", dest="vshell", action="store", help="Calculate buried volume of hollow sphere with given shell width; use -r to set radius", default=False, type=float, metavar="width")
	(options, args) = parser.parse_args()
	# index of the structure within a multi-structure file (set per structure in the loop below)
	options.structure = None

	# SambVca mode: Bondi radii scaled by 1.17, exclude H atoms
	if options.sambvca:
		options.radii = "bondi"
		options.SCALE_VDW = 1.17
		options.noH = True

	# Tensor mode: default to coarser grid spacing if user didn't set --grid
	if options.tensor and options.grid == 0.05:
		options.grid = 1.0

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
	if options.residue:
		dbstep._file_col_width += (12 if str(options.residue).lower() == "all" else len(str(options.residue)) + 5)  # room for "A:45 LEU"

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
				print("   Hydrogen atoms are {}".format("excluded" if options.noH else "included"))
				if options.residue:
					filters = [name for flag, name in ((options.nowater, "waters"), (options.nohet, "hetero groups"), (options.exclude_self, "the residue itself")) if flag]
					print("   PDB residue mode: measuring residue {}{}".format(options.residue, ", excluding " + " and ".join(filters) if filters else ""))
					if options.self_only:
						print("   Only the selected residue is kept (--self-only)")
				if options.cutoff or options.residue:
					cutoff = options.cutoff if options.cutoff else "auto"
					print("   Atoms farther than {} from atom1 are ignored (--cutoff)".format("the auto cutoff" if str(cutoff).lower() == "auto" else "{} Angstrom".format(cutoff)))
				print("")
			else:
				print("   Using {} isodensity surface with cutoff value of {:5.4f} au".format(options.surface, options.isoval))
				print("   Cartesian grid-spacing will be determined by cube file(s)\n")

	# loop over all specified output files
	runs = []
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
			vec_df.to_csv(os.path.splitext(file)[0] + "_2d_output.csv", index=False)
		else:
			# Multi-structure files (multi-xyz, multi-sdf, multi-MODEL pdb): one run per structure
			_, ext = os.path.splitext(file)
			counters = {".xyz": parse_data.get_xyz_structures, ".sdf": parse_data.get_sdf_structures, ".mol": parse_data.get_sdf_structures,
						".pdb": parse_data.get_pdb_models, ".ent": parse_data.get_pdb_models}
			n_structures = len(counters[ext](file)) if ext in counters else 1
			for idx in (range(n_structures) if n_structures > 1 else [None]):
				options.structure = idx
				if options.residue and str(options.residue).lower() == "all":
					runs.extend(all_residues(file, options=options))
				else:
					runs.append(dbstep(file, options=options))
			options.structure = None

	if dbstep._column_width and not options.quiet:
		print("   " + "-" * (dbstep._column_width - 3))

	if options.csv and runs:
		writer.csv_export(options.csv, [row for run in runs for row in run.results])
		if not options.quiet:
			print("\n   Results written to {}".format(options.csv))


if __name__ == "__main__":
	main()
