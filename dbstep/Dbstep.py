# -*- coding: UTF-8 -*-

# Python Libraries
import copy
import os, sys
import argparse
from glob import glob
import numpy as np

from dbstep import sterics, parse_data, calculator, writer, selection, trajectory, ensemble
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
			results (list of dicts, one per radius: file, frame, structure, residue, atom1, atom2,
				radius, mol_vol, percent_vbur, percent_sbur, bmin, bmax, L; also what --csv writes),
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
		# Per-residue contributions to %V_bur (--decompose): dict label -> percent, or a list of dicts for scans
		self.contributions = None
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
		# --nometals: drop metal atoms through the same ghost/renumber path as --noH, so spec atoms
		# and per-atom metadata stay consistent (a metal chosen as atom1 becomes a zero-radius ghost)
		if options.no_metals and options.surface == "vdw":
			is_metal = np.isin(mol.ATOMTYPES, list(metals))
			if is_metal.any():
				spec = [options.spec_atom_1] + list(options.spec_atom_2) + ([int(options.atom3)] if options.atom3 else [])
				new_spec = mol.exclude_mask(is_metal, spec)
				options.spec_atom_1, options.spec_atom_2 = new_spec[0], new_spec[1:1 + len(options.spec_atom_2)]
				if options.atom3:
					options.atom3 = new_spec[-1]
				if options.verbose:
					print("   Excluded {} metal atom(s) (--nometals)".format(int(is_metal.sum())))
		# the structure actually measured (after selection and crop), before translation/rotation
		self.atoms, self.coords = np.array(mol.ATOMTYPES), np.array(mol.CARTESIANS)
		self.spec_atoms = [options.spec_atom_1] + list(options.spec_atom_2)
		self.metadata = {key: np.array(values) for key, values in getattr(mol, "METADATA", {}).items()}
		# per-structure properties (SDF data fields such as <Energy>, xyz comment line) for ensemble weighting
		self.properties = dict(getattr(mol, "PROPERTIES", {}))
		self.structure_name = getattr(mol, "structure_name", None)
		self.population = None

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
				structure_idx = getattr(options, "structure", None)
				if structure_idx is not None:
					save_base += "_frame{}".format(structure_idx)
				if self.residue_label:
					save_base += "_" + self.residue_label.replace(":", "").replace(" ", "_")
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
					grid_axes = (x_vals, y_vals, z_vals)

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
		Bmin_list, Bmax_list, bur_vol_list, bur_shell_list, contributions_list = [], [], [], [], []

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
			if getattr(mol, 'structure_name', None) and getattr(mol, 'FORMAT', '') in ('sdf', 'mol'):
				# SDF titles identify the record (e.g. "ether 44" from a conformer search): keep them whole,
				# only stripping a file extension from single-word titles such as "35diMePh.log"
				structure_label = mol.structure_name.strip()
				if len(structure_label.split()) == 1 and "." in structure_label:
					structure_label = os.path.splitext(structure_label)[0]
				fname = structure_label
			elif getattr(mol, 'structure_name', None):
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
					if options.decompose:
						contributions_list.append({})
				else:
					if options.vshell:
						strip_width = options.vshell
					bur_vol, bur_shell = sterics.buried_vol(occ_grid, point_tree, origin, rad, strip_width, options, occ_dist2=occ_dist2, grid_axes=grid_axes)
					if options.decompose:
						contributions_list.append(self._decompose(mol, options, grid_axes, origin, rad if strip_width != 0.0 else options.radius))
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
				"path": file if isinstance(file, str) else "",
				"frame": structure_idx if structure_idx is not None else "",
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
				if contributions_list:
					self.contributions = contributions_list[0]
		else:
			if options.sterimol:
				self.Bmax = Bmax_list
				self.Bmin = Bmin_list
			if options.volume:
				self.bur_vol = bur_vol_list
				self.bur_shell = bur_shell_list
				if contributions_list:
					self.contributions = contributions_list

		return spheres, cylinders

	def _decompose(self, mol, options, grid_axes, origin, R):
		"""Per-residue contributions to %V_bur (--decompose), as a dict "A:45 LEU" -> percent."""
		if options.surface != "vdw" or grid_axes is None:
			sys.exit("   --decompose works with VDW surfaces only (not density cubes)")
		if not self.metadata or "resid" not in self.metadata:
			sys.exit("   --decompose needs residue information: use PDB input")
		labels = ["{} {}".format(resid, resname) for resid, resname in zip(self.metadata["resid"], self.metadata["resname"])]
		return sterics.buried_vol_by_group(mol.CARTESIANS, mol.RADII, labels, *grid_axes, origin, R)

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
		"decompose": ["decompose", False],
		"csv": ["csv", False],
		"frames": ["frames", False],
		"boltzmann": ["boltzmann", False],
		"temperature": ["temperature", 298.15],
		"energy_units": ["energy_units", "kcal"],
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


def contribution_rows(run):
	"""Per-residue contribution rows of a run (one per residue and radius), for the contributions CSV."""
	if not run.contributions:
		return []
	per_radius = run.contributions if isinstance(run.contributions, list) else [run.contributions]
	rows = []
	for result, contributions in zip(run.results, per_radius):
		for label, percent in contributions.items():
			rows.append({"file": result["file"], "frame": result["frame"], "structure": result["structure"], "residue": result["residue"],
						 "radius": result["radius"], "contributor": label, "percent_vbur": percent, "path": result.get("path", "")})
	return rows


def contribution_lines(run, dp=2):
	"""Printable per-residue contribution block for a run."""
	if not run.contributions:
		return []
	per_radius = run.contributions if isinstance(run.contributions, list) else [run.contributions]
	lines = []
	for result, contributions in zip(run.results, per_radius):
		label = " ".join(str(part) for part in (result["file"], result["structure"], result["residue"]) if part)
		lines.append("\n   %V_Bur contributions for {} (R = {:.2f} Ang, total {:.{dp}f}%):".format(label, result["radius"], result["percent_vbur"], dp=dp))
		for name, percent in sorted(contributions.items(), key=lambda item: -item[1]):
			if percent > 0:
				lines.append("      {:<16} {:>{w}.{dp}f}".format(name, percent, w=6 + dp, dp=dp))
	return lines


def format_result_row(row, options, fw):
	"""Format a result row (as in dbstep.results) the way the results table prints it."""
	dp = options.dp
	vfmt = "{{:10.{}f}}".format(dp)
	rfmt = "{{:6.{}f}}".format(dp)
	label = " ".join(str(part) for part in (row["file"], row["structure"], row["residue"]) if part)
	if options.volume and options.sterimol:
		fmt = "   {:>" + str(fw) + "} {:>6} {:>6} " + rfmt + " " + " ".join([vfmt] * 6)
		return fmt.format(label, row["atom1"], row["atom2"], row["radius"], row["mol_vol"], row["percent_vbur"], row["percent_sbur"], row["bmin"], row["bmax"], row["L"])
	if options.volume:
		fmt = "   {:>" + str(fw) + "} {:>6} " + rfmt + " " + " ".join([vfmt] * 3)
		return fmt.format(label, row["atom1"], row["radius"], row["mol_vol"], row["percent_vbur"], row["percent_sbur"])
	fmt = "   {:>" + str(fw) + "} {:>6} {:>6} " + " ".join([vfmt] * 3)
	return fmt.format(label, row["atom1"], row["atom2"], row["bmin"], row["bmax"], row["L"])


def from_rdkit(mol, **kwargs):
	"""Measure an RDKit molecule that carries a 3D conformer. Accepts the same keyword arguments as dbstep().

	Example:
		mol = Chem.AddHs(Chem.MolFromSmiles("CC(C)C")); AllChem.EmbedMolecule(mol)
		result = from_rdkit(mol, atom1=1, atom2=2, sterimol=True, volume=True)
	"""
	if isinstance(mol, str):
		sys.exit("   from_rdkit() expects an RDKit Mol object; pass file names to dbstep() instead")
	if not hasattr(mol, "GetNumConformers") or mol.GetNumConformers() == 0:
		sys.exit("   The RDKit molecule has no 3D conformer (embed it first, e.g. AllChem.EmbedMolecule)")
	return dbstep(mol, **kwargs)


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


def run_file(file, options):
	"""All runs for one input file: every selected frame (see --frames) times every residue for --residue all.

	Returns:
		list of dbstep objects in frame, then residue, order
	"""
	runs = []
	previous = options.structure
	try:
		for frame in trajectory.frame_indices(file, options):
			options.structure = frame
			if options.residue and str(options.residue).lower() == "all":
				runs.extend(all_residues(file, options=options))
			else:
				runs.append(dbstep(file, options=options))
	finally:
		options.structure = previous
	return runs


def all_frames(file, frames=None, **kwargs):
	"""Run every selected frame of a multi-structure file (multi-frame xyz, multi-record sdf, multi-MODEL pdb).

	`frames` follows --frames ('start:stop:stride', 0-based Python slice rules, or a list of indices;
	default all). Accepts the same keyword arguments as dbstep(), including residue="all".

	Returns:
		list of dbstep objects, one per frame (times residues for residue="all")
	"""
	options = kwargs["options"] if "options" in kwargs else set_options(kwargs)
	previous = getattr(options, "frames", False)
	if frames is not None:
		options.frames = frames
	try:
		return run_file(file, options)
	finally:
		options.frames = previous


def build_parser():
	"""Command line parser for the dbstep entry point."""
	parser = argparse.ArgumentParser(
		prog="dbstep",
		usage="%(prog)s [options] file1 [file2 ...]",
		description="DBSTEP: DFT-based steric parameters. Computes Sterimol parameters (L, Bmin, Bmax), percent buried volume, Sterimol2Vec and Vol2Vec descriptors from structure files.",
		allow_abbrev=False,
	)
	parser.add_argument("files", nargs="*", metavar="file", help="Input structure file(s): xyz, sdf/mol, pdb, Gaussian com/gjf or cube, or any cclib-supported output; shell wildcards are expanded")

	measure = parser.add_argument_group("What to compute")
	measure.add_argument("-s", "--sterimol", dest="sterimol", action="store_true", default=False, help="Compute Sterimol parameters (L, Bmin, Bmax)")
	measure.add_argument("-b", "--vbur", dest="volume", action="store_true", default=False, help="Calculate buried volume of input molecule")
	measure.add_argument("-r", dest="radius", type=float, default=3.5, metavar="radius", help="Radius of sphere in Angstrom (default: 3.5)")
	measure.add_argument("--scan", dest="scan", default=False, metavar="scan", help="Scan over a range of radii, format: rmin:rmax:interval")
	measure.add_argument("--vshell", dest="vshell", type=float, default=False, metavar="width", help="Calculate buried volume of hollow sphere with given shell width; use -r to set radius")
	measure.add_argument("--measure", dest="measure", choices=["classic", "grid"], default="classic", metavar="measure", help="Sterimol method: classic (Verloop, default) or grid-based")
	measure.add_argument("--pos", dest="pos", action="store_true", default=False, help="Measure Sterimol parameters in positive direction (from atom1 toward atom2)")
	measure.add_argument("-t", "--tensor", dest="tensor", action="store_true", default=False, help="Return 3D binary occupancy tensor (requires --atom1, --atom2, --atom3)")
	measure.add_argument("--save", dest="save", action="store_true", default=False, help="Save tensor to .npy file (use with --tensor)")

	atoms = parser.add_argument_group("Reference atoms and atom selection")
	atoms.add_argument("--atom1", dest="spec_atom_1", default=False, metavar="spec_atom_1", help="Specify the base atom number (default: 1)")
	atoms.add_argument("--atom2", dest="spec_atom_2", default=False, metavar="spec_atom_2", help="Specify the connected atom(s) number(s), e.g. 3 or 3,4 (default: 2)")
	atoms.add_argument("--atom3", dest="atom3", default=False, help="Align a third atom to the positive x direction")
	atoms.add_argument("--exclude", dest="exclude", default=False, metavar="exclude", help="Atom indices to ignore, comma-separated with no spaces")
	atoms.add_argument("--noH", dest="noH", action="store_true", default=False, help="Exclude hydrogen atoms from steric measurements")
	atoms.add_argument("--nometals", dest="no_metals", action="store_true", default=False, help="Exclude metal atoms from steric measurements")
	atoms.add_argument("--norot", dest="norot", action="store_true", default=False, help="Do not rotate the molecule (use if structures have been pre-aligned)")
	atoms.add_argument("--cutoff", dest="cutoff", default=False, metavar="cutoff", help="Ignore atoms farther than this distance (Angstrom) from atom1; 'auto' keeps exactly the atoms that can occupy the buried-volume sphere. Keeps the grid small for large systems (default: off)")

	protein = parser.add_argument_group("Proteins (PDB input)")
	protein.add_argument("--residue", dest="residue", default=False, metavar="residue", help="Residue to measure, e.g. A:45 (chain:number, insertion code appended) or 45; several separated by commas form one selection; 'all' runs every residue")
	protein.add_argument("--atom", dest="atom", default=False, metavar="name", help="Name of atom1 within the residue (default: CA); --atom2/--atom3 also accept atom names in residue mode (default atom2: CB)")
	protein.add_argument("--nowater", dest="nowater", action="store_true", default=False, help="Exclude water molecules")
	protein.add_argument("--nohet", dest="nohet", action="store_true", default=False, help="Exclude hetero groups (ligands, ions) other than waters, the selected residue and modified polymer residues")
	protein.add_argument("--chain", dest="chain", default=False, metavar="chain", help="Keep only this chain (plus the selected residue)")
	protein.add_argument("--exclude-self", dest="exclude_self", action="store_true", default=False, help="The selected residue occupies no volume: measure only its environment")
	protein.add_argument("--decompose", dest="decompose", action="store_true", default=False, help="Split %%V_bur between the residues that occupy the sphere (overlaps shared equally); printed after the table and written to <csv>_contributions.csv with --csv")
	protein.add_argument("--self-only", dest="self_only", action="store_true", default=False, help="Keep only the selected residue (same as measuring it extracted to its own file)")

	frames = parser.add_argument_group("Trajectories and output")
	frames.add_argument("--frames", dest="frames", default=False, metavar="frames", help="Frames of a multi-structure file to run, 0-based with Python slice rules: start:stop:stride, e.g. 0:1000:10, ::5, or a single index (default: all)")
	frames.add_argument("--boltzmann", dest="boltzmann", nargs="?", const="auto", default=False, metavar="TAG", help="Boltzmann-average the results over the structures of each multi-structure file (conformer ensembles). Energies come from an SDF data field (auto-detected: Energy, E, G, dG, ... or give the tag) or from a number in the xyz comment line")
	frames.add_argument("--temperature", dest="temperature", type=float, default=298.15, metavar="K", help="Temperature for Boltzmann weighting in K (default: 298.15)")
	frames.add_argument("--energy-units", dest="energy_units", type=str.lower, choices=["kcal", "kj", "hartree", "ev"], default="kcal", help="Units of the energies used for Boltzmann weighting (default: kcal, i.e. kcal/mol)")
	frames.add_argument("--csv", dest="csv", default=False, metavar="file", help="Write all result rows (one per file/frame/residue/radius) to this CSV file")
	frames.add_argument("--dp", dest="dp", type=int, default=2, metavar="dp", help="Number of decimal places for output values (default: 2)")
	frames.add_argument("--pymol", dest="pymol", action="store_true", default=False, help="Write PyMOL visualization and xyz output files")
	frames.add_argument("--visv", dest="visv", choices=["circle", "sphere"], default="circle", help="Visualize volume in PyMOL as circle or sphere (default: circle)")
	frames.add_argument("--viss", dest="viss", action="store_true", default=False, help="Visualize Sterimol Bmin and Bmax in PyMOL as circle outlines")
	frames.add_argument("-v", "--verbose", dest="verbose", action="store_true", default=False, help="Print verbose output")
	frames.add_argument("--quiet", dest="quiet", action="store_true", default=False, help="Suppress all print output")
	frames.add_argument("--debug", dest="debug", action="store_true", default=False, help="Debug mode: graph grid points, print extra information")

	surface = parser.add_argument_group("Surface and grid")
	surface.add_argument("--radii", dest="radii", choices=["bondi", "charry-tkatchenko"], default="bondi", help="VDW radii set: bondi or charry-tkatchenko (default: bondi)")
	surface.add_argument("--scalevdw", dest="SCALE_VDW", type=float, default=1.0, metavar="SCALE_VDW", help="Scaling factor for VDW radii (default: 1.0)")
	surface.add_argument("--sambvca", dest="sambvca", action="store_true", default=False, help="Use SambVca 2.1 defaults: scale VDW radii by 1.17 and exclude H atoms")
	surface.add_argument("--surface", dest="surface", choices=["vdw", "density"], default="vdw", metavar="surface", help="Surface type: Bondi VDW radii or density cube file (default: vdw)")
	surface.add_argument("--isoval", dest="isoval", type=float, default=0.0016, metavar="isoval", help="Density isovalue cutoff (default: 0.0016)")
	surface.add_argument("--grid", dest="grid", type=float, default=0.05, metavar="grid", help="Grid point spacing in Angstrom (default: 0.05)")
	surface.add_argument("--gridsize", dest="gridsize", default=False, help="Manual grid dimensions: xmin,xmax:ymin,ymax:zmin,zmax")

	graph = parser.add_argument_group("2D graph-based sterics (requires RDKit and pandas)")
	graph.add_argument("--2d", dest="graph", action="store_true", default=False, help="Analyze 2D steric contributions from SMILES input")
	graph.add_argument("--2d-type", dest="voltype", choices=["crippen", "mcgowan", "degree"], default="crippen", help="Atomic volume method: crippen, mcgowan, or degree (default: crippen)")
	graph.add_argument("--fg", dest="shared_fg", default=False, help="SMILES pattern of shared functional group to define the origin, e.g. 'C(O)=O'")
	graph.add_argument("--maxpath", dest="max_path_length", type=int, default=9, help="Maximum path length in bonds (default: 9)")
	return parser


def main(argv=None):
	parser = build_parser()
	options = parser.parse_args(argv)
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

	# Expand input files (shell wildcards that reached us unexpanded are globbed here)
	files = []
	for pattern in options.files:
		matches = sorted(glob(pattern)) if any(ch in pattern for ch in "*?[") else ([pattern] if os.path.exists(pattern) else [])
		if not matches:
			parser.error("input file not found: {}".format(pattern))
		files.extend(matches)
	if not files:
		parser.error("please specify at least one input file")

	# Auto-detect density surface from cube file input
	if any(f.endswith(".cube") for f in files):
		options.surface = "density"

	# Set file column width based on longest filename
	dbstep._file_col_width = max(len(os.path.basename(f)) for f in files) + 2
	if options.residue:
		dbstep._file_col_width += (12 if str(options.residue).lower() == "all" else len(str(options.residue)) + 5)  # room for "A:45 LEU"
	if options.boltzmann:
		dbstep._file_col_width += len(" boltzmann")  # the summary row is labelled "<file> boltzmann"

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
				if options.frames:
					print("   Frames {} of each multi-structure file will be run".format(options.frames))
				if options.cutoff or options.residue:
					cutoff = options.cutoff if options.cutoff else "auto"
					print("   Atoms farther than {} from atom1 are ignored (--cutoff)".format("the auto cutoff" if str(cutoff).lower() == "auto" else "{} Angstrom".format(cutoff)))
				print("")
			else:
				print("   Using {} isodensity surface with cutoff value of {:5.4f} au".format(options.surface, options.isoval))
				print("   Cartesian grid-spacing will be determined by cube file(s)\n")

	# loop over all specified output files
	runs, summary_rows, notes = [], [], []
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
			# Multi-structure files (multi-xyz, multi-sdf, multi-MODEL pdb): one run per selected frame
			file_runs = run_file(file, options)
			runs.extend(file_runs)
			if options.boltzmann:
				summary = ensemble.boltzmann_average(file_runs, tag=options.boltzmann, temperature=options.temperature, units=options.energy_units)
				summary_rows.extend(summary)
				if not options.quiet:
					for row in summary:
						print(format_result_row(row, options, dbstep._file_col_width))
					if options.verbose or len(file_runs) <= 12:
						populations = ", ".join("{} {:.3f}".format(r.structure_name or r.results[0]["file"], r.population) for r in file_runs)
						notes.append("   Boltzmann populations at {:.2f} K ({}): {}".format(options.temperature, ensemble.unit_label(options.energy_units), populations))

	if dbstep._column_width and not options.quiet:
		print("   " + "-" * (dbstep._column_width - 3))
	for note in notes:
		print(note)

	if options.decompose and not options.quiet:
		for run in runs:
			for line in contribution_lines(run, options.dp):
				print(line)

	if options.csv and runs:
		writer.csv_export(options.csv, [row for run in runs for row in run.results] + summary_rows)
		if not options.quiet:
			print("\n   Results written to {}".format(options.csv))
		if options.decompose:
			path = os.path.splitext(options.csv)[0] + "_contributions.csv"
			writer.contributions_csv_export(path, [row for run in runs for row in contribution_rows(run)])
			if not options.quiet:
				print("   Residue contributions written to {}".format(path))


if __name__ == "__main__":
	main()
