# -*- coding: UTF-8 -*-
import sys
import numpy as np

from dbstep.constants import bondi, charry_tkatchenko

"""
selection

Reduces a parsed structure to the atoms that matter for a measurement.
Currently: a radial crop around atom1 so that the grid scales with the
measurement sphere rather than with the size of the whole system.
"""


def max_vdw_radius(atomtypes, options):
	"""Largest (scaled) VDW radius among the given atom types, using the same lookup as the grid code."""
	radii_dict = charry_tkatchenko if options.radii == "charry-tkatchenko" else bondi
	if len(atomtypes) == 0:
		return 0.0
	return max(radii_dict.get(atom, 2.0) for atom in atomtypes) * options.SCALE_VDW


def auto_cutoff(options, atomtypes):
	"""Smallest distance from atom1 that still contains every atom able to occupy the buried-volume sphere.

	An atom can only occupy grid points inside a sphere of radius R if its own VDW sphere
	intersects it, i.e. if it lies within R + r_vdw of the centre. The measurement sphere is
	the largest scan radius (or -r), widened by half the shell width for --vshell / scans and by
	the same 10% margin that max_dim uses for the grid; one grid spacing absorbs rounding.
	"""
	r_max, shell = options.radius, 0.0
	if options.scan:
		try:
			_, r_max, shell = [float(s) for s in options.scan.split(":")]
		except (ValueError, AttributeError):
			sys.exit("   Can't read your scan request. Try something like --scan 3:5:0.5")
	if options.vshell:
		shell = max(shell, options.vshell)
	return 1.1 * r_max + 0.5 * shell + max_vdw_radius(atomtypes, options) + options.grid


def resolve_cutoff(options, atomtypes):
	"""Turn the --cutoff option (False/0/'none', 'auto' or a distance) into a distance in Angstrom, or None if off."""
	value = options.cutoff
	if value is False or value is None or value == 0:
		return None
	if isinstance(value, str):
		if value.lower() == "auto":
			return auto_cutoff(options, atomtypes)
		if value.lower() in ("none", "off"):
			return None
		try:
			value = float(value)
		except ValueError:
			sys.exit(f"   Can't read cutoff '{options.cutoff}'. Use a distance in Angstrom or 'auto'.")
	if value <= 0:
		sys.exit(f"   Cutoff must be positive, got {value}.")
	return float(value)


def crop(mol, options, cutoff):
	"""Keep only atoms within `cutoff` of atom1.

	Atom1 is always kept; atom2 and atom3 are kept as well when Sterimol parameters are requested,
	since they define the alignment (for a volume-only run they play no role and would only
	enlarge the grid). Renumbers options.spec_atom_1, options.spec_atom_2 and options.atom3 to
	the kept subset, exactly as DataParser.exclude_atoms does for --noH / --exclude.

	Returns:
		(n_before, n_after) atom counts
	"""
	coords = mol.CARTESIANS
	n_before = len(coords)
	center = coords[options.spec_atom_1 - 1]
	keep = np.linalg.norm(coords - center, axis=1) <= cutoff

	spec = [options.spec_atom_1 - 1]
	if options.sterimol:
		spec += [atom - 1 for atom in options.spec_atom_2]
		if options.atom3:
			spec.append(int(options.atom3) - 1)
	keep[spec] = True

	if keep.all():
		return n_before, n_before

	# new index = old index minus the number of removed atoms that came before it
	removed_before = np.cumsum(~keep) - (~keep)
	options.spec_atom_1 = int(options.spec_atom_1 - removed_before[options.spec_atom_1 - 1])
	options.spec_atom_2 = [int(atom - removed_before[atom - 1]) for atom in options.spec_atom_2]
	if options.atom3:
		atom3 = int(options.atom3)
		options.atom3 = int(atom3 - removed_before[atom3 - 1])

	mol.keep(keep)
	return n_before, int(keep.sum())
