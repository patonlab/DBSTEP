# -*- coding: UTF-8 -*-
import sys
import numpy as np

from dbstep.constants import bondi, charry_tkatchenko, WATER_RESNAMES

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


# ---------------------------------------------------------------------------------------------------
# Residue selection for PDB input
# ---------------------------------------------------------------------------------------------------

# default atom2 candidates for the Sterimol axis from CA, in order of preference
_DEFAULT_ATOM2 = ["CB", "HA", "HA2", "HA3", "N"]


def _require_metadata(mol, what):
	if not getattr(mol, "METADATA", None) or "resid" not in mol.METADATA:
		sys.exit(f"   {what} requires PDB input with residue information (.pdb/.ent)")


def parse_residue_spec(spec):
	"""Split a --residue value into tokens: 'A:45', 'A:45A' (insertion code) or '45' (any chain)."""
	if isinstance(spec, (list, tuple)):
		tokens = [str(t).strip() for t in spec]
	else:
		tokens = [t.strip() for t in str(spec).split(",")]
	tokens = [t for t in tokens if t]
	if not tokens:
		sys.exit("   --residue needs at least one residue, e.g. --residue A:45")
	return tokens


def residue_mask(mol, spec):
	"""Boolean mask of the atoms belonging to the residues named in `spec` (see parse_residue_spec)."""
	_require_metadata(mol, "--residue")
	meta = mol.METADATA
	mask = np.zeros(len(mol.ATOMTYPES), dtype=bool)
	for token in parse_residue_spec(spec):
		if token.lower() == "all":
			sys.exit("   'all' loops over residues: use --residue all on the command line or dbstep.Dbstep.all_residues() from Python")
		if ":" in token:
			hits = np.char.upper(meta["resid"].astype(str)) == token.upper()
		else:
			# residue number (with optional insertion code) in any chain
			number = token.upper()
			hits = np.array([r.split(":", 1)[1] == number for r in meta["resid"]])
			chains = sorted(set(meta["chain"][hits]))
			if len(chains) > 1:
				sys.exit(f"   Residue {token} exists in chains {', '.join(chains)}; specify one, e.g. --residue {chains[0]}:{token}")
		if not hits.any():
			available = sorted(set(meta["resid"]), key=lambda r: (r.split(":")[0], int("".join(ch for ch in r.split(":")[1] if ch.isdigit() or ch == "-") or 0)))
			preview = ", ".join(available[:8]) + (", ..." if len(available) > 8 else "")
			sys.exit(f"   Residue {token} not found. Residues in the file: {preview}")
		mask |= hits
	return mask


def find_atom_by_name(mol, mask, name):
	"""1-indexed position of the atom called `name` (case-insensitive) among the atoms selected by `mask`."""
	names = np.char.upper(mol.METADATA["name"].astype(str))
	hits = np.where(mask & (names == str(name).strip().upper()))[0]
	if len(hits) == 0:
		return None
	return int(hits[0]) + 1


def resolve_spec_atom(mol, mask, value, default_names, what, avoid=None):
	"""Turn an atom given by name ('CA') or by 1-indexed file position ('12'/12) into a 1-indexed position.

	Names are looked up within the selected residue(s). With `value` unset, the first of
	`default_names` present in the residue (and different from the 1-indexed atom `avoid`) is used.
	Returns (index, name_used).
	"""
	if value not in (False, None, ""):
		try:
			index = int(value)
		except (TypeError, ValueError):
			index = find_atom_by_name(mol, mask, value)
			if index is None:
				present = ", ".join(dict.fromkeys(mol.METADATA["name"][mask]))
				sys.exit(f"   No atom named {value} in the selected residue for {what}. Atoms present: {present}")
			return index, str(value).strip().upper()
		if index <= 0 or index > len(mol.ATOMTYPES):
			sys.exit(f"   {index} is not a valid atom index for {what} (file has {len(mol.ATOMTYPES)} atoms)")
		return index, mol.METADATA["name"][index - 1]
	for name in default_names:
		index = find_atom_by_name(mol, mask, name)
		if index is not None and index != avoid:
			return index, name
	present = ", ".join(dict.fromkeys(mol.METADATA["name"][mask]))
	sys.exit(f"   Could not pick a default atom for {what} (tried {', '.join(default_names)}). Atoms present: {present}")


def polymer_hetatm_mask(mol):
	"""HETATM residues that carry a peptide backbone (N, CA, C, O), e.g. MSE: treated as part of the chain, not as hetero groups."""
	meta = mol.METADATA
	het = meta["record"] == "HETATM"
	mask = np.zeros(len(het), dtype=bool)
	if not het.any():
		return mask
	names = np.char.upper(meta["name"].astype(str))
	for resid in set(meta["resid"][het]):
		atoms = het & (meta["resid"] == resid)
		if {"N", "CA", "C", "O"} <= set(names[atoms]):
			mask |= atoms
	return mask


def water_mask(mol):
	return np.isin(np.char.upper(mol.METADATA["resname"].astype(str)), list(WATER_RESNAMES))


def apply_residue_selection(mol, options, verbose=False):
	"""Resolve --residue/--atom names into spec atoms and apply the environment filters.

	Works on a freshly parsed PDB structure (no --noH/--exclude applied yet) and performs all
	removals in one pass so that spec atoms inside removed groups become zero-radius ghosts and
	indices are renumbered consistently:
		--noH, --exclude      as for any other input
		--nowater             drop water residues
		--nohet               drop hetero groups (HETATM) other than waters, the selected residue and
							  modified polymer residues such as MSE
		--chain X             keep only chain X (plus the selected residue)
		--exclude-self        the selected residue occupies no volume (only its environment counts)
		--self-only           keep only the selected residue

	Sets options.spec_atom_1, options.spec_atom_2 (list) and options.atom3 to renumbered indices.

	Returns:
		dict with label (e.g. "A:45 LEU"), atom names used, and per-atom counts
	"""
	_require_metadata(mol, "--residue")
	meta = mol.METADATA
	n_atoms = len(mol.ATOMTYPES)
	self_mask = residue_mask(mol, options.residue)
	first_mask = self_mask  # atom names are looked up over all selected residues (first match wins)
	if options.self_only and options.exclude_self:
		sys.exit("   --self-only and --exclude-self cannot be combined")

	# --- spec atoms by name (or by file index) -------------------------------------------------
	atom1_value = options.atom if options.atom not in (False, None, "") else options.spec_atom_1
	atom1, atom1_name = resolve_spec_atom(mol, first_mask, atom1_value, ["CA"], "atom1")
	atom2_values = options.spec_atom_2
	if isinstance(atom2_values, str):
		atom2_values = atom2_values.split(",")
	elif atom2_values in (False, None):
		atom2_values = [None]
	elif not isinstance(atom2_values, (list, tuple)):
		atom2_values = [atom2_values]
	atom2, atom2_names = [], []
	for value in atom2_values:
		index, name = resolve_spec_atom(mol, first_mask, value, _DEFAULT_ATOM2, "atom2", avoid=atom1)
		if index == atom1:
			sys.exit(f"   atom2 ({name}) is the same atom as atom1 ({atom1_name}); the Sterimol axis needs two different atoms")
		atom2.append(index)
		atom2_names.append(name)
	if options.sterimol and atom2_names == ["N"] and verbose:
		print("   Note: no CB or HA found in the selected residue; using N as atom2 for the Sterimol axis")
	atom3 = None
	if options.atom3:
		atom3, _ = resolve_spec_atom(mol, first_mask, options.atom3, [], "atom3")

	# --- removal mask ----------------------------------------------------------------------------
	remove = np.zeros(n_atoms, dtype=bool)
	if options.noH:
		remove |= mol.ATOMTYPES == "H"
	if options.exclude:
		indices = options.exclude.split(",") if isinstance(options.exclude, str) else options.exclude
		remove[[int(atom) - 1 for atom in indices]] = True
	water = water_mask(mol)
	if options.nowater:
		remove |= water
	if options.nohet:
		remove |= (meta["record"] == "HETATM") & ~water & ~self_mask & ~polymer_hetatm_mask(mol)
	if options.chain:
		remove |= (np.char.upper(meta["chain"].astype(str)) != str(options.chain).strip().upper()) & ~self_mask
	if options.self_only:
		remove |= ~self_mask
	if options.exclude_self:
		remove |= self_mask

	resname = meta["resname"][self_mask][0]
	spec = [atom1] + atom2 + ([atom3] if atom3 else [])
	new_spec = mol.exclude_mask(remove, spec)
	options.spec_atom_1 = new_spec[0]
	options.spec_atom_2 = new_spec[1:1 + len(atom2)]
	if atom3:
		options.atom3 = new_spec[-1]

	label = " ".join(parse_residue_spec(options.residue)) + " " + resname
	return {
		"label": label.strip(),
		"atom1_name": atom1_name,
		"atom2_names": atom2_names,
		"n_self": int(self_mask.sum()),
		"n_removed": int(remove.sum()),
	}


def load_pdb(file, ext, options):
	"""Parse a PDB file with no atoms removed yet (residue selection applies --noH/--exclude itself)."""
	import copy

	from dbstep import parse_data

	raw = copy.copy(options)
	raw.noH, raw.exclude, raw.spec_atom_1, raw.spec_atom_2 = False, False, 1, [1]
	return parse_data.read_input(file, ext, raw)


def list_residues(mol, options):
	"""Residues covered by --residue all: polymer residues (ATOM records, or HETATM residues with a
	peptide backbone) that contain the atom1 name, in file order; waters and other hetero groups
	are skipped and --chain restricts the chains."""
	_require_metadata(mol, "--residue all")
	# atom1 may be given as --atom or as --atom1; either must be a name in all-residue mode
	atom_name = options.atom if options.atom not in (False, None, "") else options.spec_atom_1
	if atom_name in (False, None, ""):
		atom_name = "CA"
	try:
		int(atom_name)
	except (TypeError, ValueError):
		pass
	else:
		sys.exit("   With --residue all, atom1 must be an atom name such as CA (--atom CA), not a file index")
	meta = mol.METADATA
	names = np.char.upper(meta["name"].astype(str))
	polymer = (meta["record"] == "ATOM") | polymer_hetatm_mask(mol)
	candidates = polymer & ~water_mask(mol)
	if options.chain:
		candidates &= np.char.upper(meta["chain"].astype(str)) == str(options.chain).strip().upper()
	has_atom = set(meta["resid"][candidates & (names == str(atom_name).strip().upper())])
	return [resid for resid in dict.fromkeys(meta["resid"][candidates]) if resid in has_atom]
