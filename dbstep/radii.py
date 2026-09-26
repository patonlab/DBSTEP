# -*- coding: UTF-8 -*-

"""
radii

Assignment of atomic radii: Bondi and Charry-Tkatchenko radii by element, and the Sterimol
CPK radii, which depend on the bonding environment of each atom (sp3 vs aromatic carbon,
single- vs double-bonded oxygen, ...). Atom types for the CPK set are derived from DFT-D3
style fractional coordination numbers, following the original patonlab/sterimol code, so that
no bond perception is needed.
"""

import numpy as np
from scipy.spatial import cKDTree

from dbstep.constants import bondi, charry_tkatchenko, covalent, cpk, cpk_upper_bound

# coordination-number damping function parameters (Grimme, DFT-D3)
K1 = 16.0
K2 = 4.0 / 3.0
# pair distance beyond which a neighbour contributes less than 1e-5 to any coordination number
CN_CUTOFF = 8.0

RADII_SETS = ("bondi", "charry-tkatchenko", "cpk")
CPK_TYPE_KEY = "cpk_type"


def coordination_numbers(atomtypes, coords):
	"""DFT-D3 fractional coordination numbers, CN_i = sum_j 1 / (1 + exp(-k1 (k2 (R_i + R_j) / r_ij - 1))).

	Atoms whose element has no covalent radius (ghosts, dummy atoms) get CN 0 and are not counted as neighbours.
	"""
	atomtypes = np.asarray(atomtypes, dtype=str)
	coords = np.asarray(coords, dtype=float)
	n = len(atomtypes)
	cn = np.zeros(n)
	if n < 2:
		return cn
	rcov = np.array([covalent.get(atom, np.nan) for atom in atomtypes])
	known = ~np.isnan(rcov)
	if known.sum() < 2:
		return cn
	index = np.flatnonzero(known)
	tree = cKDTree(coords[index])
	pairs = tree.query_pairs(CN_CUTOFF, output_type="ndarray")
	if len(pairs) == 0:
		return cn
	i, j = index[pairs[:, 0]], index[pairs[:, 1]]
	r = np.linalg.norm(coords[i] - coords[j], axis=1)
	with np.errstate(over="ignore"):
		damp = 1.0 / (1.0 + np.exp(-K1 * (K2 * (rcov[i] + rcov[j]) / r - 1.0)))
	np.add.at(cn, i, damp)
	np.add.at(cn, j, damp)
	return cn


def cpk_type(element, cn):
	"""Sterimol atom type of an atom from its element and fractional coordination number, or None
	when the CPK table has no entry for the element."""
	if element in ("H", "P", "F", "I", "Bq"):
		return element
	if element == "Cl":
		return "C1"
	if element == "Br":
		return "B1"
	if element == "O":  # double-bonded (carbonyl, nitro) vs single-bonded oxygen
		return "O2" if cn < 1.5 else "O"
	if element == "S":  # divalent, tetrahedral (sulfone, sulfonyl), octahedral (SF6-like)
		if cn < 2.5:
			return "S"
		return "S4" if cn < 5.5 else "S1"
	if element == "N":  # planar (amide, aromatic, imine) vs tetrahedral nitrogen
		return "N" if cn > 2.5 else "C6/N6"
	if element == "C":  # sp, sp2 (typed as aromatic, as in the original code) and sp3 carbon
		if cn < 2.5:
			return "C3"
		return "C6/N6" if cn < 3.5 else "C"
	return None


def cpk_types(atomtypes, coords):
	"""Sterimol atom types for every atom of a structure (None where the element has no CPK entry)."""
	cn = coordination_numbers(atomtypes, coords)
	return [cpk_type(atom, c) for atom, c in zip(np.asarray(atomtypes, dtype=str), cn)]


def attach_cpk_types(mol):
	"""Store the Sterimol atom type of every atom in mol.METADATA so it survives later atom removal
	(--noH, --exclude, crops, residue filters). Must be called on the complete structure, since the
	types depend on every neighbour."""
	if not hasattr(mol, "METADATA") or mol.METADATA is None:
		mol.METADATA = {}
	types = cpk_types(mol.ATOMTYPES, mol.CARTESIANS)
	mol.METADATA[CPK_TYPE_KEY] = np.array(["" if t is None else str(t) for t in types], dtype=object)


def radii_table(radii_set):
	"""Element-keyed radii used for the given set; for CPK the per-element upper bound plus Bondi fallbacks."""
	if radii_set == "charry-tkatchenko":
		return charry_tkatchenko
	if radii_set == "cpk":
		return {**bondi, **cpk_upper_bound}
	return bondi


def radii_label(radii_set):
	return {"charry-tkatchenko": "Charry-Tkatchenko", "cpk": "CPK (Sterimol atom types)"}.get(radii_set, "Bondi")


def _cpk_radii(atomtypes, types):
	radii = []
	for atom, cpk_t in zip(atomtypes, types):
		if atom == "Bq":
			radii.append(0.0)
		elif cpk_t:
			radii.append(cpk[cpk_t])
		else:
			radii.append(bondi.get(atom, 2.0))
	return np.array(radii, dtype=float)


def for_atoms(atomtypes, coords, radii_set="bondi"):
	"""Unscaled VDW radius of every atom of a complete structure for the given radii set
	(CPK types are derived from the coordinates)."""
	atomtypes = np.asarray(atomtypes, dtype=str)
	if radii_set != "cpk":
		table = radii_table(radii_set)
		return np.array([table.get(atom, 2.0) for atom in atomtypes], dtype=float)
	return _cpk_radii(atomtypes, ["" if t is None else t for t in cpk_types(atomtypes, coords)])


def atom_radii(mol, options):
	"""Unscaled VDW radius of every atom of `mol` for options.radii.

	Elements without a CPK type (metals, boron, silicon, ...) fall back to their Bondi radius;
	ghost atoms ("Bq") always have radius zero.
	"""
	atomtypes = np.asarray(mol.ATOMTYPES, dtype=str)
	types = getattr(mol, "METADATA", {}).get(CPK_TYPE_KEY) if options.radii == "cpk" else None
	if types is None:  # not CPK, or a structure parsed without the CPK hook (e.g. built by hand)
		return for_atoms(atomtypes, mol.CARTESIANS, options.radii)
	return _cpk_radii(atomtypes, types)


def max_radius(atomtypes, options):
	"""Largest (scaled) radius any of the given atom types can be assigned under options.radii."""
	if len(atomtypes) == 0:
		return 0.0
	table = radii_table(options.radii)
	return max(table.get(atom, 2.0) for atom in atomtypes) * options.SCALE_VDW
