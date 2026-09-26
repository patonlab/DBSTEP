# -*- coding: UTF-8 -*-

"""
cone

Tolman cone angles and metal-to-ligand (ring centroid or donor atom) distances for the ligands of
metal complexes, the analysis that the earlier patonlab/sterimol code performed on half-sandwich
complexes. The ligand is the bonded fragment that contains the axis atoms (a ring, or a single
donor atom) once the metal is removed; its Sterimol parameters are then measured from the metal
along the metal-to-centroid axis by the usual DBSTEP machinery, with every other ligand excluded.
"""

import math
import sys

import numpy as np

from dbstep.constants import covalent, metals

BOND_SCALE = 1.2  # two non-metal atoms are bonded when closer than BOND_SCALE * (R_i + R_j), R = covalent radius
METAL_BOND_SCALE = 1.3  # a ligand atom is bound to the metal when closer than METAL_BOND_SCALE * (R_M + R_i)
DEFAULT_COVALENT = 1.5  # covalent radius for elements missing from the table (metals beyond Pu, dummies)


def _covalent(atom):
	return covalent.get(atom, DEFAULT_COVALENT)


def neighbours(atomtypes, coords, skip=()):
	"""Bonded neighbours of every atom (list of sets, 0-based) from covalent radii; atoms in `skip` have no bonds."""
	atomtypes = np.asarray(atomtypes, dtype=str)
	coords = np.asarray(coords, dtype=float)
	n = len(atomtypes)
	rcov = np.array([_covalent(atom) for atom in atomtypes])
	adjacency = [set() for _ in range(n)]
	active = np.ones(n, dtype=bool)
	active[list(skip)] = False
	active &= atomtypes != "Bq"
	index = np.flatnonzero(active)
	if len(index) < 2:
		return adjacency
	from scipy.spatial import cKDTree

	tree = cKDTree(coords[index])
	pairs = tree.query_pairs(BOND_SCALE * 2 * rcov[index].max(), output_type="ndarray")
	for a, b in pairs:
		i, j = index[a], index[b]
		if np.linalg.norm(coords[i] - coords[j]) <= BOND_SCALE * (rcov[i] + rcov[j]):
			adjacency[i].add(j)
			adjacency[j].add(i)
	return adjacency


def connected(adjacency, seeds, allowed=None):
	"""Atoms reachable from `seeds` through the adjacency lists (restricted to `allowed` when given), sorted."""
	seen, stack = set(), list(seeds)
	while stack:
		i = stack.pop()
		if i in seen or (allowed is not None and i not in allowed):
			continue
		seen.add(i)
		stack.extend(adjacency[i] - seen)
	return sorted(seen)


def find_metal(atomtypes):
	"""Index (0-based) of the single metal atom of a structure, or an error naming the choices."""
	found = [i for i, atom in enumerate(atomtypes) if atom in metals]
	if len(found) == 1:
		return found[0]
	if not found:
		sys.exit("   --cone: no metal atom found; give the apex atom with --atom1")
	sys.exit("   --cone: several metal atoms ({}); choose one with --atom1".format(", ".join("{}{}".format(atomtypes[i], i + 1) for i in found)))


def find_ligand_atoms(atomtypes, coords, metal, adjacency):
	"""Auto-detect the atoms defining the ligand axis: the largest ring of atoms bound to the metal
	(a Cp, arene, ... ligand), otherwise the nearest non-hydrogen atom bound to the metal (a donor atom)."""
	atomtypes = np.asarray(atomtypes, dtype=str)
	coords = np.asarray(coords, dtype=float)
	distances = np.linalg.norm(coords - coords[metal], axis=1)
	r_metal = _covalent(atomtypes[metal])
	candidates = [i for i in range(len(atomtypes)) if i != metal and atomtypes[i] not in ("H", "Bq") and atomtypes[i] not in metals and distances[i] <= METAL_BOND_SCALE * (r_metal + _covalent(atomtypes[i]))]
	if not candidates:
		sys.exit("   --cone: no ligand atoms within bonding distance of {}{}; give the ring or donor atoms with --atom2".format(atomtypes[metal], metal + 1))
	pool = set(candidates)
	rings, remaining = [], set(candidates)
	while remaining:
		component = connected(adjacency, [remaining.pop()], allowed=pool)
		remaining -= set(component)
		if len(component) >= 3 and all(len(adjacency[i] & set(component)) >= 2 for i in component):
			rings.append(component)
	if rings:
		return max(rings, key=len)
	return [min(candidates, key=lambda i: distances[i])]


def half_angles(apex, coords, radii, axis):
	"""Half cone angle of every atom seen from `apex`: angle to the axis plus the angular radius of its sphere."""
	vectors = np.asarray(coords, dtype=float) - apex
	distances = np.linalg.norm(vectors, axis=1)
	if np.any(distances == 0):
		sys.exit("   --cone: a ligand atom coincides with the apex atom")
	cosines = np.clip(vectors @ axis / distances, -1.0, 1.0)
	alpha = np.arccos(cosines)
	beta = np.arcsin(np.clip(np.asarray(radii, dtype=float) / distances, 0.0, 1.0))
	return alpha + beta


def azimuths(apex, coords, axis):
	"""Angle of every atom around `axis` (radians in (-pi, pi]), measured from an arbitrary perpendicular."""
	trial = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
	e1 = np.cross(axis, trial)
	e1 /= np.linalg.norm(e1)
	e2 = np.cross(axis, e1)
	vectors = np.asarray(coords, dtype=float) - apex
	return np.arctan2(vectors @ e2, vectors @ e1)


def cone_angle(apex, axis_atoms, ligand_atoms, coords, radii, adjacency):
	"""Tolman cone angle (degrees) of the ligand `ligand_atoms` seen from `apex`.

	The axis runs from the apex to the centroid of `axis_atoms`. Each atom subtends the half angle
	alpha + asin(r / d); the cone angle is twice the mean, over the ligand's sectors, of the largest
	half angle in the sector (Tolman's construction for unsymmetrical ligands). For a ring (two or
	more axis atoms) the sectors are the azimuthal wedges around the axis closest to each ring atom;
	for a single donor atom they are the substituent branches attached to it. A ligand without
	branches (a single atom) gives the full angle of its own sphere.

	Returns:
		(cone angle in degrees, half angle per sector in degrees, apex-to-centroid distance)
	"""
	coords = np.asarray(coords, dtype=float)
	centroid = coords[list(axis_atoms)].mean(axis=0)
	axis = centroid - apex
	distance = float(np.linalg.norm(axis))
	if distance == 0:
		sys.exit("   --cone: the ligand centroid coincides with the apex atom")
	axis = axis / distance
	ligand = list(ligand_atoms)
	theta = dict(zip(ligand, half_angles(apex, coords[ligand], radii[ligand], axis)))
	if len(axis_atoms) >= 2:
		phi = dict(zip(ligand, azimuths(apex, coords[ligand], axis)))
		ring_phi = [phi[i] for i in axis_atoms]
		sectors = [[] for _ in axis_atoms]
		for i in ligand:
			separation = [abs((phi[i] - p + math.pi) % (2 * math.pi) - math.pi) for p in ring_phi]
			sectors[int(np.argmin(separation))].append(i)
	else:
		donor = axis_atoms[0]
		rest = set(ligand) - {donor}
		sectors = []
		while rest:  # one sector per substituent branch attached to the donor atom
			branch = connected(adjacency, [next(iter(rest))], allowed=rest)
			rest -= set(branch)
			sectors.append(branch + [donor])
	if not sectors:
		sectors = [ligand]
	sector_angles = [math.degrees(max(theta[i] for i in sector)) for sector in sectors if sector]
	return 2.0 * float(np.mean(sector_angles)), sector_angles, distance


def analyse(mol, options, radii, atom1_given, atom2_given, verbose=False):
	"""Cone-angle analysis of the ligand chosen by --atom1 (metal) and --atom2 (ring or donor atoms).

	Computes the cone angle and the metal-to-centroid distance on the complete structure, then
	removes every atom outside the ligand (the metal stays as a zero-radius ghost, so the Sterimol
	axis and origin are preserved) and renumbers the spec atoms.

	Returns:
		dict with metal, axis_atoms and ligand_atoms (1-based indices in the input numbering),
		cone_angle (degrees), sector_angles (degrees) and metal_centroid (Angstrom)
	"""
	atomtypes = np.asarray(mol.ATOMTYPES, dtype=str)
	coords = np.asarray(mol.CARTESIANS, dtype=float)
	metal = options.spec_atom_1 - 1 if atom1_given else find_metal(atomtypes)
	if metal >= len(atomtypes):
		sys.exit("   --cone: atom1 index {} is beyond the {} atoms of the structure".format(metal + 1, len(atomtypes)))
	metal_atoms = [i for i, atom in enumerate(atomtypes) if atom in metals]
	adjacency = neighbours(atomtypes, coords, skip=set(metal_atoms) | {metal})
	if atom2_given:
		axis_atoms = [int(a) - 1 for a in options.spec_atom_2]
		if metal in axis_atoms:
			sys.exit("   --cone: atom2 must not include the apex atom (atom1)")
	else:
		axis_atoms = find_ligand_atoms(atomtypes, coords, metal, adjacency)
	ligand = connected(adjacency, axis_atoms)
	angle, sector_angles, distance = cone_angle(coords[metal], axis_atoms, ligand, coords, np.asarray(radii, dtype=float), adjacency)
	if verbose:
		print("   Cone angle: apex {}{}, axis to the centroid of {}, ligand of {} atoms, sector half angles {}".format(
			atomtypes[metal], metal + 1, ", ".join("{}{}".format(atomtypes[i], i + 1) for i in axis_atoms), len(ligand), ", ".join("{:.1f}".format(a) for a in sector_angles)))
	# keep the ligand only; the metal is a spec atom, so exclude_mask turns it into a ghost
	remove = np.ones(len(atomtypes), dtype=bool)
	remove[ligand] = False
	spec = [metal + 1] + [i + 1 for i in axis_atoms] + ([int(options.atom3)] if options.atom3 else [])
	if options.atom3 and int(options.atom3) - 1 not in ligand and int(options.atom3) - 1 != metal:
		sys.exit("   --cone: atom3 must belong to the ligand")
	new_spec = mol.exclude_mask(remove, spec)
	options.spec_atom_1, options.spec_atom_2 = new_spec[0], new_spec[1:1 + len(axis_atoms)]
	if options.atom3:
		options.atom3 = new_spec[-1]
	return {
		"metal": metal + 1,
		"axis_atoms": [i + 1 for i in axis_atoms],
		"ligand_atoms": [i + 1 for i in ligand],
		"cone_angle": angle,
		"sector_angles": sector_angles,
		"metal_centroid": distance,
	}
