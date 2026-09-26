# -*- coding: UTF-8 -*-
import re
import sys
import numpy as np

"""
ensemble

Boltzmann weighting of steric parameters over the structures of a conformer ensemble
(multi-record SDF files with an energy data field, e.g. from AQME/CREST, or multi-frame xyz
files with the energy in the comment line).
"""

R_KCAL = 0.0019872041  # gas constant in kcal/(mol K)

# conversion factors to kcal/mol
TO_KCAL = {"kcal": 1.0, "kcal/mol": 1.0, "kj": 1.0 / 4.184, "kj/mol": 1.0 / 4.184, "hartree": 627.509474, "au": 627.509474, "ev": 23.060548}
UNIT_LABELS = {"kcal": "kcal/mol", "kj": "kJ/mol", "hartree": "hartree", "ev": "eV"}

# SDF data-field names tried (case-insensitively) when no tag is given
ENERGY_TAGS = ("energy", "e", "g", "dg", "de", "free energy", "gibbs free energy", "total energy", "scf energy", "energy (kcal/mol)", "e (kcal/mol)", "g (kcal/mol)", "rel. energy", "relative energy")

_FLOAT = re.compile(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+[eE][-+]?\d+")


def unit_label(units):
	return UNIT_LABELS.get(str(units).lower(), str(units))


def to_kcal(energies, units="kcal"):
	"""Convert energies to kcal/mol. `units` is one of kcal, kJ, hartree, eV (case-insensitive)."""
	key = str(units).lower().replace("/mol", "")
	if key not in TO_KCAL:
		sys.exit(f"   Unknown energy units '{units}'. Use kcal, kJ, hartree or eV.")
	return np.asarray(energies, dtype=float) * TO_KCAL[key]


def number_in_text(text):
	"""First floating-point number (with a decimal point or exponent) in a string, or None.

	Integers are deliberately not accepted, so a title like "ether 44" is not read as an energy.
	"""
	match = _FLOAT.search(str(text))
	return float(match.group(0)) if match else None


def field_number(text):
	"""Number in an SDF data field: the whole value as a float (so "0" and "12" work), else the first float in the text."""
	try:
		return float(str(text).strip())
	except ValueError:
		return number_in_text(text)


def _label(run):
	return run.structure_name or run.results[0]["file"] if run.results else str(run.file)


# property keys that describe the energies rather than hold one
META_KEYS = ("energy_units", "energy_source")


def _fields(properties):
	return [key for key in properties if key not in META_KEYS and key != "comment"]


def energy_and_key(run, tag=None, units="kcal"):
	"""Energy of one run in kcal/mol and the property it came from.

	With a `tag`, that data field is used (case-insensitive). Otherwise the fields in ENERGY_TAGS
	are tried (for QM output files read by cclib this is "energy": G when a frequency calculation
	provides it, else the last SCF energy), then a floating-point number in the xyz comment line.
	Values are converted from `units`, unless the structure's properties carry their own
	"energy_units" (hartree for QM outputs). Exits with a clear message when nothing usable is found,
	since silently dropping a conformer would bias the average.
	"""
	properties = getattr(run, "properties", {}) or {}
	lowered = {str(key).strip().lower(): value for key, value in properties.items()}
	run_units = properties.get("energy_units", units)
	if tag not in (None, False, "", "auto"):
		key = str(tag).strip().lower()
		if key not in lowered or key in META_KEYS:
			sys.exit(f"   No <{tag}> data field for structure '{_label(run)}'. Fields present: {', '.join(_fields(properties)) or 'none'}")
		value = field_number(lowered[key])
		if value is None:
			sys.exit(f"   Data field <{tag}> of structure '{_label(run)}' is not a number: {lowered[key]!r}")
		return float(to_kcal([value], run_units)[0]), str(tag)
	for candidate in ENERGY_TAGS:
		if candidate in lowered:
			value = field_number(lowered[candidate])
			if value is not None:
				name = candidate
				if candidate == "energy" and properties.get("energy_source"):
					name = {"G": "Gibbs free energy", "E": "SCF energy", "H": "enthalpy"}.get(properties["energy_source"], properties["energy_source"])
				return float(to_kcal([value], run_units)[0]), name
	if "comment" in lowered:
		value = number_in_text(lowered["comment"])
		if value is not None:
			return float(to_kcal([value], run_units)[0]), "comment line"
	sys.exit(f"   No energy found for structure '{_label(run)}': expected an SDF data field such as <Energy> (fields present: {', '.join(_fields(properties)) or 'none'}), an energy in the QM output file, or a number in the xyz comment line. Use --boltzmann TAG to name the field.")


def energy_of(run, tag=None, units="kcal"):
	"""Energy of one run in kcal/mol (see energy_and_key)."""
	return energy_and_key(run, tag, units)[0]


def boltzmann_weights(energies, temperature=298.15, units="kcal"):
	"""Normalised Boltzmann populations of structures with the given energies."""
	if temperature <= 0:
		sys.exit("   Temperature for Boltzmann weighting must be positive")
	energies = to_kcal(energies, units)
	relative = energies - energies.min()
	weights = np.exp(-relative / (R_KCAL * temperature))
	return weights / weights.sum()


def boltzmann_average(runs, tag=None, temperature=298.15, units="kcal", window=None, label=None):
	"""Boltzmann-weighted steric parameters over the dbstep runs of one ensemble: the frames of a
	multi-structure file (all_frames of an SDF) or one run per QM output file.

	`window` (kcal/mol) drops the structures more than that above the lowest energy from the
	weighting: they keep population 0 and stay in the results. `label` replaces the file name of the
	summary rows (used for ensembles pooled from several files).

	Sets on every run: `population`, `energy` (kcal/mol), `energy_rel` (kcal/mol above the minimum),
	`in_window` and `energy_key` (where the energy came from), and adds a "population" entry to each
	of its result rows.

	Returns:
		list of summary rows (one per radius), shaped like dbstep.results with structure "boltzmann"
		and the weighted averages of mol_vol, percent_vbur, percent_sbur, bmin, bmax, L (and the cone
		angle and metal-centroid distance in --cone mode)
	"""
	runs = list(runs)
	if not runs:
		return []
	no_window = window is None or window is False
	if not no_window and float(window) <= 0:
		sys.exit("   The energy window for Boltzmann weighting must be positive (kcal/mol)")
	energies, keys = zip(*[energy_and_key(run, tag, units) for run in runs])
	energies = np.asarray(energies, dtype=float)
	relative = energies - energies.min()
	included = np.ones(len(runs), dtype=bool) if no_window else relative <= float(window) + 1e-9
	weights = np.zeros(len(runs))
	weights[included] = boltzmann_weights(energies[included], temperature, "kcal")
	n_rows = len(runs[0].results)
	for run, weight, energy, rel, inside, key in zip(runs, weights, energies, relative, included, keys):
		if len(run.results) != n_rows:
			sys.exit("   Structures of the ensemble produced different numbers of result rows; Boltzmann averaging needs the same radii for every structure")
		run.population = float(weight)
		run.energy, run.energy_rel, run.in_window, run.energy_key = float(energy), float(rel), bool(inside), key
		for row in run.results:
			row["population"] = float(weight)

	summary = []
	for i in range(n_rows):
		first = runs[0].results[i]
		row = {
			"file": label or first["file"],
			"path": "" if label else first.get("path", ""),
			"frame": "",
			"structure": "boltzmann",
			"residue": first["residue"],
			"atom1": first["atom1"],
			"atom2": first["atom2"],
			"radius": first["radius"],
			"population": 1.0,
		}
		for key in ("mol_vol", "percent_vbur", "percent_sbur", "bmin", "bmax", "L", "cone_angle", "metal_centroid"):
			values = [run.results[i].get(key, "") for run in runs]
			row[key] = float(np.dot(weights, values)) if all(value != "" for value in values) else ""
		summary.append(row)
	return summary
