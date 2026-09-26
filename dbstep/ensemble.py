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


def energy_of(run, tag=None):
	"""Energy of one run from its structure properties.

	With a `tag`, that SDF data field is used (case-insensitive). Otherwise the fields in ENERGY_TAGS
	are tried, then a floating-point number in the xyz comment line. Exits with a clear message when
	nothing usable is found, since silently dropping a conformer would bias the average.
	"""
	properties = getattr(run, "properties", {}) or {}
	lowered = {str(key).strip().lower(): value for key, value in properties.items()}
	if tag not in (None, False, "", "auto"):
		key = str(tag).strip().lower()
		if key not in lowered:
			sys.exit(f"   No <{tag}> data field for structure '{_label(run)}'. Fields present: {', '.join(properties) or 'none'}")
		value = field_number(lowered[key])
		if value is None:
			sys.exit(f"   Data field <{tag}> of structure '{_label(run)}' is not a number: {lowered[key]!r}")
		return value
	for candidate in ENERGY_TAGS:
		if candidate in lowered:
			value = field_number(lowered[candidate])
			if value is not None:
				return value
	if "comment" in lowered:
		value = number_in_text(lowered["comment"])
		if value is not None:
			return value
	sys.exit(f"   No energy found for structure '{_label(run)}': expected an SDF data field such as <Energy> (fields present: {', '.join(k for k in properties if k != 'comment') or 'none'}) or a number in the xyz comment line. Use --boltzmann TAG to name the field.")


def boltzmann_weights(energies, temperature=298.15, units="kcal"):
	"""Normalised Boltzmann populations of structures with the given energies."""
	if temperature <= 0:
		sys.exit("   Temperature for Boltzmann weighting must be positive")
	energies = to_kcal(energies, units)
	relative = energies - energies.min()
	weights = np.exp(-relative / (R_KCAL * temperature))
	return weights / weights.sum()


def boltzmann_average(runs, tag=None, temperature=298.15, units="kcal"):
	"""Boltzmann-weighted steric parameters over the dbstep runs of one ensemble (e.g. all_frames of an SDF).

	Sets `run.population` on every run and adds a "population" entry to each of its result rows.

	Returns:
		list of summary rows (one per radius), shaped like dbstep.results with structure "boltzmann"
		and the weighted averages of mol_vol, percent_vbur, percent_sbur, bmin, bmax and L
	"""
	runs = list(runs)
	if not runs:
		return []
	energies = [energy_of(run, tag) for run in runs]
	weights = boltzmann_weights(energies, temperature, units)
	n_rows = len(runs[0].results)
	for run, weight in zip(runs, weights):
		if len(run.results) != n_rows:
			sys.exit("   Structures of the ensemble produced different numbers of result rows; Boltzmann averaging needs the same radii for every structure")
		run.population = float(weight)
		run.energy = float(energies[runs.index(run)])
		for row in run.results:
			row["population"] = float(weight)

	summary = []
	for i in range(n_rows):
		first = runs[0].results[i]
		row = {
			"file": first["file"],
			"path": first.get("path", ""),
			"frame": "",
			"structure": "boltzmann",
			"residue": first["residue"],
			"atom1": first["atom1"],
			"atom2": first["atom2"],
			"radius": first["radius"],
			"population": 1.0,
		}
		for key in ("mol_vol", "percent_vbur", "percent_sbur", "bmin", "bmax", "L"):
			values = [run.results[i][key] for run in runs]
			row[key] = float(np.dot(weights, values)) if all(value != "" for value in values) else ""
		summary.append(row)
	return summary
