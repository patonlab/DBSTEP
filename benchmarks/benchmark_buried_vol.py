#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Benchmark: Grid vs Gaussian buried volume calculations.

Compares wall-clock time and accuracy of the grid/voxel-based buried volume
algorithm against the analytical Gaussian overlap approach across multiple
molecules, sphere radii, and parameter settings.

Usage:
	python benchmarks/benchmark_buried_vol.py
	python benchmarks/benchmark_buried_vol.py --quick     # reduced test matrix
	python benchmarks/benchmark_buried_vol.py --sweep     # include steepness sweep
"""

import sys, os, time, csv, argparse
import numpy as np

# Ensure the package is importable from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dbstep import Dbstep

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

MOLECULES = [
	('H.xyz', 2), ('Me.xyz', 5), ('Et.xyz', 8), ('iPr.xyz', 11),
	('Ph.xyz', 12), ('tBu.xyz', 14), ('cHex.xyz', 18), ('Ad.xyz', 28),
]

RADII = [2.0, 3.5, 5.0]
GRID_SPACINGS = [0.05, 0.1, 0.2]
GAUSS_ORDERS = [2, 3, 4]
N_REPEATS = 5

# Reduced matrix for quick mode
QUICK_MOLECULES = [('H.xyz', 2), ('Me.xyz', 5), ('tBu.xyz', 14)]
QUICK_RADII = [3.5]
QUICK_GRID_SPACINGS = [0.1]
QUICK_GAUSS_ORDERS = [3]

# Steepness sweep prefactors for the sphere Gaussian
STEEPNESS_VALUES = [2.7, 10.0, 50.0, 200.0, 500.0]


def benchmark_grid(filepath, radius, grid_spacing, n_repeats):
	"""Time the grid-based buried volume calculation."""
	times = []
	result = None
	for _ in range(n_repeats):
		t0 = time.perf_counter()
		mol = Dbstep.dbstep(filepath, volume=True, r=radius,
			grid=grid_spacing, commandline=True, quiet=True)
		times.append(time.perf_counter() - t0)
		result = mol.bur_vol
	return np.median(times), result


def benchmark_gaussian(filepath, radius, n_repeats):
	"""Time the Gaussian buried volume calculation."""
	times = []
	result = None
	for _ in range(n_repeats):
		t0 = time.perf_counter()
		mol = Dbstep.dbstep(filepath, volume=True, r=radius,
			method='gaussian', commandline=True, quiet=True)
		times.append(time.perf_counter() - t0)
		result = mol.bur_vol
	return np.median(times), result


def benchmark_gaussian_steepness(filepath, radius, p_sphere):
	"""Compute Gaussian buried volume with a specific sphere prefactor.

	This bypasses the Dbstep class to directly call the sterics function
	with a custom prefactor for the sphere Gaussian steepness sweep.
	"""
	from dbstep.sterics import gaussian_buried_vol
	from dbstep import Dbstep as D
	from dbstep.constants import bondi

	# Parse the molecule the same way Dbstep does
	mol_obj = D.dbstep(filepath, sterimol=True, measure='classic',
		commandline=True, quiet=True)

	# We need coords and radii after translation. Re-parse to get volume setup.
	# Use a minimal grid call just to get transformed coords
	mol_setup = D.dbstep(filepath, volume=True, r=radius, grid=0.5,
		commandline=True, quiet=True)

	# For the steepness sweep, we call the function directly with custom p
	# This requires access to the transformed coordinates, which we get
	# by re-running the full pipeline. Instead, compare against the default.
	from dbstep import parse_data, calculator
	from dbstep.constants import bondi, metals

	options = D.set_options({'volume': True, 'r': radius, 'commandline': True,
		'quiet': True})
	options.spec_atom_1 = 1
	options.spec_atom_2 = [2]
	mol = parse_data.read_input(filepath, os.path.splitext(filepath)[1], options)
	mol.RADII = np.array([bondi.get(a, 2.0) for a in mol.ATOMTYPES])
	origin = np.array([0, 0, 0])
	mol.CARTESIANS = calculator.translate_mol(mol, options, origin)

	# Remove metals
	keep = [i for i, a in enumerate(mol.ATOMTYPES) if a not in metals]
	coords = mol.CARTESIANS[keep]
	radii = mol.RADII[keep]

	t0 = time.perf_counter()
	result = gaussian_buried_vol(coords, radii, origin, radius, p=p_sphere)
	elapsed = time.perf_counter() - t0
	return elapsed, result


def print_table(results):
	"""Print a formatted results table."""
	header = "{:<12} {:>5} {:>6} {:>10} {:>10} {:>10} {:>10}".format(
		"Molecule", "Atoms", "R(Å)", "Method", "Param", "Time(s)", "%Vbur")
	print("\n" + "=" * len(header))
	print("BURIED VOLUME BENCHMARK RESULTS")
	print("=" * len(header))
	print(header)
	print("-" * len(header))
	for r in results:
		print("{:<12} {:>5} {:>6.1f} {:>10} {:>10} {:>10.4f} {:>10.2f}".format(
			r['molecule'], r['atoms'], r['radius'], r['method'],
			r['param'], r['time'], r['vbur']))
	print("=" * len(header))


def write_csv(results, filename):
	"""Write results to CSV file."""
	fieldnames = ['molecule', 'atoms', 'radius', 'method', 'param', 'time', 'vbur']
	with open(filename, 'w', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(results)
	print(f"\nResults written to {filename}")


def run_benchmark(molecules, radii, grid_spacings, gauss_orders, n_repeats):
	"""Run the main benchmark suite."""
	results = []
	total = len(molecules) * len(radii) * (len(grid_spacings) + len(gauss_orders))
	count = 0

	for mol_name, n_atoms in molecules:
		filepath = os.path.join('dbstep', 'examples', mol_name)
		if not os.path.exists(filepath):
			print(f"  WARNING: {filepath} not found, skipping")
			continue

		for radius in radii:
			# Grid methods
			for spacing in grid_spacings:
				count += 1
				sys.stdout.write(f"\r  [{count}/{total}] {mol_name} R={radius} grid={spacing}    ")
				sys.stdout.flush()
				t, vbur = benchmark_grid(filepath, radius, spacing, n_repeats)
				results.append({
					'molecule': mol_name, 'atoms': n_atoms, 'radius': radius,
					'method': 'grid', 'param': f'g={spacing}', 'time': t, 'vbur': vbur
				})

			# Gaussian methods
			for order in gauss_orders:
				count += 1
				sys.stdout.write(f"\r  [{count}/{total}] {mol_name} R={radius} gauss ord={order}    ")
				sys.stdout.flush()
				t, vbur = benchmark_gaussian(filepath, radius, n_repeats)
				results.append({
					'molecule': mol_name, 'atoms': n_atoms, 'radius': radius,
					'method': 'gaussian', 'param': f'ord={order}', 'time': t, 'vbur': vbur
				})

	print()  # newline after progress
	return results


def run_steepness_sweep(molecules, radius=3.5):
	"""Sweep sphere Gaussian steepness (prefactor p) and compare to grid reference."""
	print("\n" + "=" * 70)
	print("SPHERE GAUSSIAN STEEPNESS SWEEP (R={:.1f} Å)".format(radius))
	print("=" * 70)
	header = "{:<12} {:>5} {:>10} {:>10} {:>10} {:>10}".format(
		"Molecule", "Atoms", "p_sphere", "Time(s)", "%Vbur", "Δ vs grid")
	print(header)
	print("-" * len(header))

	for mol_name, n_atoms in molecules:
		filepath = os.path.join('dbstep', 'examples', mol_name)
		if not os.path.exists(filepath):
			continue

		# Grid reference at fine spacing
		_, grid_ref = benchmark_grid(filepath, radius, 0.05, 1)

		for p_val in STEEPNESS_VALUES:
			try:
				t, vbur = benchmark_gaussian_steepness(filepath, radius, p_val)
				delta = vbur - grid_ref
				print("{:<12} {:>5} {:>10.1f} {:>10.4f} {:>10.2f} {:>+10.2f}".format(
					mol_name, n_atoms, p_val, t, vbur, delta))
			except Exception as e:
				print("{:<12} {:>5} {:>10.1f}  ERROR: {}".format(
					mol_name, n_atoms, p_val, str(e)[:30]))

	print("=" * 70)


def main():
	parser = argparse.ArgumentParser(description='Benchmark buried volume calculations')
	parser.add_argument('--quick', action='store_true',
		help='Run reduced test matrix for faster results')
	parser.add_argument('--sweep', action='store_true',
		help='Include sphere Gaussian steepness sweep')
	parser.add_argument('--repeats', type=int, default=N_REPEATS,
		help=f'Number of timing repeats (default: {N_REPEATS})')
	parser.add_argument('--csv', type=str, default='benchmarks/results.csv',
		help='Output CSV file path')
	args = parser.parse_args()

	if args.quick:
		molecules = QUICK_MOLECULES
		radii = QUICK_RADII
		grid_spacings = QUICK_GRID_SPACINGS
		gauss_orders = QUICK_GAUSS_ORDERS
	else:
		molecules = MOLECULES
		radii = RADII
		grid_spacings = GRID_SPACINGS
		gauss_orders = GAUSS_ORDERS

	print("DBSTEP Buried Volume Benchmark")
	print(f"  Molecules: {len(molecules)}")
	print(f"  Radii: {radii}")
	print(f"  Grid spacings: {grid_spacings}")
	print(f"  Gaussian orders: {gauss_orders}")
	print(f"  Repeats per config: {args.repeats}")
	print()

	results = run_benchmark(molecules, radii, grid_spacings, gauss_orders, args.repeats)
	print_table(results)
	write_csv(results, args.csv)

	if args.sweep:
		sweep_mols = QUICK_MOLECULES if args.quick else molecules
		run_steepness_sweep(sweep_mols)


if __name__ == '__main__':
	main()
