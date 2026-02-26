import pytest
import math
import numpy as np

from dbstep import Dbstep
from dbstep.sterics import gaussian_params, gaussian_overlap_vol, gaussian_buried_vol


class TestGaussianParams:
	"""Tests for gaussian_params: verify kappa values produce correct volume integrals."""

	def test_single_radius(self):
		"""Kappa for a single atom should give volume integral = (4/3)*pi*R^3."""
		p = 2.7
		R = 1.70  # Carbon VDW radius
		kappas = gaussian_params([R], p)
		# Volume integral: p * (pi/kappa)^(3/2) should equal (4/3)*pi*R^3
		vol_integral = p * (math.pi / kappas[0])**1.5
		expected_vol = 4.0 / 3.0 * math.pi * R**3
		assert abs(vol_integral - expected_vol) < 1e-10

	@pytest.mark.parametrize("R", [1.09, 1.20, 1.52, 1.70, 1.80, 2.10, 3.50])
	def test_volume_integral_matches_sphere(self, R):
		"""For various radii, the Gaussian volume integral should match hard-sphere volume."""
		p = 2.7
		kappas = gaussian_params([R], p)
		vol_integral = p * (math.pi / kappas[0])**1.5
		expected_vol = 4.0 / 3.0 * math.pi * R**3
		assert abs(vol_integral - expected_vol) / expected_vol < 1e-10

	def test_array_input(self):
		"""Should handle arrays of radii."""
		radii = [1.20, 1.70, 1.55, 1.80]
		p = 2.7
		kappas = gaussian_params(radii, p)
		assert len(kappas) == 4
		for i, R in enumerate(radii):
			vol = p * (math.pi / kappas[i])**1.5
			expected = 4.0 / 3.0 * math.pi * R**3
			assert abs(vol - expected) / expected < 1e-10

	def test_larger_radius_smaller_kappa(self):
		"""Larger atoms should have smaller kappa (broader Gaussian)."""
		kappas = gaussian_params([1.20, 1.70, 2.10])
		assert kappas[0] > kappas[1] > kappas[2]


class TestGaussianOverlapVol:
	"""Tests for gaussian_overlap_vol: verify overlap integrals are correct."""

	def test_self_overlap(self):
		"""Overlap of a Gaussian with itself (at the same center) should be
		p^2 * (pi/(2*kappa))^(3/2) since delta=0 and K=2*kappa."""
		p = 2.7
		R = 1.70
		kappas = gaussian_params([R], p)
		k = kappas[0]
		vol = gaussian_overlap_vol([p, p], [k, k],
			[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
		expected = p**2 * (math.pi / (2 * k))**1.5
		assert abs(vol - expected) < 1e-10

	def test_distant_atoms_zero_overlap(self):
		"""Two atoms very far apart should have near-zero overlap."""
		p = 2.7
		kappas = gaussian_params([1.70, 1.70], p)
		vol = gaussian_overlap_vol([p, p], kappas,
			[[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
		assert vol < 1e-50

	def test_two_body_symmetry(self):
		"""Overlap of A with B should equal overlap of B with A."""
		p = 2.7
		kappas = gaussian_params([1.70, 1.55], p)
		c1 = [0.0, 0.0, 0.0]
		c2 = [1.5, 0.0, 0.0]
		vol_ab = gaussian_overlap_vol([p, p], kappas, [c1, c2])
		vol_ba = gaussian_overlap_vol([p, p], [kappas[1], kappas[0]], [c2, c1])
		assert abs(vol_ab - vol_ba) < 1e-10

	def test_three_body_overlap(self):
		"""Three-body overlap should be smaller than any two-body overlap."""
		p = 2.7
		kappas = gaussian_params([1.70, 1.70, 1.70], p)
		centers = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.75, 1.3, 0.0]]
		v3 = gaussian_overlap_vol([p, p, p], kappas, centers)
		v12 = gaussian_overlap_vol([p, p], kappas[:2], centers[:2])
		v13 = gaussian_overlap_vol([p, p], [kappas[0], kappas[2]],
			[centers[0], centers[2]])
		v23 = gaussian_overlap_vol([p, p], kappas[1:], centers[1:])
		assert v3 < v12
		assert v3 < v13
		assert v3 < v23


class TestGaussianBuriedVol:
	"""Tests for gaussian_buried_vol: verify buried volume results."""

	def test_single_atom_at_origin(self):
		"""Single atom at origin: buried volume should be positive and non-trivial.

		Note: with p=2.7, the soft Gaussian tails extend well beyond the hard-sphere
		radius, so the Gaussian result will be significantly larger than the
		hard-sphere value (R_atom/R_sphere)^3. This is expected and is why the
		steepness sweep in the benchmark explores different prefactors.
		"""
		R_atom = 1.70  # Carbon
		R_sphere = 3.5
		result = gaussian_buried_vol(
			np.array([[0.0, 0.0, 0.0]]),
			np.array([R_atom]),
			np.array([0.0, 0.0, 0.0]),
			R_sphere)
		# The Gaussian result should be positive and less than 100%
		assert result > 0.0
		assert result < 100.0
		# With soft Gaussians, expect roughly 15-30% for a carbon at origin in a 3.5A sphere
		assert result > 5.0
		assert result < 50.0

	def test_atom_far_outside_sphere(self):
		"""Atom far from origin should contribute ~0% buried volume."""
		result = gaussian_buried_vol(
			np.array([[50.0, 0.0, 0.0]]),
			np.array([1.70]),
			np.array([0.0, 0.0, 0.0]),
			3.5)
		assert result < 0.1

	def test_more_atoms_more_buried(self):
		"""Adding more atoms near origin should increase buried volume."""
		origin = np.array([0.0, 0.0, 0.0])
		R = 3.5
		# Single carbon
		v1 = gaussian_buried_vol(
			np.array([[0.0, 0.0, 1.5]]),
			np.array([1.70]),
			origin, R)
		# Two carbons
		v2 = gaussian_buried_vol(
			np.array([[0.0, 0.0, 1.5], [0.0, 1.5, 0.0]]),
			np.array([1.70, 1.70]),
			origin, R)
		assert v2 > v1

	def test_max_order_effect(self):
		"""Higher inclusion-exclusion order should refine the result for overlapping atoms."""
		coords = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]])
		radii = np.array([1.70, 1.70])
		origin = np.array([0.0, 0.0, 0.0])
		R = 3.5
		v_order1 = gaussian_buried_vol(coords, radii, origin, R, max_order=1)
		v_order2 = gaussian_buried_vol(coords, radii, origin, R, max_order=2)
		# Order 2 should subtract the overlap, giving a smaller value than order 1
		assert v_order2 < v_order1

	def test_returns_bounded_result(self):
		"""Result should be between 0 and 100."""
		coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]])
		radii = np.array([1.70, 1.55, 1.52])
		origin = np.array([0.0, 0.0, 0.0])
		result = gaussian_buried_vol(coords, radii, origin, 3.5)
		assert 0.0 <= result <= 100.0


class TestGaussianVsGrid:
	"""Cross-method validation: compare Gaussian and grid buried volumes."""

	@pytest.mark.parametrize("mol_file", ['H.xyz', 'Me.xyz', 'Et.xyz', 'tBu.xyz'])
	def test_agreement_with_grid(self, mol_file):
		"""Gaussian buried volume should be in reasonable agreement with grid method."""
		path = 'dbstep/examples/' + mol_file
		# Grid reference at fine spacing
		grid_mol = Dbstep.dbstep(path, volume=True, r=3.5, grid=0.05,
			commandline=True, quiet=True)
		# Gaussian method
		gauss_mol = Dbstep.dbstep(path, volume=True, r=3.5, method='gaussian',
			commandline=True, quiet=True)
		# Both should produce a numeric result
		assert grid_mol.bur_vol is not False
		assert gauss_mol.bur_vol is not False
		# Agreement within 15 percentage points (Gaussian is an approximation
		# using soft Gaussians, so tolerance must account for this)
		assert abs(grid_mol.bur_vol - gauss_mol.bur_vol) < 15.0, \
			f"{mol_file}: grid={grid_mol.bur_vol:.2f}%, gaussian={gauss_mol.bur_vol:.2f}%"
