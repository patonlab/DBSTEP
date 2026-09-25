"""Density-based (cube file) steric measurements."""

import math

import pytest

from dbstep import Dbstep

cube_dir = "tests/cube_files/"


def test_neon_density_volume_is_a_sphere():
	"""A lone Ne atom: the isodensity volume is a sphere, and %V_bur at r = 2 Å is that volume over the sphere volume."""
	mol = Dbstep.dbstep(cube_dir + "Ne_medium.cube", atom1=1, volume=True, r=2.0, quiet=True)
	r_iso = (3 * mol.occ_vol / (4 * math.pi)) ** (1 / 3)
	assert 1.4 < r_iso < 1.6  # close to the Bondi radius of Ne (1.54 Å)
	expected = mol.occ_vol / (4 / 3 * math.pi * 2.0**3) * 100
	assert mol.bur_vol == pytest.approx(expected, abs=0.5)


def test_benzene_density_buried_volume():
	mol = Dbstep.dbstep(cube_dir + "benzene_coarse.cube", atom1=1, volume=True, quiet=True)
	assert mol.occ_vol == pytest.approx(102.8, abs=0.5)
	assert mol.bur_vol == pytest.approx(33.97, abs=0.1)


def test_benzene_density_grid_sterimol():
	mol = Dbstep.dbstep(cube_dir + "benzene_coarse.cube", atom1=1, atom2=2, sterimol=True, measure="grid", quiet=True)
	assert mol.Bmin == pytest.approx(1.85, abs=0.05)
	assert mol.Bmax == pytest.approx(3.38, abs=0.05)
	assert mol.L == pytest.approx(6.16, abs=0.05)


def test_ch2cme3_density_buried_volume():
	mol = Dbstep.dbstep(cube_dir + "CH2CMe3_100.cube", atom1=1, volume=True, quiet=True)
	assert mol.occ_vol == pytest.approx(116.69, abs=0.5)
	assert mol.bur_vol == pytest.approx(39.83, abs=0.1)
