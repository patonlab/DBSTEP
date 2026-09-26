"""--decompose: per-residue contributions to %V_bur."""

import csv
import re
import subprocess
import sys

import numpy as np
import pytest

from dbstep import Dbstep, sterics

pdb_dir = "tests/pdb_files/"
A8O = pdb_dir + "1a8o.pdb"
ALA5 = pdb_dir + "ala5.pdb"


def run(file, **kwargs):
	return Dbstep.dbstep(file, quiet=True, **kwargs)


@pytest.mark.parametrize("resid, flags", [("A:186", {}), ("A:186", {"nowater": True}), ("A:160", {"noH": True}), ("A:200", {"nowater": True, "nohet": True}), ("A:186", {"sterimol": True, "measure": "grid"})])
def test_contributions_add_up_to_buried_volume(resid, flags):
	mol = run(A8O, residue=resid, volume=True, decompose=True, **flags)
	assert mol.contributions
	assert all(value >= 0 for value in mol.contributions.values())
	# exact on the direct-lattice path; grid Sterimol takes the KD-tree path whose boundary arithmetic differs slightly
	tolerance = 1e-2 if flags.get("measure") == "grid" else 1e-5
	assert sum(mol.contributions.values()) == pytest.approx(mol.bur_vol, abs=tolerance)


def test_self_residue_is_the_top_contributor_and_vanishes_with_exclude_self():
	mol = run(A8O, residue="A:186", volume=True, nowater=True, decompose=True)
	top = max(mol.contributions, key=mol.contributions.get)
	assert top == "A:186 THR" == mol.residue_label
	env = run(A8O, residue="A:186", volume=True, nowater=True, decompose=True, exclude_self=True)
	assert env.contributions["A:186 THR"] == 0.0
	assert sum(env.contributions.values()) == pytest.approx(env.bur_vol, abs=1e-5)
	# every other residue contributes at least as much as before, since the ghosted residue no longer shares points
	for label, value in env.contributions.items():
		if label != "A:186 THR":
			assert value >= mol.contributions.get(label, 0.0) - 1e-9


def test_self_only_gives_a_single_contributor():
	mol = run(ALA5, residue="A:3", volume=True, decompose=True, self_only=True, cutoff="none")
	assert list(mol.contributions) == ["A:3 ALA"]
	assert mol.contributions["A:3 ALA"] == pytest.approx(mol.bur_vol, abs=1e-9)


def test_water_shows_up_only_when_kept():
	wet = run(ALA5, residue="A:3", volume=True, decompose=True)
	dry = run(ALA5, residue="A:3", volume=True, decompose=True, nowater=True)
	assert wet.contributions["A:101 HOH"] > 0
	assert not any("HOH" in label for label in dry.contributions)
	assert wet.contributions["A:101 HOH"] == pytest.approx(wet.bur_vol - dry.bur_vol, abs=0.5)


def test_scan_gives_one_decomposition_per_radius():
	mol = run(ALA5, residue="A:3", volume=True, decompose=True, scan="2.0:4.0:1.0")
	assert len(mol.contributions) == 3
	for contributions, vbur in zip(mol.contributions, mol.bur_vol):
		assert sum(contributions.values()) == pytest.approx(vbur, abs=1e-5)


def test_overlap_is_shared_equally():
	"""Two identical atoms on top of each other: each gets half of the occupied points."""
	x = np.round(np.linspace(-3, 3, 121), 8)
	coords = np.array([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [-2.0, 0.0, 0.0]])
	radii = np.array([1.0, 1.0, 0.5])
	shares = sterics.buried_vol_by_group(coords, radii, ["a", "b", "c"], x, x, x, np.zeros(3), 3.0)
	assert shares["a"] == pytest.approx(shares["b"])
	single = sterics.buried_vol_by_group(coords[:1], radii[:1], ["a"], x, x, x, np.zeros(3), 3.0)
	assert shares["a"] + shares["b"] == pytest.approx(single["a"])
	assert shares["c"] > 0 and list(shares) == ["a", "b", "c"]


def test_decompose_requires_pdb_and_vdw_surface():
	with pytest.raises(SystemExit, match="residue information"):
		run("dbstep/data/Et.xyz", atom1=2, volume=True, decompose=True)


def test_cli_prints_and_writes_contributions(tmp_path):
	out = tmp_path / "res.csv"
	cmd = [sys.executable, "-m", "dbstep", ALA5, "--residue", "A:3", "-b", "--decompose", "--csv", str(out)]
	result = subprocess.run(cmd, capture_output=True, text=True)
	assert result.returncode == 0, result.stdout + result.stderr
	text = result.stdout
	assert re.search(r"%V_Bur contributions for ala5\.pdb A:3 ALA \(R = 3\.50 Ang, total [\d.]+%\):", text)
	assert re.search(r"A:3 ALA\s+[\d.]+", text) and re.search(r"A:101 HOH\s+[\d.]+", text)
	contributions = tmp_path / "res_contributions.csv"
	assert f"Residue contributions written to {contributions}" in text
	with open(contributions, newline="") as f:
		rows = list(csv.DictReader(f))
	assert list(rows[0].keys()) == ["file", "frame", "structure", "residue", "radius", "contributor", "percent_vbur", "path"]
	assert {row["contributor"] for row in rows} >= {"A:3 ALA", "A:2 ALA", "A:4 ALA", "A:101 HOH"}
	table = float(re.search(r"ala5\.pdb A:3 ALA\s+CA\s+3\.50\s+[\d.]+\s+([\d.]+)", text).group(1))
	assert sum(float(row["percent_vbur"]) for row in rows) == pytest.approx(table, abs=0.006)
