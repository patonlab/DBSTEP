"""Conformer ensembles: SDF data fields, Boltzmann weights and weighted averages."""

import csv
import math
import re
import subprocess
import sys

import numpy as np
import pytest

from dbstep import Dbstep, ensemble, parse_data

ETHER = "tests/sdf_files/ether_conformers.sdf"
ENERGIES = [1.6241422178985285, 3.14336556931666, 4.656221082683384]  # kcal/mol, from the <Energy> fields
TITLES = ["ether 44", "ether 12", "ether 6"]


def run_cli(*args):
	result = subprocess.run([sys.executable, "-m", "dbstep", *args], capture_output=True, text=True)
	assert result.returncode == 0, result.stdout + result.stderr
	return result.stdout


def ether_runs(**kwargs):
	return Dbstep.all_frames(ETHER, atom1=3, atom2=2, sterimol=True, volume=True, quiet=True, **kwargs)


# --- parsing ------------------------------------------------------------------------------------------


def test_sdf_data_fields_and_titles():
	for idx, (title, energy) in enumerate(zip(TITLES, ENERGIES)):
		mol = parse_data.read_input(ETHER, ".sdf", Dbstep.set_options({"atom1": 1, "atom2": [2], "structure": idx}))
		assert mol.structure_name == title
		assert list(mol.PROPERTIES) == ["Energy", "Real charge", "Mult", "SMILES"]
		assert float(mol.PROPERTIES["Energy"]) == energy
		assert mol.PROPERTIES["SMILES"] == "CCOCC"
		assert len(mol.ATOMTYPES) == 15


def test_sdf_data_field_parser_handles_multiline_and_missing_blank_line():
	lines = [">  <Note>  (1) \n", "line one\n", "line two\n", "\n", "> <Last>\n", "42\n"]
	assert parse_data._parse_sdf_data_fields(lines) == {"Note": "line one\nline two", "Last": "42"}


def test_conformer_titles_are_kept_whole_in_results():
	runs = ether_runs()
	assert [r.results[0]["structure"] for r in runs] == TITLES


# --- weights ----------------------------------------------------------------------------------------


def test_boltzmann_weights_by_hand():
	rt = 0.0019872041 * 298.15
	expected = np.exp(-(np.array(ENERGIES) - ENERGIES[0]) / rt)
	expected /= expected.sum()
	assert ensemble.boltzmann_weights(ENERGIES) == pytest.approx(expected)
	assert expected[0] == pytest.approx(0.923, abs=0.002)  # dominated by the lowest conformer
	assert ensemble.boltzmann_weights(ENERGIES).sum() == pytest.approx(1.0)


def test_weights_units_and_temperature():
	kcal = ensemble.boltzmann_weights(ENERGIES)
	assert ensemble.boltzmann_weights([e * 4.184 for e in ENERGIES], units="kJ") == pytest.approx(kcal)
	assert ensemble.boltzmann_weights([e / 627.509474 for e in ENERGIES], units="hartree") == pytest.approx(kcal)
	assert ensemble.boltzmann_weights([e / 23.060548 for e in ENERGIES], units="eV") == pytest.approx(kcal)
	hot = ensemble.boltzmann_weights(ENERGIES, temperature=1e7)
	assert hot == pytest.approx([1 / 3] * 3, abs=1e-4)
	cold = ensemble.boltzmann_weights(ENERGIES, temperature=10.0)
	assert cold[0] == pytest.approx(1.0) and cold[1] < 1e-10
	with pytest.raises(SystemExit):
		ensemble.boltzmann_weights(ENERGIES, units="furlongs")
	with pytest.raises(SystemExit):
		ensemble.boltzmann_weights(ENERGIES, temperature=0)


@pytest.mark.parametrize("text, expected", [("1.62", 1.62), ("ether 44", None), ("-40.123456", -40.123456), ("conf 3 E=-12.5 kcal", -12.5), ("1e-3", 1e-3), ("", None)])
def test_number_in_text(text, expected):
	assert ensemble.number_in_text(text) == expected


# --- averages ---------------------------------------------------------------------------------------


def test_boltzmann_average_matches_manual_weighting():
	runs = ether_runs()
	summary = ensemble.boltzmann_average(runs)
	weights = ensemble.boltzmann_weights(ENERGIES)
	assert [r.population for r in runs] == pytest.approx(list(weights))
	assert [r.energy for r in runs] == pytest.approx(ENERGIES)
	assert len(summary) == 1
	row = summary[0]
	assert (row["structure"], row["frame"], row["population"], row["file"]) == ("boltzmann", "", 1.0, "ether_conformers.sdf")
	for key, attr in (("percent_vbur", "bur_vol"), ("L", "L"), ("bmin", "Bmin"), ("bmax", "Bmax"), ("mol_vol", "occ_vol")):
		assert row[key] == pytest.approx(sum(w * getattr(r, attr) for w, r in zip(weights, runs)))
	assert all(r.results[0]["population"] == r.population for r in runs)
	# the conformers differ, so the average is a genuine mix, not one of them
	assert min(r.L for r in runs) < row["L"] < max(r.L for r in runs)


def test_explicit_tag_equals_auto_and_missing_tag_errors():
	runs = ether_runs()
	assert ensemble.boltzmann_average(runs, tag="energy") == ensemble.boltzmann_average(runs)
	with pytest.raises(SystemExit, match="No <Gibbs> data field"):
		ensemble.boltzmann_average(runs, tag="Gibbs")
	with pytest.raises(SystemExit, match="is not a number"):
		ensemble.boltzmann_average(runs, tag="SMILES")


def test_lowest_conformer_dominates_at_low_temperature():
	runs = ether_runs()
	summary = ensemble.boltzmann_average(runs, temperature=5.0)
	assert summary[0]["percent_vbur"] == pytest.approx(runs[0].bur_vol, abs=1e-9)
	assert summary[0]["L"] == pytest.approx(runs[0].L, abs=1e-9)


def test_scan_gives_one_summary_row_per_radius():
	runs = ether_runs(scan="2.0:4.0:1.0")
	summary = ensemble.boltzmann_average(runs)
	assert [row["radius"] for row in summary] == [2.0, 3.0, 4.0]
	weights = ensemble.boltzmann_weights(ENERGIES)
	for i, row in enumerate(summary):
		assert row["percent_vbur"] == pytest.approx(sum(w * r.bur_vol[i] for w, r in zip(weights, runs)))


def test_frames_restrict_the_ensemble():
	runs = Dbstep.all_frames(ETHER, frames="0,2", atom1=3, volume=True, quiet=True)
	ensemble.boltzmann_average(runs)
	expected = ensemble.boltzmann_weights([ENERGIES[0], ENERGIES[2]])
	assert [r.population for r in runs] == pytest.approx(list(expected))


def test_xyz_comment_line_energies(tmp_path):
	"""CREST-style multi-frame xyz: the comment line holds the energy (here in hartree)."""
	src = parse_data.read_input("dbstep/data/Et.xyz", ".xyz", Dbstep.set_options({"atom1": 1, "atom2": [2]}))
	xyz = tmp_path / "confs.xyz"
	with open(xyz, "w") as f:
		for energy in (-79.8000, -79.7990):
			f.write(f"{len(src.ATOMTYPES)}\n{energy:.6f}\n")
			for atom, (x, y, z) in zip(src.ATOMTYPES, src.CARTESIANS):
				f.write(f"{atom} {x:.4f} {y:.4f} {z:.4f}\n")
	runs = Dbstep.all_frames(str(xyz), atom1=2, atom2=5, sterimol=True, quiet=True)
	assert [r.properties["comment"] for r in runs] == ["-79.800000", "-79.799000"]
	ensemble.boltzmann_average(runs, units="hartree")
	expected = ensemble.boltzmann_weights([-79.8, -79.799], units="hartree")
	assert [r.population for r in runs] == pytest.approx(list(expected))
	assert expected[0] > 0.6  # 0.001 hartree = 0.63 kcal/mol


def test_missing_energy_is_an_error():
	runs = Dbstep.all_frames("dbstep/data/all.xyz", frames="0,1", atom1=2, volume=True, quiet=True)
	with pytest.raises(SystemExit, match="No energy found"):
		ensemble.boltzmann_average(runs)


# --- command line ----------------------------------------------------------------------------------


def test_cli_boltzmann_table_and_csv(tmp_path):
	out = tmp_path / "ether.csv"
	text = run_cli(ETHER, "--atom1", "3", "--atom2", "2", "-s", "-b", "--boltzmann", "--csv", str(out))
	assert re.search(r"ether 44\s+3\s+2\s+3\.50", text) and re.search(r"ether_conformers\.sdf boltzmann\s+3\s+2\s+3\.50", text)
	assert re.search(r"Boltzmann populations at 298\.15 K \(kcal/mol\): ether 44 0\.92\d, ether 12 0\.07\d, ether 6 0\.00\d", text)
	with open(out, newline="") as f:
		rows = list(csv.DictReader(f))
	assert [row["structure"] for row in rows] == TITLES + ["boltzmann"]
	populations = [float(row["population"]) for row in rows[:3]]
	assert sum(populations) == pytest.approx(1.0)
	weighted = sum(p * float(row["percent_vbur"]) for p, row in zip(populations, rows[:3]))
	assert float(rows[3]["percent_vbur"]) == pytest.approx(weighted, abs=1e-6)
	assert rows[3]["frame"] == "" and rows[3]["population"] == "1.0"


def test_cli_boltzmann_with_tag_units_and_temperature():
	text = run_cli(ETHER, "--atom1", "3", "-b", "--boltzmann", "Energy", "--energy-units", "kJ", "--temperature", "500")
	match = re.search(r"Boltzmann populations at 500\.00 K \(kJ/mol\): ether 44 ([\d.]+)", text)
	assert match
	expected = ensemble.boltzmann_weights(ENERGIES, temperature=500.0, units="kJ")[0]
	assert float(match.group(1)) == pytest.approx(expected, abs=5e-4)
	assert not math.isclose(expected, ensemble.boltzmann_weights(ENERGIES)[0])
