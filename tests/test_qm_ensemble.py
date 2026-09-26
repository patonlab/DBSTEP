"""Boltzmann weighting across separate QM output files (energies via cclib) and --energy-window.

Fixtures: tests/qm_files/pentane_*.out.gz, the nine M06-2X/6-31G(d) pentane conformers of the wSterimol
example (patonlab/wsterimol, example_gaussian), trimmed to the last optimisation step and gzipped.
wSterimol's weighted.txt (CPK radii, 298 K, atoms 1 and 3): wL 6.33, wB1 1.79, wB5 3.75, populations
pentane_18 27.27 %, 19/22/7/10 12.22 %, 20/1 11.47 %, 21/9 0.45 %.
"""

import csv
import glob
import subprocess
import sys

import pytest

from dbstep import Dbstep, ensemble

FILES = sorted(glob.glob("tests/qm_files/pentane_*.out.gz"))
ETHER = "tests/sdf_files/ether_conformers.sdf"
WSTERIMOL = {"pentane_18": 0.2727, "pentane_19": 0.1222, "pentane_22": 0.1222, "pentane_7": 0.1222, "pentane_10": 0.1222, "pentane_20": 0.1147, "pentane_1": 0.1147, "pentane_21": 0.0045, "pentane_9": 0.0045}


def run(file, **kwargs):
	return Dbstep.dbstep(file, quiet=True, **kwargs)


def run_cli(*args):
	result = subprocess.run([sys.executable, "-m", "dbstep", *args], capture_output=True, text=True)
	assert result.returncode == 0, result.stdout + result.stderr
	return result.stdout


def read_csv(path):
	with open(path, newline="") as f:
		return list(csv.DictReader(f))


def stem(row):
	return row["file"].split(".")[0]


# --- energies from QM outputs -----------------------------------------------------------------------------


def test_cclib_parser_exposes_energies_in_hartree():
	assert len(FILES) == 9
	mol = run("tests/qm_files/pentane_18.out.gz", atom1=1, atom2=3, sterimol=True)
	assert mol.properties["E"] == pytest.approx(-197.650408, abs=2e-6)  # Etot (Hartree) in wSterimol's archive
	assert mol.properties["energy"] == mol.properties["E"] and mol.properties["energy_source"] == "E"
	assert mol.properties["energy_units"] == "hartree" and "G" not in mol.properties  # no frequency calculation
	assert ensemble.energy_of(mol) == pytest.approx(-197.650408 * 627.509474, rel=1e-8)  # kcal/mol whatever the CLI units
	assert ensemble.energy_and_key(mol)[1] == "SCF energy"
	assert mol.n_atoms_total == 17 and mol.Bmax > mol.Bmin > 0


def test_gzipped_output_matches_wsterimol_per_conformer():
	mol = run("tests/qm_files/pentane_18.out.gz", atom1=1, atom2=3, sterimol=True, radii="cpk")
	assert (round(mol.L + 0.4, 2), round(mol.Bmin, 2), round(mol.Bmax, 2)) == pytest.approx((7.39, 1.67, 2.74), abs=0.011)


# --- pooled ensembles ---------------------------------------------------------------------------------------


def test_cli_pools_single_structure_files_and_reproduces_wsterimol(tmp_path):
	out = tmp_path / "pentane.csv"
	text = run_cli(*FILES, "--sterimol", "--atom1", "1", "--atom2", "3", "--radii", "cpk", "--boltzmann", "--temperature", "298", "--csv", str(out))
	rows = read_csv(out)
	assert len(rows) == 10 and rows[-1]["file"] == "ensemble" and rows[-1]["structure"] == "boltzmann" and rows[-1]["path"] == ""
	weighted = rows[-1]
	assert float(weighted["L"]) + 0.4 == pytest.approx(6.33, abs=0.015)
	assert float(weighted["bmin"]) == pytest.approx(1.79, abs=0.015) and float(weighted["bmax"]) == pytest.approx(3.75, abs=0.015)
	for row in rows[:-1]:
		# wSterimol used slightly different constants (hartree -> kcal/mol, R), hence the loose tolerance
		assert float(row["population"]) == pytest.approx(WSTERIMOL[stem(row)], abs=0.003)
	assert "ensemble boltzmann" in text and "Energies of ensemble: SCF energy read from the output files" in text
	assert "Boltzmann populations at 298.00 K (hartree): pentane_1.out.gz 0.114" in text


def test_python_api_over_separate_files():
	runs = [run(f, atom1=1, atom2=3, sterimol=True, radii="cpk") for f in FILES]
	summary = ensemble.boltzmann_average(runs, temperature=298.0, label="pentane")
	assert summary[0]["file"] == "pentane" and summary[0]["structure"] == "boltzmann"
	assert summary[0]["L"] + 0.4 == pytest.approx(6.33, abs=0.015)
	by_name = {r.results[0]["file"].split(".")[0]: r for r in runs}
	assert by_name["pentane_18"].energy_rel == 0.0 and by_name["pentane_21"].energy_rel == pytest.approx(2.44, abs=0.03)
	assert by_name["pentane_18"].energy == pytest.approx(-197.650408 * 627.509474, rel=1e-8)
	assert all(r.in_window for r in runs) and {r.energy_key for r in runs} == {"SCF energy"}
	assert sum(r.population for r in runs) == pytest.approx(1.0)


def test_explicit_units_do_not_override_output_file_units():
	runs = [run(f, atom1=1, atom2=3, sterimol=True) for f in FILES[:3]]
	default = ensemble.boltzmann_average(runs)
	pops = [r.population for r in runs]
	ensemble.boltzmann_average(runs, units="hartree")
	assert [r.population for r in runs] == pytest.approx(pops) and ensemble.boltzmann_average(runs, units="kcal")[0]["L"] == pytest.approx(default[0]["L"])


def test_choosing_the_energy_field_by_tag():
	runs = [run(f, atom1=1, atom2=3, sterimol=True) for f in FILES[:3]]
	ensemble.boltzmann_average(runs, tag="E")
	pops = [r.population for r in runs]
	ensemble.boltzmann_average(runs)
	assert [r.population for r in runs] == pytest.approx(pops)
	with pytest.raises(SystemExit, match="No <G> data field"):
		ensemble.boltzmann_average(runs, tag="G")
	with pytest.raises(SystemExit, match="No <energy_units> data field"):
		ensemble.boltzmann_average(runs, tag="energy_units")


def test_pooling_rule(tmp_path):
	"""Several multi-structure files keep one summary per file; single-structure xyz files are pooled."""
	a, b = tmp_path / "a.sdf", tmp_path / "b.sdf"
	a.write_text(open(ETHER).read())
	b.write_text(open(ETHER).read())
	out = tmp_path / "two.csv"
	run_cli(str(a), str(b), "--atom1", "3", "--atom2", "2", "--sterimol", "--boltzmann", "--csv", str(out))
	rows = read_csv(out)
	assert [row["file"] for row in rows if row["structure"] == "boltzmann"] == ["a.sdf", "b.sdf"]
	# three single-structure xyz files with energies in the comment line form one ensemble
	names = []
	for i, energy in enumerate((0.0, 1.0, 5.0)):
		src = open("dbstep/data/Et.xyz").read().splitlines()
		src[1] = f"ethane conformer E = {energy:.1f} kcal/mol"
		xyz = tmp_path / f"c{i}.xyz"
		xyz.write_text("\n".join(src) + "\n")
		names.append(str(xyz))
	out = tmp_path / "xyz.csv"
	text = run_cli(*names, "--atom1", "2", "--atom2", "5", "--sterimol", "--boltzmann", "--csv", str(out))
	rows = read_csv(out)
	assert [row["file"] for row in rows] == ["c0.xyz", "c1.xyz", "c2.xyz", "ensemble"]
	assert float(rows[0]["population"]) > 0.8 and "ensemble boltzmann" in text
	# a single file is never pooled: the old per-file behaviour
	text = run_cli(ETHER, "--atom1", "3", "--atom2", "2", "--sterimol", "--boltzmann")
	assert "ether_conformers.sdf boltzmann" in text


# --- energy window --------------------------------------------------------------------------------------------


def test_energy_window_excludes_high_conformers_but_keeps_them_listed(tmp_path):
	out = tmp_path / "window.csv"
	text = run_cli(*FILES, "--sterimol", "--atom1", "1", "--atom2", "3", "--radii", "cpk", "--boltzmann", "--temperature", "298", "--energy-window", "1.0", "--csv", str(out))
	rows = read_csv(out)
	assert len(rows) == 10
	excluded = [stem(row) for row in rows[:-1] if float(row["population"]) == 0.0]
	assert sorted(excluded) == ["pentane_21", "pentane_9"]
	assert sum(float(row["population"]) for row in rows[:-1]) == pytest.approx(1.0, abs=1e-6)
	# wSterimol: the seven conformers within 1 kcal/mol account for 99.1 % of the population, so renormalising changes little
	assert float(next(row for row in rows if stem(row) == "pentane_18")["population"]) == pytest.approx(0.2727 / 0.991, abs=0.003)
	assert "Energy window 1.00 kcal/mol: 7 of 9 structures weighted" in text
	assert float(rows[-1]["L"]) + 0.4 == pytest.approx(6.34, abs=0.015)


def test_energy_window_python_api_and_validation():
	runs = Dbstep.all_frames(ETHER, atom1=3, atom2=2, sterimol=True, quiet=True)
	ensemble.boltzmann_average(runs, window=2.0)
	assert [r.in_window for r in runs] == [True, True, False]  # relative energies 0, 1.52, 3.03 kcal/mol
	assert runs[2].population == 0.0 and runs[0].population + runs[1].population == pytest.approx(1.0)
	assert runs[2].results[0]["population"] == 0.0
	with pytest.raises(SystemExit, match="must be positive"):
		ensemble.boltzmann_average(runs, window=0)
	# a window wider than the spread changes nothing
	full = ensemble.boltzmann_average(runs)
	assert ensemble.boltzmann_average(runs, window=100.0)[0]["bmax"] == pytest.approx(full[0]["bmax"])
	with pytest.raises(SystemExit, match="must be positive"):
		ensemble.boltzmann_average(runs, window=-1)
	result = subprocess.run([sys.executable, "-m", "dbstep", ETHER, "--sterimol", "--boltzmann", "--energy-window", "0"], capture_output=True, text=True)
	assert result.returncode != 0 and "must be positive" in result.stdout + result.stderr
