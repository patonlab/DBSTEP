"""Frames of multi-structure files: --frames selection, per-frame runs and CSV output."""

import csv
import re
import subprocess
import sys

import numpy as np
import pytest

from dbstep import Dbstep, trajectory

pdb_dir = "tests/pdb_files/"
TRAJ = pdb_dir + "ala5_traj.pdb"
ALL_XYZ = "dbstep/data/all.xyz"


def run(file, **kwargs):
	return Dbstep.dbstep(file, quiet=True, **kwargs)


def run_cli(*args):
	result = subprocess.run([sys.executable, "-m", "dbstep", *args], capture_output=True, text=True)
	assert result.returncode == 0, result.stdout + result.stderr
	return result.stdout


# --- frame selection ------------------------------------------------------------------------------


@pytest.mark.parametrize(
	"spec, expected",
	[
		(False, list(range(10))),
		("", list(range(10))),
		("3", [3]),
		("-1", [9]),
		("2:8:2", [2, 4, 6]),
		("::5", [0, 5]),
		("7:", [7, 8, 9]),
		(":3", [0, 1, 2]),
		("0,9,4:6", [0, 9, 4, 5]),
		([1, 2], [1, 2]),
	],
)
def test_parse_frames(spec, expected):
	assert trajectory.parse_frames(spec, 10) == expected


@pytest.mark.parametrize("spec", ["10", "-11", "abc", "1:2:3:4", "20:30"])
def test_parse_frames_errors(spec):
	with pytest.raises(SystemExit):
		trajectory.parse_frames(spec, 10)


def test_count_frames():
	assert trajectory.count_frames(TRAJ) == 10
	assert trajectory.count_frames(ALL_XYZ) == 22
	assert trajectory.count_frames("dbstep/data/Et.xyz") == 1
	assert trajectory.count_frames(pdb_dir + "1a8o.pdb") == 1


def test_frames_on_single_structure_file():
	options = Dbstep.set_options({})
	assert trajectory.frame_indices("dbstep/data/Et.xyz", options) == [None]
	options.frames = "0"
	assert trajectory.frame_indices("dbstep/data/Et.xyz", options) == [0]
	options.frames = "1"
	with pytest.raises(SystemExit, match="out of range"):
		trajectory.frame_indices("dbstep/data/Et.xyz", options)


# --- trajectory runs -------------------------------------------------------------------------------


def test_water_approach_raises_buried_volume_frame_by_frame():
	runs = Dbstep.all_frames(TRAJ, residue="A:3", volume=True, quiet=True)
	assert len(runs) == 10
	vbur = [r.bur_vol for r in runs]
	assert all(later > earlier for earlier, later in zip(vbur, vbur[1:]))
	assert [row["frame"] for r in runs for row in r.results] == list(range(10))
	assert [row["structure"] for r in runs for row in r.results] == [f"model{i}" for i in range(1, 11)]


def test_rigid_translation_leaves_results_unchanged_without_water():
	runs = Dbstep.all_frames(TRAJ, residue="A:3", volume=True, sterimol=True, nowater=True, quiet=True)
	first = (runs[0].L, runs[0].Bmin, runs[0].Bmax, runs[0].bur_vol)
	for mol in runs[1:]:
		assert (mol.L, mol.Bmin, mol.Bmax, mol.bur_vol) == pytest.approx(first, abs=1e-9)


def test_frames_selection_and_equivalence_with_extracted_models(tmp_path):
	runs = Dbstep.all_frames(TRAJ, frames="2:8:2", residue="A:3", volume=True, quiet=True)
	assert [r.results[0]["frame"] for r in runs] == [2, 4, 6]
	lines = open(TRAJ).read().splitlines()
	for mol in runs:
		frame = mol.results[0]["frame"]
		start = lines.index(f"MODEL     {frame + 1:4d}")
		end = lines.index("ENDMDL", start)
		single = tmp_path / f"model{frame}.pdb"
		single.write_text("\n".join(lines[start + 1:end] + ["END"]) + "\n")
		ref = run(str(single), residue="A:3", volume=True)
		assert mol.bur_vol == pytest.approx(ref.bur_vol, abs=1e-9)
		assert np.allclose(mol.coords, ref.coords)


def test_all_frames_times_all_residues():
	runs = Dbstep.all_frames(TRAJ, frames="0,9", residue="all", volume=True, grid=0.2, quiet=True)
	assert len(runs) == 10
	assert [r.results[0]["frame"] for r in runs] == [0] * 5 + [9] * 5
	assert [r.residue_label for r in runs[:5]] == [f"A:{i} ALA" for i in range(1, 6)]


def test_multi_frame_xyz_with_frames():
	runs = Dbstep.all_frames(ALL_XYZ, frames="::7", atom1=2, atom2=1, sterimol=True, quiet=True)
	assert [r.results[0]["frame"] for r in runs] == [0, 7, 14, 21]
	assert runs[0].results[0]["structure"] == "1nap"


# --- command line ----------------------------------------------------------------------------------


def test_cli_frames_and_csv(tmp_path):
	out = tmp_path / "traj.csv"
	text = run_cli(TRAJ, "--residue", "A:3", "-b", "--frames", "::3", "--csv", str(out))
	assert "Frames ::3 of each multi-structure file will be run" in text
	assert re.search(r"model1 A:3 ALA\s+CA\s+3\.50", text) and "model10 A:3 ALA" in text
	with open(out, newline="") as f:
		rows = list(csv.DictReader(f))
	assert [row["frame"] for row in rows] == ["0", "3", "6", "9"]
	assert [row["structure"] for row in rows] == ["model1", "model4", "model7", "model10"]
	assert all(row["residue"] == "A:3 ALA" for row in rows)
	vbur = [float(row["percent_vbur"]) for row in rows]
	assert vbur == sorted(vbur) and vbur[0] < vbur[-1]


def test_cli_frames_out_of_range_reports_count():
	result = subprocess.run([sys.executable, "-m", "dbstep", TRAJ, "--residue", "A:3", "-b", "--frames", "12"], capture_output=True, text=True)
	assert result.returncode != 0
	assert "out of range (file has 10 frames)" in result.stdout + result.stderr
