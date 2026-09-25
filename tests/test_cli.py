"""End-to-end tests of the command line entry point (python -m dbstep)."""

import re
import subprocess
import sys

import pytest

XYZ_DIR = "dbstep/data/"


def run_cli(*args):
	"""Run dbstep as a module and return its combined stdout/stderr."""
	result = subprocess.run([sys.executable, "-m", "dbstep", *args], capture_output=True, text=True)
	assert result.returncode == 0, result.stdout + result.stderr
	return result.stdout + result.stderr


def test_sterimol_readme_example():
	"""README example 1: Sterimol parameters for ethane along C2-C5."""
	out = run_cli(XYZ_DIR + "Et.xyz", "--sterimol", "--atom1", "2", "--atom2", "5")
	assert re.search(r"Et\.xyz\s+2\s+5\s+1\.99\s+2\.13\s+3\.24", out)


@pytest.mark.parametrize("flag", ["-b", "--vbur"])
def test_vbur_readme_example(flag):
	"""README example 3: %Vbur for 1-naphthyl, using either spelling of the flag (see issue #37)."""
	out = run_cli(XYZ_DIR + "1Nap.xyz", "--atom1", "2", flag)
	assert re.search(r"1Nap\.xyz\s+2\s+3\.50\s+118\.65\s+41\.77\s+0\.00", out)


def test_dp_controls_decimal_places():
	out = run_cli(XYZ_DIR + "1Nap.xyz", "--atom1", "2", "--vbur", "--dp", "3")
	assert re.search(r"1Nap\.xyz\s+2\s+3\.500\s+118\.6\d\d\s+41\.7\d\d\s+0\.000", out)


def test_multi_file_run_with_noH_gives_identical_results():
	"""Renumbered spec atoms from --noH must not leak from one file into the next."""
	out = run_cli(XYZ_DIR + "Et.xyz", XYZ_DIR + "Et.xyz", "--sterimol", "--atom1", "5", "--atom2", "2", "--noH")
	rows = re.findall(r"Et\.xyz\s+5\s+2\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", out)
	assert len(rows) == 2
	assert rows[0] == rows[1]


@pytest.mark.parametrize("filename", ["all.xyz", "all.sdf"])
def test_multi_structure_file_labels_each_structure(filename):
	out = run_cli(XYZ_DIR + filename, "--sterimol", "--atom1", "2", "--atom2", "1")
	assert re.search(r"1nap\s+2\s+1\s+1\.70\s+5\.59\s+4\.91", out)
	assert re.search(r"Ad\s+2\s+1\s+3\.25\s+3\.58\s+4\.76", out)


def test_quiet_suppresses_output():
	out = run_cli(XYZ_DIR + "Et.xyz", "--sterimol", "--atom1", "2", "--atom2", "5", "--quiet")
	assert out.strip() == ""
