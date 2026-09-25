"""The example notebooks must keep running: execute their code cells in order."""

import json
import os

import pytest

NOTEBOOK = "examples/proteins_and_conformers.ipynb"


def test_proteins_and_conformers_notebook_runs(monkeypatch, tmp_path):
	with open(NOTEBOOK) as f:
		notebook = json.load(f)
	code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
	assert len(code_cells) >= 6
	# the notebook resolves data relative to examples/; run it from a scratch copy so it does not litter the repo
	examples = tmp_path / "examples"
	examples.mkdir()
	(tmp_path / "tests").symlink_to(os.path.abspath("tests"), target_is_directory=True)
	monkeypatch.chdir(examples)
	monkeypatch.setenv("MPLBACKEND", "Agg")
	namespace = {}
	for cell in code_cells:
		exec("".join(cell["source"]), namespace)
	assert (examples / "1a8o_vbur.csv").exists()
	assert namespace["summary"][0]["structure"] == "boltzmann"
	assert sum(namespace["mol"].contributions.values()) == pytest.approx(namespace["mol"].bur_vol, abs=1e-5)
