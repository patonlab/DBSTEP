# CLAUDE.md

## Project Overview

DBSTEP (DFT-Based Steric Parameters) is a Python package for computing steric parameters from chemical structures. It calculates Sterimol parameters (L, Bmin, Bmax), percent buried volume, Sterimol2Vec, and Vol2Vec parameters from molecular structure files and quantum chemistry output files.

## Repository Structure

- `pyproject.toml` — Project metadata, dependencies, and tool configuration
- `CHANGELOG.md` — Release notes per version (update with every user-facing change)
- `dbstep/` — Main package source code
  - `Dbstep.py` — Core `dbstep` class, CLI entry point (`main()`, argparse in `build_parser()`), `all_residues()`, `all_frames()`, `from_rdkit()`
  - `calculator.py` — Math/geometry routines (rotations, angles)
  - `sterics.py` — Steric parameter calculations
  - `selection.py` — Atom selection before measurement: radial crop (`--cutoff`) and PDB residue selection (`--residue`, water/het/self filters)
  - `sterics.py` also holds `buried_vol_by_group` (per-residue %V_bur decomposition for `--decompose`)
  - `trajectory.py` — Frames of multi-structure files (`--frames` selection, frame counting)
  - `ensemble.py` — Boltzmann weighting over conformer ensembles (`--boltzmann`, energies from SDF data fields or xyz comments)
  - `parse_data.py` — Input file parsing (xyz, sdf/mol, pdb with residue metadata, cube, cclib-supported formats)
  - `constants.py` — Chemical constants (periodic table, Bondi radii, metals)
  - `graph.py` — 2D graph-based steric contribution calculations
  - `writer.py` — Output formatting and file writing (PyMOL scripts, xyz, CSV results)
  - `__init__.py` — Package init, `__version__`, `__all__`
  - `__main__.py` — Module entry point for `python -m dbstep`
  - `data/` — Benchmark molecular structure files (xyz format)
- `tests/` — Pytest test suite
  - `test_dbstep.py` — Sterimol parameter validation against Verloop's reference values
  - `test_calculator.py` — Unit tests for rotation/geometry math
  - `test_parse_data.py` — Input parsing tests (xyz, multi-structure xyz/sdf, cube)
  - `test_cli.py` — End-to-end tests of the `python -m dbstep` command line
  - `test_cube.py` — Density-based (cube file) buried volume and Sterimol tests
  - `test_crop.py` — `--cutoff` radial crop: exactness of %V_bur, renumbering, large-cluster equivalence
  - `test_pdb_parser.py` — PDB parsing: columns, element inference, metadata, altlocs, MODEL blocks
  - `test_protein.py` — `--residue` selection: exact equivalence with the XYZ path, crop invariance, water/het/self semantics
  - `test_residue_all.py` — `--residue all`, per-run `results` records and `--csv` output
  - `test_trajectory.py` — `--frames` parsing, per-frame runs on `ala5_traj.pdb`, multi-frame xyz, CSV time series
  - `test_decompose.py` — `--decompose` per-residue contributions: sums to %V_bur, overlap sharing, CLI/CSV
  - `test_examples.py` — Executes the code cells of `examples/proteins_and_conformers.ipynb`
  - `test_ensemble.py` — SDF data fields, Boltzmann weights/averages, CLI `--boltzmann` on `sdf_files/ether_conformers.sdf` (AQME output)
  - `cube_files/` — Test cube file fixtures (keep these small; use `benzene_coarse.cube` / `Ne_medium.cube` for new tests)
  - `sdf_files/` — `ether_conformers.sdf`: three diethyl ether conformers from AQME with `<Energy>` fields (kcal/mol)
  - `pdb_files/` — PDB fixtures: `1a8o.pdb` (real, no H, waters, MSE) and generated `ala5.pdb` (with H, waters, Na) and `ala5_traj.pdb` (10 models, water approaching A:3); see its README
- `examples/` — Jupyter notebook examples (`proteins_and_conformers.ipynb` is kept runnable by `tests/test_examples.py`; `carbene_sterics.ipynb` predates 2.0)
- `analysis/` — Standalone analysis scripts and data behind the paper (not part of the package; linted by ruff; extra deps via `uv sync --group analysis`)
- `reference/` — Reference data
- `docs/plans/` — Design and implementation plans (e.g. proteins and trajectories for 2.0)
- `.github/workflows/ci.yml` — GitHub Actions CI (test, lint)
- `.github/workflows/release.yml` — Publishes to PyPI when a GitHub Release is published
- `.github/dependabot.yml` — Monthly grouped updates for uv.lock and GitHub Actions

## Development Commands

### Install (using uv)
```
uv sync
```

### Install with dev tools
```
uv sync --extra dev
```

### Install extras for the analysis scripts (matplotlib, pandas, rdkit, tqdm)
```
uv sync --group analysis
```

### Run tests
```
uv run pytest
```

### Lint
```
uv run ruff check .
```

### Run the tool
```
uv run dbstep <file> --sterimol --atom1 <idx> --atom2 <idx>
```

### Build
```
uv build
```

### Release
Add a CHANGELOG.md entry, bump `__version__` in `dbstep/__init__.py` (and the version in `meta.yaml`), merge to master, then publish a GitHub Release whose tag is the version (e.g. `1.2.0`). `.github/workflows/release.yml` checks the tag against `__version__`, runs the tests, uploads to PyPI via trusted publishing and attaches the sdist/wheel to the release.

## Key Dependencies

- numpy, scipy, cclib
- Optional: RDKit, pandas (install with `uv sync --extra graph2d`)
- Dev tools: pytest, ruff, pre-commit (install with `uv sync --extra dev`)

## Code Conventions

- Uses tabs for indentation throughout
- Python 3.10+ required (CI runs 3.10 through 3.14)
- Main class is lowercase `dbstep` in `dbstep/Dbstep.py`
- Atom indexing is 1-based (matching chemical structure file conventions)
- Tests compare computed values against Verloop's reference Sterimol parameters with a tolerance of 0.01
- Version is defined in `dbstep/__init__.py` as the single source of truth
