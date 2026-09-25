# CLAUDE.md

## Project Overview

DBSTEP (DFT-Based Steric Parameters) is a Python package for computing steric parameters from chemical structures. It calculates Sterimol parameters (L, Bmin, Bmax), percent buried volume, Sterimol2Vec, and Vol2Vec parameters from molecular structure files and quantum chemistry output files.

## Repository Structure

- `pyproject.toml` — Project metadata, dependencies, and tool configuration
- `dbstep/` — Main package source code
  - `Dbstep.py` — Core `dbstep` class and CLI entry point (`main()`)
  - `calculator.py` — Math/geometry routines (rotations, angles)
  - `sterics.py` — Steric parameter calculations
  - `selection.py` — Atom selection before measurement (radial crop around atom1 for `--cutoff`)
  - `parse_data.py` — Input file parsing (xyz, cube, cclib-supported formats)
  - `constants.py` — Chemical constants (periodic table, Bondi radii, metals)
  - `graph.py` — 2D graph-based steric contribution calculations
  - `writer.py` — Output formatting and file writing
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
  - `cube_files/` — Test cube file fixtures (keep these small; use `benzene_coarse.cube` / `Ne_medium.cube` for new tests)
- `examples/` — Jupyter notebook examples
- `analysis/` — Standalone analysis scripts and data behind the paper (not part of the package; linted by ruff; extra deps via `uv sync --group analysis`)
- `reference/` — Reference data
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
Bump `__version__` in `dbstep/__init__.py` (and the version in `meta.yaml`), merge to master, then publish a GitHub Release whose tag is the version (e.g. `1.2.0`). `.github/workflows/release.yml` checks the tag against `__version__`, runs the tests, uploads to PyPI via trusted publishing and attaches the sdist/wheel to the release.

## Key Dependencies

- numpy, scipy, cclib
- Optional: RDKit, pandas (install with `uv sync --extra graph2d`)
- Dev tools: pytest, ruff, pre-commit (install with `uv sync --extra dev`)

## Code Conventions

- Uses tabs for indentation throughout
- Python 3.9+ required
- Main class is lowercase `dbstep` in `dbstep/Dbstep.py`
- Atom indexing is 1-based (matching chemical structure file conventions)
- Tests compare computed values against Verloop's reference Sterimol parameters with a tolerance of 0.01
- Version is defined in `dbstep/__init__.py` as the single source of truth
