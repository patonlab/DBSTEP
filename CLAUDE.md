# CLAUDE.md

## Project Overview

DBSTEP (DFT-Based Steric Parameters) is a Python package for computing steric parameters from chemical structures. It calculates Sterimol parameters (L, Bmin, Bmax), percent buried volume, Sterimol2Vec, and Vol2Vec parameters from molecular structure files and quantum chemistry output files.

## Repository Structure

- `pyproject.toml` — Project metadata, dependencies, and tool configuration
- `dbstep/` — Main package source code
  - `Dbstep.py` — Core `dbstep` class and CLI entry point (`main()`)
  - `calculator.py` — Math/geometry routines (rotations, angles)
  - `sterics.py` — Steric parameter calculations
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
  - `test_parse_data.py` — Input parsing tests
  - `cube_files/` — Test cube file fixtures
- `.github/workflows/ci.yml` — GitHub Actions CI (test, lint, publish)

## Development Commands

### Install (using uv)
```
uv sync
```

### Install with dev tools
```
uv sync --extra dev
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
