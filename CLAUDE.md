# CLAUDE.md

## Project Overview

DBSTEP (DFT-Based Steric Parameters) is a Python package for computing steric parameters from chemical structures. It calculates Sterimol parameters (L, Bmin, Bmax), percent buried volume, Sterimol2Vec, and Vol2Vec parameters from molecular structure files and quantum chemistry output files.

## Repository Structure

- `dbstep/` — Main package source code
  - `Dbstep.py` — Core `dbstep` class and CLI entry point (`main()`)
  - `calculator.py` — Math/geometry routines (rotations, angles)
  - `sterics.py` — Steric parameter calculations
  - `parse_data.py` — Input file parsing (xyz, cube, cclib-supported formats)
  - `constants.py` — Chemical constants (periodic table, Bondi radii, metals)
  - `graph.py` — 2D graph-based steric contribution calculations
  - `writer.py` — Output formatting and file writing
  - `__main__.py` — Module entry point for `python -m dbstep`
  - `examples/` — Example molecular structure files (xyz format)
- `tests/` — Pytest test suite
  - `test_dbstep.py` — Sterimol parameter validation against Verloop's reference values
  - `test_calculator.py` — Unit tests for rotation/geometry math
  - `test_parse_data.py` — Input parsing tests
  - `cube_files/` — Test cube file fixtures
- `setup.py` — Package configuration (setuptools)

## Development Commands

### Install dependencies
```
pip install -r dbstep/requirements.txt
```

### Install the package (editable/development)
```
pip install -e .
```

### Run tests
```
pytest -v
```

### Run the tool
```
python -m dbstep <file> --sterimol --atom1 <idx> --atom2 <idx>
```

## Key Dependencies

- numpy, numba, scipy, cclib
- Optional: RDKit, pandas (for 2D graph-based steric features)
- Test framework: pytest

## Code Conventions

- Uses tabs for indentation throughout
- Python 3.6+ required
- Main class is lowercase `dbstep` in `dbstep/Dbstep.py`
- Atom indexing is 1-based (matching chemical structure file conventions)
- Tests compare computed values against Verloop's reference Sterimol parameters with a tolerance of 0.01
