# Changelog

All notable changes to DBSTEP. Versions follow [semantic versioning](https://semver.org/).

## Unreleased

### Added
- `--decompose`: per-residue contributions to %V_bur (overlaps shared equally, contributions sum to the total); `contributions` on the Python object and `<csv>_contributions.csv` with `--csv`.
- `examples/proteins_and_conformers.ipynb`: worked notebook for the protein, trajectory and conformer-ensemble workflows, executed by the test suite.

## 2.0.0 — 2026-09

### Breaking changes
- Python 3.10 or newer is required (3.9 reached end of life in October 2025); CI covers 3.10 to 3.14.
- The command line is parsed with argparse: `--help` is grouped, `--opt=value` works, abbreviations are not accepted, and unknown options or invalid choices exit with status 2. Input files are positional arguments; a missing file is an error. Option values are no longer globbed as input files (previously `--csv out.csv` re-read an existing `out.csv` as a structure).
- With a cutoff active (`--cutoff`, and always in residue mode) the molecular volume column reports only the kept atoms and is labelled `MolVol_cut`.
- `dbstep.Dbstep.from_rdkit(mol, ...)` is the public entry point for RDKit molecules.

### Added
- **Proteins.** Native `.pdb`/`.ent` parsing with residue metadata. `--residue A:45` (insertion codes, chain-less numbers, comma-separated lists, `all`) selects a residue; `--atom CA` and atom names for `--atom2`/`--atom3` pick the reference atoms. Environment filters `--nowater`, `--nohet`, `--chain`, `--exclude-self` and `--self-only`. Results label the residue; `all_residues()` from Python.
- **Radial crop.** `--cutoff <Å|auto>` ignores atoms too far from atom1 to matter, so the grid scales with the sphere rather than the system; `auto` is exact for %V_bur and is the default in residue mode.
- **Trajectories.** Multi-frame `.xyz`, multi-record `.sdf` and multi-MODEL `.pdb` files run frame by frame; `--frames start:stop:stride` selects frames; `all_frames()` from Python.
- **Conformer ensembles.** `--boltzmann [TAG]` weights the results of a multi-record SDF (AQME, CREST, RDKit) or multi-frame xyz by Boltzmann populations from an energy data field or the xyz comment line, with `--temperature` and `--energy-units`; `dbstep.ensemble` from Python.
- **Result records and CSV.** Every run keeps `results` (one row per radius); `--csv` writes all rows of a command line, with frame, structure, residue and population columns.
- `--exclude` accepts a list from the Python API; SDF titles are kept as structure labels.

### Fixed
- Grid-based Sterimol used a 361 × N projection matrix, which needed gigabytes for fine grids around large systems; the angular sweep now uses the convex hull of the XY projection (exact).
- Grid lattice coordinates are snapped to 8 decimals so results no longer depend on the box extent through rounding noise.
- SDF records with 100 or more atoms and bonds, orbital cube files with a negative atom count, `--pos` combined with `--scan`, and option leakage between files of a multi-file run.

### Internal
- Release workflow publishes to PyPI from GitHub Releases via trusted publishing; dependabot config; analysis-script dependencies moved to an opt-in group; large unused test fixtures removed.

## 1.2.0 — 2026-09-25

First release since 1.1.0: multi-structure `.xyz`/`.sdf` input, the `dbstep` console command, `--radii charry-tkatchenko`, `--sambvca`, `--measure`, `--dp`, `--quiet`, `--atom3`, `--pos`, `--norot`, `--scalevdw`, `--gridsize`, occupancy tensors (`--tensor`/`--save`), tabular output; fixes to cube-file grid ordering, 2D McGowan volumes, grid Sterimol L and cube parsing; hydrogen Bondi radius 1.09 Å, default grid 0.05 Å, metals included by default; renamed `--volume`→`--vbur`, `--addmetals`→`--nometals`, `--commandline`→`--pymol`; Python 3.9+, pyproject/uv packaging, GitHub Actions CI.
