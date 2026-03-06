# Radii Benchmarking Pipeline

Comparison of VDW radii sets and scaling strategies for reproducing isodensity molecular volumes and buried volumes.

## Overview

Three benchmarking analyses are provided:

1. **Radii comparison** — qualitative comparison of Bondi vs Charry-Tkatchenko radii
2. **Parity plots** — quantitative accuracy of three radii conditions vs isodensity reference (486 molecules, 1044 buried volume samples)
3. **Optimal scaling** — systematic sweep of Bondi scaling factors to minimise deviation from isodensity reference

---

## 1. Radii Comparison

**Script:** `compare_radii.py`
**Output:** `radii_comparison.png`

Generates a 3-panel figure comparing the two VDW radii sets available in DBSTEP:

- **(a)** Scatter plot of Bondi vs Charry-Tkatchenko radii for all 54 shared elements. The Charry-Tkatchenko radii (free-atom, derived from dipole polarizability) are systematically larger than Bondi radii for most elements, particularly for transition metals (e.g. Cu, Zn, Pd, Pt).
- **(b)** Molecular volumes computed with both radii sets for all 22 molecules in `dbstep/data/`.
- **(c)** Percent buried volumes (%V_Bur) at R = 3.5 Å for the same molecules.

**References:**
- Bondi: *J. Phys. Chem.* **1964**, *68*, 441; Mantina et al. *J. Phys. Chem. A* **2009**, *113*, 5806
- Charry-Tkatchenko: *J. Chem. Theory Comput.* **2024**, *20*, 7844-7855 (R_vdW^free[alpha] from Table S1, SI)

```bash
uv run python analysis/radii/compare_radii.py
```

![Radii Comparison](radii_comparison.png)

---

## 2. Parity Plots vs Isodensity Reference

**Script:** `parity_plots.py`
**Output:** `parity_plots.png`

A 2×3 figure comparing three radii conditions against the isodensity reference (0.0016 e/Bohr³ electron density surface):

| Condition | Radii | Scaling | Hydrogens |
|-----------|-------|---------|-----------|
| Bondi | Bondi | 1.0× | included |
| Charry-Tkatchenko | Charry-Tkatchenko | 1.0× | included |
| SambVca | Bondi | 1.17× | excluded |

**Top row — molecular volumes** (486 molecules from ZINC dataset):
- Parity plots of VDW vs isodensity molecular volume for each condition
- Reference: `isodensity_volumes.txt`, `bondi_volumes.txt`, `charry_volumes.txt`, `sambvca_volumes.txt`

**Bottom row — buried volumes** (1044 stratified atom-center samples):
- Parity plots of %V_Bur vs isodensity %V_Bur, coloured by center-atom element
- Reference: `isodensity_sampled.csv`, `bondi_sampled.csv`, `charry_sampled.csv`, `sambvca_sampled.csv`

### Sampling strategy

Buried volume samples were generated using a stratified element-based approach to ensure diverse element coverage:

| Element | Quota | Notes |
|---------|-------|-------|
| C | 250 | |
| H | 250 | |
| N | 150 | |
| O | 150 | |
| S | 100 | |
| F | 94 | capped (only 94 molecules contain F) |
| Cl | 50 | |
| **Total** | **1044** | |

At most one atom of each element is sampled per molecule. See `sample_atoms.py` for details; `sample_atoms.csv` contains the full list of (mol_file, atom1_idx, element) triples.

### Running the calculations

Buried volume calculations for all three conditions are run with:

```bash
uv run python analysis/radii/run_sampled_vbur.py
```

This produces `bondi_sampled.csv`, `charry_sampled.csv`, and `sambvca_sampled.csv`.

The isodensity reference (`isodensity_sampled.csv`) requires cube files and should be run on a server:

```bash
# Copy sample_atoms.csv and run_sampled_vbur.py to server, then:
python analysis/radii/run_sampled_vbur.py --isodensity
```

### Generating the figure

```bash
uv run python analysis/radii/parity_plots.py
```

![Parity Plots](parity_plots.png)

---

## 3. Optimal Bondi Scaling Factor

**Script:** `optimal_scaling.py`
**Output:** `optimal_scaling.png`

Determines the optimal Bondi radius scaling factor for reproducing isodensity molecular volumes and buried volumes by running actual DBSTEP calculations across a sweep of scaling factors.

**Sweep:** s = 1.00 to 1.20 in steps of 0.01 (21 values total)
- s = 1.00 uses `bondi_sampled.csv`
- s = 1.01–1.20 uses `bondi_x{s:.2f}_sampled.csv`

**Reference data:**
- Molecular volumes: `isodensity_volumes.txt` (486 molecules, mol_id from filename)
- Buried volumes: `isodensity_sampled.csv` (1044 samples)

**Output panels:**
- **(a)** RMSE vs scaling factor for both molecular volume (Å³, left y-axis) and %V_Bur (%, right y-axis), with vertical lines marking the optimal s for each metric
- **(b)** Molecular volume parity plot at the optimal scaling factor
- **(c)** %V_Bur parity plot at the optimal scaling factor, coloured by center-atom element

### Running the sweep

First generate the sweep CSV files (skips any that already exist):

```bash
uv run python analysis/radii/run_sampled_vbur.py --sweep
```

Then generate the figure:

```bash
uv run python analysis/radii/optimal_scaling.py
```

![Optimal Scaling](optimal_scaling.png)

---

## Full Pipeline

To reproduce all results from scratch (excluding the isodensity reference which requires cube files):

```bash
# 1. Generate atom sample list
uv run python analysis/radii/sample_atoms.py

# 2. Run VDW calculations (Bondi, Charry-Tkatchenko, SambVca)
uv run python analysis/radii/run_sampled_vbur.py

# 3. Run scaling sweep (Bondi x1.01 to x1.20)
uv run python analysis/radii/run_sampled_vbur.py --sweep

# 4. Generate figures
uv run python analysis/radii/compare_radii.py
uv run python analysis/radii/parity_plots.py
uv run python analysis/radii/optimal_scaling.py
```

Steps 2–4 require `isodensity_sampled.csv` and the `*_volumes.txt` files from the isodensity reference calculations.
