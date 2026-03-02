# Analysis

Comparison scripts and figures for DBSTEP steric parameter calculations.

## Radii Comparison: Bondi vs Charry-Tkatchenko

**Script:** `compare_radii.py`

Generates a 3-panel figure comparing the two VDW radii sets available in DBSTEP:

- **(a)** Scatter plot of Bondi vs Charry-Tkatchenko radii for all 54 shared elements. The Charry-Tkatchenko radii (free-atom, derived from dipole polarizability) are systematically larger than Bondi radii for most elements, particularly for transition metals (e.g. Cu, Zn, Pd, Pt).
- **(b)** Molecular volumes computed with both radii sets for all 22 molecules in `dbstep/data/`.
- **(c)** Percent buried volumes (%V_Bur) at R = 3.5 A for the same molecules.

**References:**
- Bondi: *J. Phys. Chem.* **1964**, *68*, 441; Mantina et al. *J. Phys. Chem. A* **2009**, *113*, 5806
- Charry-Tkatchenko: *J. Chem. Theory Comput.* **2024**, *20*, 7844-7855 (R_vdW^free[alpha] from Table S1, SI)

### Running

```bash
uv run python analysis/compare_radii.py
```

Outputs `analysis/radii_comparison.png`.

![Radii Comparison](radii_comparison.png)
