# PDB test fixtures

- `1a8o.pdb` — HIV-1 capsid C-terminal domain (PDB ID 1A8O), 70 residues, 644 atoms including
  four selenomethionines (HETATM MSE with Se) and 88 waters, no hydrogens. Copied from the
  Biopython test suite; original data from the RCSB Protein Data Bank (public domain, CC0).
- `ala5.pdb` — penta-alanine with all hydrogens (53 atoms), two waters and a sodium ion, generated
  by `make_ala5.py` with RDKit (embedding seed 7, MMFF-optimised). Small enough that a grid over the
  whole molecule is feasible, which the crop-invariance tests rely on.
