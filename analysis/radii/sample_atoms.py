"""
Sample atom centers for buried volume benchmarking.

Generates a CSV of (mol_file, atom1_idx, element) pairs covering a
stratified selection of element types. For each element, at most one
atom per molecule is selected (picked at random from that element's
atoms in the molecule). Sampling is seeded for reproducibility.

Output: sample_atoms.csv
"""

import os
import random
import csv
from collections import defaultdict

SEED = 42
XYZ_DIR = os.path.join(os.path.dirname(__file__), "xyz_files")
OUTPUT = os.path.join(os.path.dirname(__file__), "sample_atoms.csv")

QUOTAS = {
    "S":   100,
    "F":   100,
    "Cl":   50,
    "N":   150,
    "O":   150,
    "C":   250,
    "H":   250,
}


def parse_xyz_dir(xyz_dir):
    """Return dict: mol_file -> list of (atom_idx_1based, element)."""
    mol_atoms = {}
    for fname in sorted(os.listdir(xyz_dir)):
        if not fname.endswith(".xyz"):
            continue
        path = os.path.join(xyz_dir, fname)
        with open(path) as f:
            lines = f.readlines()
        n = int(lines[0])
        atoms = []
        for i, line in enumerate(lines[2:2 + n]):
            el = line.split()[0]
            atoms.append((i + 1, el))  # 1-based index
        mol_atoms[fname] = atoms
    return mol_atoms


def build_element_pool(mol_atoms):
    """Return dict: element -> list of (mol_file, atom_idx) pairs."""
    pool = defaultdict(list)
    for mol_file, atoms in mol_atoms.items():
        for atom_idx, el in atoms:
            pool[el].append((mol_file, atom_idx))
    return pool


def sample_stratified(mol_atoms, quotas, seed=SEED):
    """
    For each element, sample up to `quota` atoms with at most one atom
    per molecule. Returns list of (mol_file, atom_idx, element) tuples.
    """
    rng = random.Random(seed)

    # Group by element then by molecule: element -> mol -> [atom_idx, ...]
    by_el_mol = defaultdict(lambda: defaultdict(list))
    for mol_file, atoms in mol_atoms.items():
        for atom_idx, el in atoms:
            by_el_mol[el][mol_file].append(atom_idx)

    samples = []
    for el, quota in quotas.items():
        # List of molecules that have this element
        mols = list(by_el_mol[el].keys())
        rng.shuffle(mols)

        # Cap quota at what's available
        n = min(quota, len(mols))
        if n < quota:
            print(f"  Warning: {el} only has {len(mols)} molecules, using all {n}")

        selected_mols = mols[:n]
        for mol_file in selected_mols:
            atom_idx = rng.choice(by_el_mol[el][mol_file])
            samples.append((mol_file, atom_idx, el))

    return samples


def main():
    print(f"Parsing xyz files in {XYZ_DIR} ...")
    mol_atoms = parse_xyz_dir(XYZ_DIR)
    print(f"  Found {len(mol_atoms)} molecules")

    print("\nElement availability:")
    by_el_mol = defaultdict(set)
    for mol_file, atoms in mol_atoms.items():
        for _, el in atoms:
            by_el_mol[el].add(mol_file)
    for el, quota in QUOTAS.items():
        avail = len(by_el_mol[el])
        print(f"  {el:3s}: quota={quota:4d}, available in {avail:3d} molecules"
              + (" *** UNDER QUOTA" if avail < quota else ""))

    print(f"\nSampling (seed={SEED}) ...")
    samples = sample_stratified(mol_atoms, QUOTAS, seed=SEED)

    # Sort for readability
    samples.sort(key=lambda x: (x[2], x[0], x[1]))

    print(f"\nSample counts by element:")
    from collections import Counter
    for el, cnt in sorted(Counter(s[2] for s in samples).items()):
        print(f"  {el:3s}: {cnt}")
    print(f"  Total: {len(samples)}")

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mol_file", "atom1_idx", "element"])
        writer.writerows(samples)

    print(f"\nWritten to {OUTPUT}")


if __name__ == "__main__":
    main()
