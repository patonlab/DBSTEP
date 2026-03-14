"""
Terminal BRICS fragment extraction from a SMILES CSV, with supplemental methyl rule.

Fragments each molecule using BRICS (Degen et al., ChemMedChem 2008, 3, 1503-1507),
then retains only terminal fragments — those with exactly one attachment point (*).
Linker fragments (2+ attachment points) are discarded.

A supplemental SMARTS rule is applied after BRICS to catch terminal methyl groups
on ring atoms that BRICS misses (e.g. N-methyls on fused heterocycles like caffeine,
C-methyls on steroid rings like testosterone). The rule cuts any single bond between
a ring atom and a -CH3 group not already cut by BRICS.

Counts reflect total occurrences across the dataset (including duplicates within a
molecule). molecule_count tracks how many distinct molecules contain each fragment.

References:
  Degen, J. et al. ChemMedChem 2008, 3, 1503-1507.
  https://doi.org/10.1002/cmdc.200800178

Usage:
  python brics_fragments_from_csv.py --input molecules.csv --smiles_col smiles --name_col name

Outputs:
  <stem>_brics_results.csv      : per-molecule fragment list
  <stem>_brics_distribution.csv : ranked frequency table (total_count + molecule_count)
  <stem>_brics_counts.csv       : wide-format count matrix (one col per unique fragment)
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import BRICS, rdmolops
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


# ── SMARTS for supplemental fragmentation rules ──────────────────────────────
# Rule A: any ring atom bonded to a terminal methyl (catches C-Me, N-Me on rings
#         that BRICS misses, e.g. testosterone C-Me, caffeine N-Me on aliphatic N)
_METHYL_PAT = Chem.MolFromSmarts("[cR,CR,nR,NR,sR,SR,oR,OR]-[CH3]")
# Rule B: aromatic ring N bonded to any non-ring substituent (catches all N-substitution
#         on heteroaromatic rings like purines, pyrimidines that BRICS misses)
_AROM_N_PAT = Chem.MolFromSmarts("[nR]-[!R]")


# ── core extraction ──────────────────────────────────────────────────────────

def canonical_frag(smi):
    """Strip isotope labels from * attachment points and return canonical SMILES."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetIsotope(0)
    return Chem.MolToSmiles(mol, canonical=True)


def get_terminal_brics_fragments(mol):
    """
    Return a list of canonical SMILES for terminal fragments (exactly one *),
    including duplicates within the molecule.

    Bond-cutting strategy:
      1. BRICS bonds (FindBRICSBonds) — the 16 retrosynthetically motivated rules
      2. Supplemental methyl bonds — any ring-atom → CH3 bond not already cut by BRICS

    All bonds are cut simultaneously with FragmentOnBonds; fragments with exactly
    one attachment point are retained.

    Returns [] if no bonds are found; None on failure.
    """
    if mol is None:
        return None
    try:
        # Step 1: BRICS bonds — restrict to single bonds only to avoid
        # cutting conjugated C=C bonds (e.g. in alpha,beta-unsaturated systems)
        # which produce *=C fragments
        brics_bonds = list(BRICS.FindBRICSBonds(mol))
        brics_indices = {
            mol.GetBondBetweenAtoms(a1, a2).GetIdx() for (a1, a2), _ in brics_bonds
            if mol.GetBondBetweenAtoms(a1, a2).GetBondType() == Chem.BondType.SINGLE
        }

        # Step 2: supplemental bonds not already covered by BRICS (single bonds only)
        extra_indices = set()
        for pat in (_METHYL_PAT, _AROM_N_PAT):
            for match in mol.GetSubstructMatches(pat):
                bond = mol.GetBondBetweenAtoms(match[0], match[1])
                if (bond
                        and bond.GetBondType() == Chem.BondType.SINGLE
                        and bond.GetIdx() not in brics_indices):
                    extra_indices.add(bond.GetIdx())

        all_indices = sorted(brics_indices | extra_indices)
        if not all_indices:
            return []

        # Cut all bonds simultaneously
        frag_mol = rdmolops.FragmentOnBonds(mol, all_indices, addDummies=True)
        frags = rdmolops.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)

        terminal = []
        for frag in frags:
            edit = Chem.RWMol(frag)
            for atom in edit.GetAtoms():
                atom.ClearProp("molAtomMapNumber")
                atom.SetIsotope(0)
            try:
                Chem.SanitizeMol(edit)
                smi = Chem.MolToSmiles(edit.GetMol(), canonical=True)
            except Exception:
                continue
            fmol = Chem.MolFromSmiles(smi)
            if fmol is None:
                continue
            n_attach = sum(1 for a in fmol.GetAtoms() if a.GetAtomicNum() == 0)
            if n_attach == 1 and smi != "*":
                terminal.append(smi)

        return sorted(terminal)

    except Exception:
        return None


# ── CSV pipeline ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",      required=True,    help="Input CSV file")
    parser.add_argument("--smiles_col", default="smiles", help="SMILES column name (default: smiles)")
    parser.add_argument("--name_col",   default=None,     help="Molecule name/ID column (optional)")
    parser.add_argument("--sep",        default=",",      help="CSV separator (default: ',')")
    parser.add_argument("--output_dir", default=None,     help="Output directory (default: same as input)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Reading: {args.input}")
    df = pd.read_csv(args.input, sep=args.sep, low_memory=False)

    if args.smiles_col not in df.columns:
        sys.exit(f"ERROR: Column '{args.smiles_col}' not found. Available: {list(df.columns)}")

    n_before = len(df)
    df = df[df[args.smiles_col].notna()].reset_index(drop=True)
    print(f"  {n_before} rows loaded, {len(df)} have non-null SMILES.")

    print("Running BRICS fragmentation (+ supplemental methyl rule)...")

    def safe_extract(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            return get_terminal_brics_fragments(mol)
        except Exception:
            return None

    tqdm.pandas(desc="Fragmenting")
    df["brics_fragments"] = df[args.smiles_col].progress_apply(safe_extract)
    df["n_brics_fragments"] = df["brics_fragments"].apply(
        lambda x: len(x) if isinstance(x, list) else None
    )

    n_failed = df["brics_fragments"].isna().sum()
    n_empty  = (df["n_brics_fragments"] == 0).sum()
    print(f"  Succeeded             : {len(df) - n_failed}")
    print(f"  Failed                : {n_failed}")
    print(f"  No terminal fragments : {n_empty}")

    # --- output paths ---
    stem    = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(out_dir, exist_ok=True)

    def outpath(suffix):
        return os.path.join(out_dir, f"{stem}_{suffix}.csv")

    # ── Output 1: per-molecule results ──────────────────────────────────────
    results_df = df.drop(columns=["brics_fragments"]).copy()
    results_df["brics_fragments_str"] = df["brics_fragments"].apply(
        lambda x: " | ".join(x) if isinstance(x, list) else ""
    )
    p1 = outpath("brics_results")
    results_df.to_csv(p1, index=False)
    print(f"\n[1/3] Per-molecule results      -> {p1}")

    # ── Output 2: distribution ───────────────────────────────────────────────
    valid = df[df["brics_fragments"].apply(lambda x: isinstance(x, list) and len(x) > 0)]

    total_counts = Counter()
    mol_counts   = Counter()
    for frags in valid["brics_fragments"]:
        total_counts.update(frags)
        mol_counts.update(set(frags))

    frag_dist = pd.DataFrame({
        "fragment_smiles": list(total_counts.keys()),
        "total_count":     list(total_counts.values()),
        "molecule_count":  [mol_counts[f] for f in total_counts.keys()],
    }).sort_values("total_count", ascending=False).reset_index(drop=True)

    frag_dist["rank"]              = frag_dist.index + 1
    frag_dist["total_fraction"]    = frag_dist["total_count"]    / len(df)
    frag_dist["molecule_fraction"] = frag_dist["molecule_count"] / len(df)

    p2 = outpath("brics_distribution")
    frag_dist.to_csv(p2, index=False)
    print(f"[2/3] Fragment frequency table  -> {p2}")
    print(f"      Unique terminal fragments : {len(frag_dist)}")
    print("\n  Top 15 most common fragments (by total count):")
    print(frag_dist[["rank","fragment_smiles","total_count","molecule_count"]].head(15).to_string(index=False))

    # ── Output 3: per-molecule count matrix ─────────────────────────────────
    id_col = args.name_col if args.name_col and args.name_col in df.columns else args.smiles_col

    # Write the count matrix row-by-row in chunks to avoid materialising the
    # full dense matrix in memory (e.g. 250k molecules × 16k fragments = 4B cells).
    # Each chunk builds a small dense block, writes it, then is discarded.
    ordered_cols = frag_dist["fragment_smiles"].tolist()
    col_index = {frag: i for i, frag in enumerate(ordered_cols)}  # O(1) lookup
    n_mols = len(df)
    CHUNK = 5_000  # rows per chunk — tune down if still tight on RAM

    p3 = outpath("brics_counts")
    with open(p3, "w") as fout:
        # header
        fout.write(",".join([id_col] + ordered_cols) + "\n")

        with tqdm(total=n_mols, desc="[3/3] Count matrix", unit="mol",
                  bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                  ) as pbar:
            for start in range(0, n_mols, CHUNK):
                chunk = df.iloc[start : start + CHUNK]
                # Build a zero-filled numpy array for this chunk only
                block = np.zeros((len(chunk), len(ordered_cols)), dtype=np.int32)
                for row_i, frags in enumerate(chunk["brics_fragments"]):
                    if isinstance(frags, list):
                        for frag, cnt in Counter(frags).items():
                            col_i = col_index.get(frag)
                            if col_i is not None:
                                block[row_i, col_i] = cnt
                # Write rows
                ids = chunk[id_col].astype(str).values
                for row_i in range(len(chunk)):
                    fout.write(ids[row_i] + "," +
                               ",".join(block[row_i].astype(str).tolist()) + "\n")
                pbar.update(len(chunk))

    print(f"\n[3/3] Fragment count matrix     -> {p3}")
    print(f"      Shape: {n_mols} molecules x {len(ordered_cols)} unique fragments")

    print("\nDone.")


if __name__ == "__main__":
    main()
