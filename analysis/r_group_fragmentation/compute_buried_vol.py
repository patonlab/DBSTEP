"""
Compute buried volume (%V_Bur) at the cap attachment atom for capped fragment conformers.

Reads an SDF file from Auto3D (with multiple conformers per fragment) and a
capped fragment CSV (from cap_fragments.py) that contains the attach_atom_idx
and cap_atoms columns. Uses DBSTEP to compute the buried volume at the
attachment point, excluding the cap group atoms (heavy atoms + their hydrogens)
from the steric measurement.

The attach_atom_idx and cap_atoms from the CSV are 0-based heavy-atom indices
in the canonical SMILES, which are stable when hydrogens are added (H atoms are
appended after heavy atoms by RDKit). At runtime, hydrogens bonded to cap heavy
atoms are also identified and excluded.

Usage:
  python compute_buried_vol.py --sdf aimnet2_out.sdf --csv capped_phenyl.csv --radius 3.5

Output:
  <stem>_buried_vol.csv         : per-conformer buried volumes
  <stem>_buried_vol_summary.csv : Boltzmann-weighted averages per fragment
"""

import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from dbstep.Dbstep import dbstep
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sdf", required=True, help="Input SDF from Auto3D")
    parser.add_argument("--csv", required=True,
                        help="Capped fragment CSV with attach_atom_idx column")
    parser.add_argument("--radius", type=float, default=3.5,
                        help="Sphere radius for buried volume (default: 3.5 Å)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: <stem>_buried_vol.csv)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load attachment atom indices and cap group atoms from the capped CSV
    cap_df = pd.read_csv(args.csv)
    attach_map = dict(zip(cap_df["name"], cap_df["attach_atom_idx"]))
    cap_atoms_map = dict(zip(
        cap_df["name"],
        cap_df["cap_atoms"].apply(lambda s: [int(x) for x in str(s).split(",")] if pd.notna(s) else [])
    ))
    print(f"Loaded {len(attach_map)} fragment definitions from {args.csv}")

    # Load conformers from SDF
    suppl = Chem.SDMolSupplier(args.sdf, removeHs=False)
    mols = [m for m in suppl if m is not None]
    print(f"Read {len(mols)} conformers from {args.sdf}")

    records = []
    n_missing = 0
    for mol in mols:
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
        e_rel = float(mol.GetProp("E_rel(kcal/mol)")) if mol.HasProp("E_rel(kcal/mol)") else 0.0
        e_tot = float(mol.GetProp("E_tot")) if mol.HasProp("E_tot") else None

        if name not in attach_map:
            print(f"  WARNING: {name} not found in CSV, skipping")
            n_missing += 1
            continue

        attach_idx = attach_map[name]
        cap_heavy = cap_atoms_map[name]

        # Build exclude list: cap heavy atoms + their hydrogens (1-indexed for DBSTEP)
        exclude_set = set(cap_heavy)
        for idx in cap_heavy:
            for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
                if nbr.GetAtomicNum() == 1:  # hydrogen
                    exclude_set.add(nbr.GetIdx())
        exclude_str = ",".join(str(i + 1) for i in sorted(exclude_set))  # 1-indexed

        # DBSTEP uses 1-indexed atoms
        result = dbstep(mol, atom1=attach_idx + 1, volume=True, r=args.radius,
                        exclude=exclude_str, verbose=False)

        records.append({
            "name": name,
            "E_rel(kcal/mol)": e_rel,
            "E_tot": e_tot,
            "attach_atom_idx": attach_idx,
            "pct_V_bur": result.bur_vol,
        })

    df = pd.DataFrame(records)
    print(f"Computed buried volume for {len(df)} conformers")
    if n_missing:
        print(f"  Skipped {n_missing} conformers (name not in CSV)")

    # Boltzmann-weighted average per fragment
    RT = 0.5922  # kcal/mol at 298.15 K
    summary = []
    for name, group in df.groupby("name"):
        e_rel = group["E_rel(kcal/mol)"].values
        vbur = group["pct_V_bur"].values

        # Boltzmann weights
        weights = np.exp(-e_rel / RT)
        weights /= weights.sum()

        vbur_boltz = np.sum(weights * vbur)
        vbur_min_e = vbur[np.argmin(e_rel)]

        summary.append({
            "name": name,
            "n_conformers": len(group),
            "pct_V_bur_boltz": round(vbur_boltz, 2),
            "pct_V_bur_min_E": round(vbur_min_e, 2),
            "pct_V_bur_min": round(vbur.min(), 2),
            "pct_V_bur_max": round(vbur.max(), 2),
        })

    summary_df = pd.DataFrame(summary)

    # Output paths
    stem = os.path.splitext(os.path.basename(args.sdf))[0]
    out_dir = os.path.dirname(os.path.abspath(args.sdf))

    # Per-conformer results
    out_conf = args.output or os.path.join(out_dir, f"{stem}_buried_vol.csv")
    df.to_csv(out_conf, index=False)
    print(f"\nPer-conformer results -> {out_conf}")

    # Summary per fragment
    out_summary = os.path.join(out_dir, f"{stem}_buried_vol_summary.csv")
    summary_df.to_csv(out_summary, index=False)
    print(f"Boltzmann-weighted summary -> {out_summary}")

    print(f"\n{summary_df.to_string(index=False)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
