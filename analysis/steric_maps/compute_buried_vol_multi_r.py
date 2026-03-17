"""
Compute buried volume (%V_Bur) at multiple sphere radii in a single pass.

Same protocol as compute_buried_vol.py but evaluates all requested radii for
each conformer, avoiding repeated SDF reads.

Usage:
  python compute_buried_vol_multi_r.py --sdf aimnet2_out.sdf --csv capped.csv \
      --radii 2.5 3.5 4.5 5.5 6.5 7.5 8.5

Output:
  <stem>_multi_r_buried_vol.csv         : per-conformer results (all radii)
  <stem>_multi_r_buried_vol_summary.csv : Boltzmann-weighted summaries (all radii)
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
	parser.add_argument("--radii", type=float, nargs="+", required=True,
						help="Sphere radii for buried volume (Å)")
	parser.add_argument("--output", default=None,
						help="Output CSV path (default: <stem>_multi_r_buried_vol.csv)")
	return parser.parse_args()


def main():
	args = parse_args()
	radii = sorted(args.radii)

	# Load attachment atom indices and cap group atoms from the capped CSV
	cap_df = pd.read_csv(args.csv)
	attach_map = dict(zip(cap_df["name"], cap_df["attach_atom_idx"]))
	cap_atoms_map = dict(zip(
		cap_df["name"],
		cap_df["cap_atoms"].apply(lambda s: [int(x) for x in str(s).split(",")] if pd.notna(s) else [])
	))
	print(f"Loaded {len(attach_map)} fragment definitions from {args.csv}")
	print(f"Radii: {radii} Å")

	# Load conformers from SDF
	suppl = Chem.SDMolSupplier(args.sdf, removeHs=False)
	mols = [m for m in suppl if m is not None]
	print(f"Read {len(mols)} conformers from {args.sdf}")

	records = []
	n_missing = 0
	for i, mol in enumerate(mols):
		name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
		e_rel = float(mol.GetProp("E_rel(kcal/mol)")) if mol.HasProp("E_rel(kcal/mol)") else 0.0
		e_tot = float(mol.GetProp("E_tot")) if mol.HasProp("E_tot") else None

		if name not in attach_map:
			n_missing += 1
			continue

		attach_idx = attach_map[name]
		cap_heavy = cap_atoms_map[name]

		# Build exclude list: cap heavy atoms + their hydrogens (1-indexed)
		exclude_set = set(cap_heavy)
		for idx in cap_heavy:
			for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
				if nbr.GetAtomicNum() == 1:
					exclude_set.add(nbr.GetIdx())
		exclude_str = ",".join(str(j + 1) for j in sorted(exclude_set))

		row = {"name": name, "E_rel(kcal/mol)": e_rel, "E_tot": e_tot}
		for r in radii:
			result = dbstep(mol, atom1=attach_idx + 1, volume=True, r=r,
							exclude=exclude_str, verbose=False, quiet=True)
			row[f"pct_V_bur_r{r}"] = result.bur_vol
		records.append(row)

		if (i + 1) % 1000 == 0:
			print(f"  Processed {i + 1}/{len(mols)} conformers...")

	df = pd.DataFrame(records)
	print(f"Computed buried volume for {len(df)} conformers")
	if n_missing:
		print(f"  Skipped {n_missing} conformers (name not in CSV)")

	# Boltzmann-weighted summary per fragment
	RT = 0.5922  # kcal/mol at 298.15 K
	summary = []
	for name, group in df.groupby("name"):
		e_rel = group["E_rel(kcal/mol)"].values
		weights = np.exp(-e_rel / RT)
		weights /= weights.sum()

		row = {"name": name, "n_conformers": len(group)}
		for r in radii:
			col = f"pct_V_bur_r{r}"
			vbur = group[col].values
			row[f"boltz_r{r}"] = round(np.sum(weights * vbur), 2)
			row[f"min_E_r{r}"] = round(vbur[np.argmin(e_rel)], 2)
			row[f"min_r{r}"] = round(vbur.min(), 2)
			row[f"max_r{r}"] = round(vbur.max(), 2)
		summary.append(row)

	summary_df = pd.DataFrame(summary)

	# Output paths
	stem = os.path.splitext(os.path.basename(args.sdf))[0]
	out_dir = os.path.dirname(os.path.abspath(args.sdf))

	out_conf = args.output or os.path.join(out_dir, f"{stem}_multi_r_buried_vol.csv")
	df.to_csv(out_conf, index=False)
	print(f"\nPer-conformer results -> {out_conf}")

	out_summary = os.path.join(out_dir, f"{stem}_multi_r_buried_vol_summary.csv")
	summary_df.to_csv(out_summary, index=False)
	print(f"Boltzmann-weighted summary -> {out_summary}")
	print("\nDone.")


if __name__ == "__main__":
	main()
