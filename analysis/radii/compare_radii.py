"""
Compare Bondi and Charry-Tkatchenko VDW radii sets.

Generates a 3-panel figure:
  (a) Bondi vs Charry-Tkatchenko radii for all shared elements
  (b) Molecular volumes with both radii sets for molecules in data/
  (c) Buried volumes with both radii sets for molecules in data/

Usage:
    python analysis/compare_radii.py
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

# ensure the package is importable when running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dbstep.Dbstep import dbstep
from dbstep.constants import bondi, charry_tkatchenko, periodic_table


def get_shared_elements():
	"""Return elements present in both radii dicts, ordered by atomic number."""
	return [el for el in periodic_table
			if el in bondi and el in charry_tkatchenko and el not in ("Bq", "")]


def compute_volumes(data_dir):
	"""Compute Mol_Vol and %V_Bur with both radii sets for all xyz files in data_dir."""
	files = sorted(f for f in os.listdir(data_dir) if f.endswith(".xyz"))
	names, bondi_mvol, ct_mvol, bondi_vbur, ct_vbur = [], [], [], [], []

	for f in files:
		path = os.path.join(data_dir, f)
		label = os.path.splitext(f)[0]

		r_bondi = dbstep(path, volume=True, radii="bondi", quiet=True)
		r_ct = dbstep(path, volume=True, radii="charry-tkatchenko", quiet=True)

		names.append(label)
		bondi_mvol.append(r_bondi.occ_vol)
		ct_mvol.append(r_ct.occ_vol)
		bondi_vbur.append(r_bondi.bur_vol)
		ct_vbur.append(r_ct.bur_vol)

	return names, bondi_mvol, ct_mvol, bondi_vbur, ct_vbur


def main():
	data_dir = os.path.join(os.path.dirname(__file__), "..", "dbstep", "data")

	# --- Panel (a): radii comparison ---
	elements = get_shared_elements()
	b_radii = [bondi[el] for el in elements]
	ct_radii = [charry_tkatchenko[el] for el in elements]

	# --- Panels (b) & (c): volume comparison ---
	print("Computing volumes for molecules in data/ ...")
	names, bondi_mvol, ct_mvol, bondi_vbur, ct_vbur = compute_volumes(data_dir)

	# --- Plot ---
	fig, axes = plt.subplots(1, 3, figsize=(15, 5))

	# Panel (a): radii scatter
	ax = axes[0]
	lim = [0.8, max(max(b_radii), max(ct_radii)) + 0.2]
	ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
	ax.scatter(b_radii, ct_radii, s=30, c="steelblue", edgecolors="k", linewidths=0.5, zorder=3)
	# label notable outliers
	for i, el in enumerate(elements):
		if abs(b_radii[i] - ct_radii[i]) > 0.4:
			ax.annotate(el, (b_radii[i], ct_radii[i]), fontsize=7, ha="left",
						xytext=(4, 4), textcoords="offset points")
	ax.set_xlabel("Bondi radius (Å)")
	ax.set_ylabel("Charry-Tkatchenko radius (Å)")
	ax.set_title("(a) VDW Radii Comparison")
	ax.set_xlim(lim)
	ax.set_ylim(lim)
	ax.set_aspect("equal")

	# Panel (b): molecular volume parity
	ax = axes[1]
	mvol_lim = [0, max(max(bondi_mvol), max(ct_mvol)) * 1.1]
	ax.plot(mvol_lim, mvol_lim, "k--", lw=0.8, alpha=0.5)
	ax.scatter(bondi_mvol, ct_mvol, s=30, c="steelblue", edgecolors="k", linewidths=0.5, zorder=3)
	for i, name in enumerate(names):
		ax.annotate(name, (bondi_mvol[i], ct_mvol[i]), fontsize=6, ha="left",
					xytext=(4, 4), textcoords="offset points")
	ax.set_xlabel("Mol. Vol. Bondi (ų)")
	ax.set_ylabel("Mol. Vol. Charry-Tkatchenko (ų)")
	r2_mvol = np.corrcoef(bondi_mvol, ct_mvol)[0, 1] ** 2
	ax.set_title(f"(b) Molecular Volume ($R^2$ = {r2_mvol:.4f})")
	ax.set_xlim(mvol_lim)
	ax.set_ylim(mvol_lim)
	ax.set_aspect("equal")

	# Panel (c): buried volume parity
	ax = axes[2]
	vbur_lim = [0, max(max(bondi_vbur), max(ct_vbur)) * 1.1]
	ax.plot(vbur_lim, vbur_lim, "k--", lw=0.8, alpha=0.5)
	ax.scatter(bondi_vbur, ct_vbur, s=30, c="steelblue", edgecolors="k", linewidths=0.5, zorder=3)
	for i, name in enumerate(names):
		ax.annotate(name, (bondi_vbur[i], ct_vbur[i]), fontsize=6, ha="left",
					xytext=(4, 4), textcoords="offset points")
	ax.set_xlabel("%V_Bur Bondi")
	ax.set_ylabel("%V_Bur Charry-Tkatchenko")
	r2_vbur = np.corrcoef(bondi_vbur, ct_vbur)[0, 1] ** 2
	ax.set_title(f"(c) Buried Volume ($R^2$ = {r2_vbur:.4f})")
	ax.set_xlim(vbur_lim)
	ax.set_ylim(vbur_lim)
	ax.set_aspect("equal")

	plt.tight_layout()
	out_path = os.path.join(os.path.dirname(__file__), "radii_comparison.png")
	fig.savefig(out_path, dpi=200, bbox_inches="tight")
	print(f"Figure saved to {out_path}")
	plt.close()


if __name__ == "__main__":
	main()
