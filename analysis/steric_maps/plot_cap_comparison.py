"""Compare buried volumes for phenyl- vs t-butyl-capped fragments.

Produces a 2x2 parity plot (Boltzmann, min-E, min, max) coloured by
the number of rotatable bonds in each fragment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = __file__.replace("plot_cap_comparison.py", "")

# ── Load & merge ─────────────────────────────────────────────────────────────

ph_cap = pd.read_csv(HERE + "zinc_fragments_phenyl_capped.csv")[["name", "fragment_smiles"]].drop_duplicates("name")
tb_cap = pd.read_csv(HERE + "zinc_fragments_t_butyl_capped.csv")[["name", "fragment_smiles"]].drop_duplicates("name")

ph_vbur = pd.read_csv(HERE + "zinc_fragments_phenyl_capped_buried_vol_summary.csv")
tb_vbur = pd.read_csv(HERE + "zinc_fragments_t_butyl_capped_buried_vol_summary.csv")

ph = ph_cap.merge(ph_vbur, on="name")
tb = tb_cap.merge(tb_vbur, on="name")
df = ph.merge(tb, on="fragment_smiles", suffixes=("_ph", "_tbu"))

# ── Rotatable bonds ──────────────────────────────────────────────────────────

def n_rotbonds(smi):
	mol = Chem.MolFromSmiles(smi.replace("*", "[H]"))
	return Descriptors.NumRotatableBonds(mol) if mol else None

df["n_rotbonds"] = df["fragment_smiles"].apply(n_rotbonds)
print(f"Matched {len(df)} fragments")
print(f"Rotatable bonds: {df['n_rotbonds'].value_counts().sort_index().to_dict()}")

# ── Colour scheme ────────────────────────────────────────────────────────────

ROTBOND_COLORS = {
	0: "#2176AE",
	1: "#57A773",
	2: "#E8A838",
	3: "#E8553A",
	4: "#9B59B6",
	5: "#E91E63",
}

# ── Figure ───────────────────────────────────────────────────────────────────

metrics = ["pct_V_bur_boltz", "pct_V_bur_min_E", "pct_V_bur_min", "pct_V_bur_max"]
titles = ["Boltzmann-weighted", "Lowest-E conformer", "Minimum across conformers", "Maximum across conformers"]

fig, axes = plt.subplots(2, 2, figsize=(11, 11))

for ax, m, title in zip(axes.flat, metrics, titles):
	x = df[f"{m}_ph"].values
	y = df[f"{m}_tbu"].values
	nrot = df["n_rotbonds"].values

	# Plot each rotatable bond count separately (low first so high overlay)
	for nr in sorted(ROTBOND_COLORS.keys()):
		mask = nrot == nr
		if not mask.any():
			continue
		ax.scatter(x[mask], y[mask], s=12, alpha=0.5, color=ROTBOND_COLORS[nr],
		           edgecolors="none", zorder=3 + nr, rasterized=True, label=str(nr))

	all_vals = np.concatenate([x, y])
	lo, hi = all_vals.min() * 0.9, all_vals.max() * 1.1
	ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)

	slope, intercept, r, _, _ = stats.linregress(x, y)
	r2 = r ** 2
	mae = np.mean(np.abs(y - x))
	rmse = np.sqrt(np.mean((y - x) ** 2))

	xfit = np.linspace(lo, hi, 200)
	ax.plot(xfit, slope * xfit + intercept, "-", color="#E8553A", lw=1.2, alpha=0.6)

	ax.set_xlim(lo, hi)
	ax.set_ylim(lo, hi)
	ax.set_aspect("equal")
	ax.set_xlabel(r"% $V_{bur}$ phenyl cap", fontsize=10)
	ax.set_ylabel(r"% $V_{bur}$ t-butyl cap", fontsize=10)
	ax.set_title(title, fontsize=11, fontweight="bold")

	stats_text = f"$R^2$ = {r2:.3f}\nMAE = {mae:.2f}%\nRMSE = {rmse:.2f}%\nn = {len(df)}"
	ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
	        va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

# Shared legend
legend_handles = [mpatches.Patch(color=ROTBOND_COLORS[n], label=str(n))
                  for n in sorted(ROTBOND_COLORS.keys()) if n in df["n_rotbonds"].values]
fig.legend(handles=legend_handles, title="Rotatable bonds", loc="lower center",
           ncol=len(legend_handles), fontsize=9, title_fontsize=9,
           bbox_to_anchor=(0.5, 0.0), frameon=True)

plt.tight_layout(rect=[0, 0.04, 1, 1])

for ext in ("png", "pdf"):
	out = HERE + f"cap_comparison.{ext}"
	fig.savefig(out, dpi=300, bbox_inches="tight")
	print(f"Saved {out}")
