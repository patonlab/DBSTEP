"""PCA of radial buried volume profiles for phenyl vs t-butyl caps.

2x2 plot (Boltzmann, min-E, min, max). Each panel uses 7 radial %V_bur
values (2.5–8.5 Å) as dimensions, standardised and projected onto PC1/PC2.
Key functional groups are highlighted and labelled.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

HERE = __file__.replace("plot_pca_radial.py", "")

RADII = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]

# Functional groups to highlight (fragment_smiles → label)
HIGHLIGHT_FGS = {
	# Upper-left: small, branched/flexible
	"*C": "Me",
	"*OC": "OMe",
	"*CC": "Et",
	"*C(C)C": "iPr",
	"*C(C)(C)C": "tBu",
	"*CC(C)C": "iBu",
	"*[C@@H](C)CC": "sBu",
	"*[C@H](C)CC": "sBu",
	"*c1ccccc1": "Ph",
	"*C(F)(F)F": "CF₃",
	# Upper-right: bulky substituents
	"*c1ccccc1Cl": "o-ClPh",
	"*N1CCCCCC1": "azepane",
	"*c1ccccc1[N+](=O)[O-]": "o-NO₂Ph",
	"*c1ccccc1S(C)(=O)=O": "PhSO₂Me",
	# Left: small heterocyclic
	"*n1cncn1": "1,2,4-triazolyl",
	"*n1cnnn1": "tetrazolyl",
	# Lower-left: flat, extended aromatics
	"*c1nc2ccccc2s1": "benzothiazolyl",
	"*c1nc2ccccc2[nH]1": "benzimidazolyl",
	"*N": "NH₂",
	# Lower-center: fused bicyclics with NH
	"*N[C@H]1CCc2ccccc21": "indanyl-NH",
	"*N[C@@H]1CCc2ccccc21": "indanyl-NH",
	# Lower-right: constrained fused rings
	"*N1C(=O)c2ccccc2C1=O": "phthalimidyl",
	"*N1CCc2ccccc21": "indolinyl",
}

# ── Load & merge ─────────────────────────────────────────────────────────────

ph_cap = pd.read_csv(HERE + "zinc_fragments_phenyl_capped.csv")[["name", "fragment_smiles"]].drop_duplicates("name")
tb_cap = pd.read_csv(HERE + "zinc_fragments_t_butyl_capped.csv")[["name", "fragment_smiles"]].drop_duplicates("name")

ph_vbur = pd.read_csv(HERE + "zinc_fragments_phenyl_capped_multi_r_buried_vol_summary.csv")
tb_vbur = pd.read_csv(HERE + "zinc_fragments_t_butyl_capped_multi_r_buried_vol_summary.csv")

ph = ph_cap.merge(ph_vbur, on="name")
tb = tb_cap.merge(tb_vbur, on="name")
df = ph.merge(tb, on="fragment_smiles", suffixes=("_ph", "_tbu"))
print(f"Matched {len(df)} fragments")

# Map highlight labels (take first match for sBu enantiomers)
df["fg_label"] = df["fragment_smiles"].map(HIGHLIGHT_FGS)
highlight_mask = df["fg_label"].notna()
# Deduplicate sBu (keep first occurrence)
seen_labels = set()
dedup_idx = []
for idx, label in df["fg_label"].items():
	if pd.notna(label):
		if label not in seen_labels:
			seen_labels.add(label)
			dedup_idx.append(idx)
		else:
			dedup_idx.append(idx)  # still highlight, just don't label twice

# ── PCA via SVD ──────────────────────────────────────────────────────────────

def pca_2d(X):
	"""Standardise columns, compute PCA via SVD, return scores and % variance."""
	mu = X.mean(axis=0)
	std = X.std(axis=0, ddof=0)
	std[std == 0] = 1.0
	Z = (X - mu) / std
	U, S, Vt = np.linalg.svd(Z, full_matrices=False)
	explained = (S ** 2) / (S ** 2).sum() * 100
	scores = Z @ Vt[:2].T
	return scores, explained[:2]

# ── Figure ───────────────────────────────────────────────────────────────────

metrics = ["boltz", "min_E", "min", "max"]
titles = ["Boltzmann-weighted", "Lowest-E conformer",
          "Minimum across conformers", "Maximum across conformers"]

CAP_COLORS = {"Phenyl": "#2176AE", "t-Butyl": "#E8553A"}

# Per-label annotation offsets (dx, dy in points) and alignment overrides
LABEL_OFFSETS = {
	"Me":               (-12, -12),
	"OMe":              (  6, -10),
	"Et":               (-14,   4),
	"iPr":              (-10,   6),
	"tBu":              (  0,   8),
	"iBu":              (-14,  -4),
	"sBu":              (  6,   8),
	"Ph":               (  8, -10),
	"CF₃":              (-14,   6),
	"o-ClPh":           (  8,  -8),
	"azepane":          (  8,   6),
	"o-NO₂Ph":          (  8,  -8),
	"PhSO₂Me":          (  8,   6),
	"1,2,4-triazolyl":  (  8,   6),
	"tetrazolyl":       (-14,   8),
	"benzothiazolyl":   (  8,   8),
	"benzimidazolyl":   (  8, -10),
	"NH₂":              (  0, -12),
	"indanyl-NH":       (  8,  -4),
	"phthalimidyl":     (-14,   6),
	"indolinyl":        (  8,   6),
}

fig = plt.figure(figsize=(11, 15))
gs = fig.add_gridspec(3, 4, height_ratios=[0.8, 2, 2], hspace=0.4, wspace=0.35)

# ── Top row: histograms of %V_bur at r=3.5 Å ────────────────────────────────

hist_r = 3.5
hist_bins = np.arange(0, 51, 2)

for col_idx, (metric, title) in enumerate(zip(metrics, titles)):
	ax_h = fig.add_subplot(gs[0, col_idx])
	ph_col = f"{metric}_r{hist_r}_ph"
	tb_col = f"{metric}_r{hist_r}_tbu"

	ax_h.hist(df[ph_col], bins=hist_bins, alpha=0.6, color=CAP_COLORS["Phenyl"],
	          label="Phenyl", edgecolor="none", density=True)
	ax_h.hist(df[tb_col], bins=hist_bins, alpha=0.6, color=CAP_COLORS["t-Butyl"],
	          label="t-Butyl", edgecolor="none", density=True)

	ax_h.set_xlabel("%V$_{bur}$ (r = 3.5 Å)", fontsize=9)
	ax_h.set_ylabel("Density", fontsize=9)
	ax_h.set_title(title, fontsize=10, fontweight="bold")
	ax_h.tick_params(labelsize=8)

# ── Bottom 2×2: PCA panels ───────────────────────────────────────────────────

pca_axes = [
	fig.add_subplot(gs[1, 0:2]),  # Boltzmann
	fig.add_subplot(gs[1, 2:4]),  # min-E
	fig.add_subplot(gs[2, 0:2]),  # min
	fig.add_subplot(gs[2, 2:4]),  # max
]
metric_to_ax = {i: pca_axes[i] for i in range(4)}

for panel_idx, (metric, title) in enumerate(zip(metrics, titles)):
	ax = metric_to_ax[panel_idx]

	# Build feature matrix (n_fragments × 7 radii) — phenyl only
	ph_cols = [f"{metric}_r{r}_ph" for r in RADII]
	X_ph = df[ph_cols].values
	scores, explained = pca_2d(X_ph)

	n = len(df)
	scores_ph = scores

	# Background points: non-highlighted fragments
	bg = ~highlight_mask.values
	ax.scatter(scores_ph[bg, 0], scores_ph[bg, 1], s=10, alpha=0.25,
	           color=CAP_COLORS["Phenyl"], edgecolors="none",
	           rasterized=True, zorder=4)

	# Highlighted FG points: phenyl cap only, labelled
	hi_idx = highlight_mask.values.nonzero()[0]

	ax.scatter(scores_ph[hi_idx, 0], scores_ph[hi_idx, 1], s=60, alpha=0.9,
	           color=CAP_COLORS["Phenyl"], edgecolors="k", linewidths=0.5,
	           zorder=7)

	labelled = set()
	for i in hi_idx:
		label = df.iloc[i]["fg_label"]
		if label in labelled:
			continue
		labelled.add(label)
		dx, dy = LABEL_OFFSETS.get(label, (0, 6))
		ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
		va = "bottom" if dy > 0 else ("top" if dy < 0 else "center")
		ax.annotate(label, (scores_ph[i, 0], scores_ph[i, 1]),
		            fontsize=7.5, fontweight="bold",
		            ha=ha, va=va, xytext=(dx, dy),
		            textcoords="offset points",
		            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8, lw=0),
		            zorder=8)

	ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)", fontsize=10)
	ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)", fontsize=10)
	ax.set_title(title, fontsize=11, fontweight="bold")

	stats_text = f"PC1: {explained[0]:.1f}%\nPC2: {explained[1]:.1f}%\nn = {n} fragments"
	ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
	        va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

	ax.axhline(0, color="k", lw=0.5, alpha=0.3)
	ax.axvline(0, color="k", lw=0.5, alpha=0.3)

# Shared legend
legend_handles = [
	mlines.Line2D([], [], marker="o", color=CAP_COLORS["Phenyl"], linestyle="None",
	              markersize=6, label="Phenyl cap"),
	mlines.Line2D([], [], marker="s", color=CAP_COLORS["t-Butyl"], linestyle="None",
	              markersize=6, label="t-Butyl cap (histograms only)"),
]
fig.legend(handles=legend_handles, loc="lower center",
           ncol=2, fontsize=10,
           bbox_to_anchor=(0.5, 0.0), frameon=True)

for ext in ("png", "pdf"):
	out = HERE + f"pca_radial.{ext}"
	fig.savefig(out, dpi=300, bbox_inches="tight")
	print(f"Saved {out}")
