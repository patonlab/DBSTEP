"""
Compare 2D graph-based volumes (McGowan, Crippen) against TZ isodensity reference.

Produces a 2x2 figure:
  (a) McGowan molecular volume vs isodensity molecular volume
  (b) Crippen molar refractivity vs isodensity molecular volume
  (c) McGowan layer-regression predicted %Vbur vs isodensity %Vbur
  (d) Crippen layer-regression predicted %Vbur vs isodensity %Vbur

Bottom row uses per-layer atomic contributions (binned by graph distance from
the center atom) as features in a linear regression, evaluated with 5-fold CV.

Usage:
    uv run python analysis/radii/2d_parity_plots.py
"""

import os, sys, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdmolops, Crippen

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dbstep.constants import mcgowan_volumes, periodic_table

HERE = os.path.dirname(os.path.abspath(__file__)) + os.sep
XYZ_DIR = os.path.join(HERE, "xyz_files")

# Maximum graph distance (bonds) for layer features
MAX_LAYERS = 6

# McGowan cm³/mol → Å³/molecule
MCGOWAN_TO_A3 = 1.0 / 6.02214076e23 * 1e24  # ≈ 1.6611

ELEMENT_COLORS = {
	"C": "#555555", "H": "#AAAAAA", "N": "#4477BB",
	"O": "#DD4444", "S": "#DDAA22", "F": "#44BB88", "Cl": "#BB44BB",
}
ELEMENT_ORDER = ["C", "H", "N", "O", "S", "F", "Cl"]

N_FOLDS = 5
RNG_SEED = 42


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_dbstep_file(filepath):
	data = {}
	with open(filepath) as f:
		for line in f:
			m = re.match(r'\s+mol_(\d+)_\d+\S*\s+\d+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)', line)
			if m:
				data[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
	return data


def mol_id_from_filename(fname):
	m = re.match(r'mol_(\d+)_', fname)
	return int(m.group(1)) if m else None


def xyz_to_mol(path):
	"""Read xyz file and determine bonds using RDKit."""
	raw = Chem.MolFromXYZFile(path)
	if raw is None:
		return None
	try:
		rdDetermineBonds.DetermineBonds(raw)
		Chem.SanitizeMol(raw)
	except Exception:
		return None
	return raw


def per_atom_mcgowan(mol):
	"""Per-atom McGowan volume contributions (all atoms incl. H), bond-corrected."""
	contribs = []
	for at in mol.GetAtoms():
		symbol = periodic_table[at.GetAtomicNum()]
		vol = mcgowan_volumes.get(symbol, 0.0)
		vol -= 0.5 * 6.56 * at.GetDegree()
		contribs.append(vol)
	return contribs


def per_atom_crippen_mr(mol):
	"""Per-atom Crippen molar refractivity contributions (all atoms incl. H)."""
	mrContribs = Crippen.rdMolDescriptors._CalcCrippenContribs(mol)
	_, mrs = zip(*mrContribs)
	return list(mrs)


def layer_features(contribs, dists, max_layers):
	"""Bin per-atom contributions by integer graph distance into layer features."""
	feats = np.zeros(max_layers + 1)
	for j, c in enumerate(contribs):
		d = int(dists[j])
		if d > max_layers:
			d = max_layers
		feats[d] += c
	return feats


def calc_stats(x, y):
	slope, intercept, r, _, _ = stats.linregress(x, y)
	return r**2, np.mean(np.abs(y - x)), np.sqrt(np.mean((y - x)**2)), slope, intercept


def kfold_cv_regression(X, y, n_folds, seed, nonneg=False):
	"""K-fold cross-validated linear regression. Returns CV predictions.
	If nonneg=True, layer weights are constrained >= 0 (intercept is unconstrained)."""
	from scipy.optimize import nnls
	n = len(y)
	rng = np.random.RandomState(seed)
	idx = rng.permutation(n)
	fold_size = n // n_folds
	preds = np.zeros(n)

	for k in range(n_folds):
		start = k * fold_size
		end = (k + 1) * fold_size if k < n_folds - 1 else n
		test_idx = idx[start:end]
		train_idx = np.concatenate([idx[:start], idx[end:]])

		X_train, y_train = X[train_idx], y[train_idx]
		X_test = X[test_idx]

		if nonneg:
			# Center y so intercept is handled separately
			y_mean = y_train.mean()
			w, _ = nnls(X_train, y_train - y_mean)
			preds[test_idx] = X_test @ w + y_mean
		else:
			X_train_b = np.column_stack([X_train, np.ones(len(X_train))])
			X_test_b = np.column_stack([X_test, np.ones(len(X_test))])
			w, _, _, _ = np.linalg.lstsq(X_train_b, y_train, rcond=None)
			preds[test_idx] = X_test_b @ w

	return preds


def fit_full_regression(X, y, nonneg=False):
	"""Fit linear regression on all data. Returns weights and intercept.
	If nonneg=True, layer weights are constrained >= 0."""
	from scipy.optimize import nnls
	if nonneg:
		y_mean = y.mean()
		w, _ = nnls(X, y - y_mean)
		intercept = y_mean - X.mean(axis=0) @ w + y_mean  # recompute properly
		# Actually: pred = X @ w + y_mean, so intercept = y_mean
		return w, y_mean
	else:
		X_b = np.column_stack([X, np.ones(len(X))])
		w, _, _, _ = np.linalg.lstsq(X_b, y, rcond=None)
		return w[:-1], w[-1]


def parity_panel_cv(ax, x_true, y_pred, color, title, xlabel, ylabel, unit,
                    elements=None):
	"""Parity panel for cross-validated predictions."""
	if elements is not None:
		for el in ELEMENT_ORDER:
			mask = elements == el
			if mask.any():
				ax.scatter(x_true[mask], y_pred[mask], s=18, alpha=0.65,
				           color=ELEMENT_COLORS[el], edgecolors="none",
				           label=el, zorder=3, rasterized=True)
	else:
		ax.scatter(x_true, y_pred, s=18, alpha=0.65, color=color,
		           edgecolors="none", zorder=3, rasterized=True)

	all_vals = np.concatenate([x_true, y_pred])
	lo, hi = all_vals.min() * 0.9, all_vals.max() * 1.1
	ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)

	r2 = 1 - np.sum((x_true - y_pred)**2) / np.sum((x_true - x_true.mean())**2)
	mae = np.mean(np.abs(x_true - y_pred))
	rmse = np.sqrt(np.mean((x_true - y_pred)**2))

	ax.set_xlim(lo, hi)
	ax.set_ylim(lo, hi)
	ax.set_aspect("equal")
	ax.set_xlabel(xlabel, fontsize=10)
	ax.set_ylabel(ylabel, fontsize=10)
	ax.set_title(title, fontsize=10, fontweight="bold")

	stats_text = f"$R^2_{{CV}}$ = {r2:.3f}\nMAE = {mae:.1f} {unit}\nRMSE = {rmse:.1f} {unit}\nn = {len(x_true)}"
	ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
	        va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))


# ── Load reference data ──────────────────────────────────────────────────────

ref_txt = parse_dbstep_file(HERE + "tz_isodensity_volumes.txt")
iso_sampled = pd.read_csv(HERE + "tz_isodensity_sampled.csv")
sample_atoms = pd.read_csv(HERE + "sample_atoms.csv")

sample_with_iso = sample_atoms.merge(
	iso_sampled[["mol_file", "atom1_idx", "element", "vbur"]],
	on=["mol_file", "atom1_idx", "element"]
).rename(columns={"vbur": "vbur_iso"})

print(f"Reference: {len(ref_txt)} molecules, {len(sample_with_iso)} buried vol samples")

sample_by_mol = sample_with_iso.groupby("mol_file")

# ── Process xyz files ────────────────────────────────────────────────────────

mol_vol_rows = []
bvur_rows = []
failures = []

xyz_files = sorted(f for f in os.listdir(XYZ_DIR) if f.endswith('.xyz'))
print(f"Processing {len(xyz_files)} xyz files...")

for i, f in enumerate(xyz_files):
	if (i + 1) % 100 == 0:
		print(f"  {i + 1}/{len(xyz_files)}")

	path = os.path.join(XYZ_DIR, f)
	mol = xyz_to_mol(path)
	if mol is None:
		failures.append(f)
		continue

	mid = mol_id_from_filename(f)

	# Per-atom contributions
	mcg = per_atom_mcgowan(mol)
	try:
		cr = per_atom_crippen_mr(mol)
	except Exception:
		failures.append(f)
		continue

	mcg_total_a3 = sum(mcg) * MCGOWAN_TO_A3
	cr_total = sum(cr)

	# Molecular volume comparison
	if mid in ref_txt:
		mol_vol_rows.append({
			'mol_id': mid,
			'mcgowan_vol': mcg_total_a3,
			'crippen_mr': cr_total,
			'iso_vol': ref_txt[mid][0],
		})

	# Buried volume comparison — compute per-layer features
	if f in sample_by_mol.groups:
		dist_mat = rdmolops.GetDistanceMatrix(mol)
		group = sample_by_mol.get_group(f)

		for _, row in group.iterrows():
			center = row.atom1_idx - 1  # 1-based → 0-based
			if center >= len(dist_mat):
				continue

			dists = dist_mat[center]

			mcg_layers = layer_features(mcg, dists, MAX_LAYERS) * MCGOWAN_TO_A3
			cr_layers = layer_features(cr, dists, MAX_LAYERS)

			bvur_rows.append({
				'mol_file': f,
				'atom1_idx': row.atom1_idx,
				'element': row.element,
				'vbur_iso': row.vbur_iso,
				**{f'mcg_L{k}': mcg_layers[k] for k in range(MAX_LAYERS + 1)},
				**{f'cr_L{k}': cr_layers[k] for k in range(MAX_LAYERS + 1)},
			})

if failures:
	print(f"WARNING: {len(failures)} molecules failed bond perception")

mol_df = pd.DataFrame(mol_vol_rows)
bvur_df = pd.DataFrame(bvur_rows)

print(f"Molecular volume: {len(mol_df)} molecules")
print(f"Buried volume: {len(bvur_df)} samples")

# ── Layer regression for buried volume ───────────────────────────────────────

mcg_cols = [f'mcg_L{k}' for k in range(MAX_LAYERS + 1)]
cr_cols = [f'cr_L{k}' for k in range(MAX_LAYERS + 1)]

X_mcg = bvur_df[mcg_cols].values
X_cr = bvur_df[cr_cols].values
y_vbur = bvur_df["vbur_iso"].values
elements = bvur_df["element"].values

# Cross-validated predictions (non-negative layer weights)
mcg_cv_preds = kfold_cv_regression(X_mcg, y_vbur, N_FOLDS, RNG_SEED, nonneg=False)
cr_cv_preds = kfold_cv_regression(X_cr, y_vbur, N_FOLDS, RNG_SEED, nonneg=False)

# Full-data fit to report weights
mcg_weights, mcg_intercept = fit_full_regression(X_mcg, y_vbur, nonneg=False)
cr_weights, cr_intercept = fit_full_regression(X_cr, y_vbur, nonneg=False)

print(f"\nMcGowan layer regression weights (layers 0–{MAX_LAYERS}):")
for k in range(MAX_LAYERS + 1):
	print(f"  Layer {k}: {mcg_weights[k]:+.4f}")
print(f"  Intercept: {mcg_intercept:.2f}")

print(f"\nCrippen layer regression weights (layers 0–{MAX_LAYERS}):")
for k in range(MAX_LAYERS + 1):
	print(f"  Layer {k}: {cr_weights[k]:+.4f}")
print(f"  Intercept: {cr_intercept:.2f}")

# Per-element stats
print(f"\n{'Element':<6} {'n':>5}  {'McG R²_CV':>9} {'McG MAE':>8} {'McG RMSE':>9}  {'Cr R²_CV':>9} {'Cr MAE':>8} {'Cr RMSE':>9}")
for el in ELEMENT_ORDER:
	mask = elements == el
	if not mask.any():
		continue
	n_el = mask.sum()
	y_el = y_vbur[mask]
	ss_tot = np.sum((y_el - y_el.mean())**2)
	for label, preds in [("mcg", mcg_cv_preds), ("cr", cr_cv_preds)]:
		p_el = preds[mask]
		r2 = 1 - np.sum((y_el - p_el)**2) / ss_tot if ss_tot > 0 else float('nan')
		mae = np.mean(np.abs(y_el - p_el))
		rmse = np.sqrt(np.mean((y_el - p_el)**2))
		if label == "mcg":
			mcg_r2, mcg_mae, mcg_rmse = r2, mae, rmse
		else:
			cr_r2, cr_mae, cr_rmse = r2, mae, rmse
	print(f"{el:<6} {n_el:>5}  {mcg_r2:>9.3f} {mcg_mae:>7.1f}% {mcg_rmse:>8.1f}%  {cr_r2:>9.3f} {cr_mae:>7.1f}% {cr_rmse:>8.1f}%")

# Per-element fits (separate model per element)
print(f"\n--- Per-element fits (separate regression per element) ---")
print(f"{'Element':<6} {'n':>5}  {'McG R²_CV':>9} {'McG MAE':>8} {'McG RMSE':>9}  {'Cr R²_CV':>9} {'Cr MAE':>8} {'Cr RMSE':>9}")
mcg_cv_per_el = np.zeros(len(y_vbur))
cr_cv_per_el = np.zeros(len(y_vbur))
for el in ELEMENT_ORDER:
	mask = elements == el
	if not mask.any() or mask.sum() < N_FOLDS:
		continue
	idx_el = np.where(mask)[0]
	n_el = len(idx_el)
	X_m_el, X_c_el, y_el = X_mcg[idx_el], X_cr[idx_el], y_vbur[idx_el]
	mcg_p = kfold_cv_regression(X_m_el, y_el, N_FOLDS, RNG_SEED, nonneg=False)
	cr_p = kfold_cv_regression(X_c_el, y_el, N_FOLDS, RNG_SEED, nonneg=False)
	mcg_cv_per_el[idx_el] = mcg_p
	cr_cv_per_el[idx_el] = cr_p
	ss_tot = np.sum((y_el - y_el.mean())**2)
	mcg_r2 = 1 - np.sum((y_el - mcg_p)**2) / ss_tot if ss_tot > 0 else float('nan')
	cr_r2 = 1 - np.sum((y_el - cr_p)**2) / ss_tot if ss_tot > 0 else float('nan')
	mcg_mae = np.mean(np.abs(y_el - mcg_p))
	cr_mae = np.mean(np.abs(y_el - cr_p))
	mcg_rmse = np.sqrt(np.mean((y_el - mcg_p)**2))
	cr_rmse = np.sqrt(np.mean((y_el - cr_p)**2))
	print(f"{el:<6} {n_el:>5}  {mcg_r2:>9.3f} {mcg_mae:>7.1f}% {mcg_rmse:>8.1f}%  {cr_r2:>9.3f} {cr_mae:>7.1f}% {cr_rmse:>8.1f}%")

# Overall stats from per-element fits
valid = mcg_cv_per_el != 0  # all elements with fits
ss_tot_all = np.sum((y_vbur[valid] - y_vbur[valid].mean())**2)
mcg_r2_pe = 1 - np.sum((y_vbur[valid] - mcg_cv_per_el[valid])**2) / ss_tot_all
cr_r2_pe = 1 - np.sum((y_vbur[valid] - cr_cv_per_el[valid])**2) / ss_tot_all
mcg_mae_pe = np.mean(np.abs(y_vbur[valid] - mcg_cv_per_el[valid]))
cr_mae_pe = np.mean(np.abs(y_vbur[valid] - cr_cv_per_el[valid]))
print(f"{'TOTAL':<6} {valid.sum():>5}  {mcg_r2_pe:>9.3f} {mcg_mae_pe:>7.1f}%{'':>9}  {cr_r2_pe:>9.3f} {cr_mae_pe:>7.1f}%")

# ── Figure ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
col_mcg = "#2176AE"
col_cr = "#E8553A"

# (a) McGowan mol vol vs isodensity mol vol
from functools import partial

ax = axes[0, 0]
x = mol_df["iso_vol"].values
y = mol_df["mcgowan_vol"].values
ax.scatter(x, y, s=18, alpha=0.65, color=col_mcg, edgecolors="none", rasterized=True)
r2, mae, rmse, slope, intercept = calc_stats(x, y)
all_vals = np.concatenate([x, y])
lo, hi = all_vals.min() * 0.9, all_vals.max() * 1.1
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
xfit = np.linspace(lo, hi, 200)
ax.plot(xfit, slope * xfit + intercept, "-", color=col_mcg, lw=1.2, alpha=0.8)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
ax.set_xlabel(r"Isodensity Mol. Vol. ($\AA^3$)", fontsize=10)
ax.set_ylabel(r"McGowan Mol. Vol. ($\AA^3$)", fontsize=10)
ax.set_title("(a) McGowan Mol. Volume", fontsize=10, fontweight="bold")
ax.text(0.05, 0.95,
        f"$R^2$ = {r2:.3f}\nMAE = {mae:.1f} $\\AA^3$\nRMSE = {rmse:.1f} $\\AA^3$\nn = {len(x)}",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

# (b) Crippen MR vs isodensity mol vol
ax = axes[0, 1]
x = mol_df["iso_vol"].values
y = mol_df["crippen_mr"].values
ax.scatter(x, y, s=18, alpha=0.65, color=col_cr, edgecolors="none", rasterized=True)
r2, mae, rmse, slope, intercept = calc_stats(x, y)
lo_x, hi_x = x.min() * 0.9, x.max() * 1.1
lo_y, hi_y = y.min() * 0.9, y.max() * 1.1
xfit = np.linspace(lo_x, hi_x, 200)
ax.plot(xfit, slope * xfit + intercept, "-", color=col_cr, lw=1.2, alpha=0.8)
ax.set_xlim(lo_x, hi_x)
ax.set_ylim(lo_y, hi_y)
ax.set_xlabel(r"Isodensity Mol. Vol. ($\AA^3$)", fontsize=10)
ax.set_ylabel(r"Crippen Molar Refractivity (cm$^3$/mol)", fontsize=10)
ax.set_title("(b) Crippen Molar Refractivity", fontsize=10, fontweight="bold")
ax.text(0.05, 0.95, f"$R^2$ = {r2:.3f}\nn = {len(x)}",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

# (c) McGowan per-element layer-regression CV predicted %Vbur vs isodensity %Vbur
parity_panel_cv(
	axes[1, 0],
	y_vbur, mcg_cv_per_el,
	color=col_mcg,
	title=r"(c) McGowan per-element layer regression %$V_{bur}$",
	xlabel=r"Isodensity %$V_{bur}$",
	ylabel=r"Predicted %$V_{bur}$ (5-fold CV)",
	unit="%",
	elements=pd.Series(elements),
)

# (d) Crippen per-element layer-regression CV predicted %Vbur vs isodensity %Vbur
parity_panel_cv(
	axes[1, 1],
	y_vbur, cr_cv_per_el,
	color=col_cr,
	title=r"(d) Crippen per-element layer regression %$V_{bur}$",
	xlabel=r"Isodensity %$V_{bur}$",
	ylabel=r"Predicted %$V_{bur}$ (5-fold CV)",
	unit="%",
	elements=pd.Series(elements),
)

# Shared element legend
legend_handles = [
	mpatches.Patch(color=ELEMENT_COLORS[el], label=el)
	for el in ELEMENT_ORDER if el in elements
]
fig.legend(handles=legend_handles, title="Center atom", loc="lower center",
           ncol=len(legend_handles), fontsize=9, title_fontsize=9,
           bbox_to_anchor=(0.5, 0.0), frameon=True)

plt.tight_layout(rect=[0, 0.04, 1, 1])

for ext in ("png", "pdf"):
	out = HERE + f"2d_parity_plots.{ext}"
	fig.savefig(out, dpi=300, bbox_inches="tight")
	print(f"Saved {out}")
plt.show()
