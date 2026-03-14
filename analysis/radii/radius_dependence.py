"""
Analyse how buried volume accuracy depends on the sphere radius R.

Compares Bondi x1.15 VDW buried volumes against isodensity reference at
sphere radii from 2.0 to 5.0 Å, and tests how well 2D McGowan layer
regression predicts %Vbur at each radius.

Prerequisites:
    python analysis/radii/run_sampled_vbur.py --radius-sweep
    python analysis/radii/run_sampled_vbur.py --radius-sweep --isodensity

Panels:
  (a) 3D accuracy: RMSE of Bondi x1.15 vs isodensity %Vbur at each R
  (b) 2D predictability: McGowan per-element layer regression R²_CV at each R
  (c) %Vbur distribution: mean and std of isodensity %Vbur at each R
  (d) Best parity plot at the radius with highest 2D R²_CV

Usage:
    uv run python analysis/radii/radius_dependence.py
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

RADII = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
MAX_LAYERS = 6
MCGOWAN_TO_A3 = 1.0 / 6.02214076e23 * 1e24
N_FOLDS = 5
RNG_SEED = 42

ELEMENT_COLORS = {
	"C": "#555555", "H": "#AAAAAA", "N": "#4477BB",
	"O": "#DD4444", "S": "#DDAA22", "F": "#44BB88", "Cl": "#BB44BB",
}
ELEMENT_ORDER = ["C", "H", "N", "O", "S", "F", "Cl"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def mol_id_from_filename(fname):
	m = re.match(r'mol_(\d+)_', fname)
	return int(m.group(1)) if m else None


def xyz_to_mol(path):
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
	contribs = []
	for at in mol.GetAtoms():
		symbol = periodic_table[at.GetAtomicNum()]
		vol = mcgowan_volumes.get(symbol, 0.0)
		vol -= 0.5 * 6.56 * at.GetDegree()
		contribs.append(vol)
	return contribs


def layer_features(contribs, dists, max_layers):
	feats = np.zeros(max_layers + 1)
	for j, c in enumerate(contribs):
		d = int(dists[j])
		if d > max_layers:
			d = max_layers
		feats[d] += c
	return feats


def kfold_cv_regression(X, y, n_folds, seed):
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
		X_train_b = np.column_stack([X[train_idx], np.ones(len(train_idx))])
		X_test_b = np.column_stack([X[test_idx], np.ones(len(test_idx))])
		w, _, _, _ = np.linalg.lstsq(X_train_b, y[train_idx], rcond=None)
		preds[test_idx] = X_test_b @ w
	return preds


def r2_cv(y_true, y_pred):
	ss_res = np.sum((y_true - y_pred)**2)
	ss_tot = np.sum((y_true - y_true.mean())**2)
	return 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')


# ── Check available data ────────────────────────────────────────────────────

available_bondi = {}
available_iso = {}
for r in RADII:
	bf = HERE + f"bondi_x1.15_r{r:.1f}_sampled.csv"
	if os.path.exists(bf):
		available_bondi[r] = pd.read_csv(bf)
	isof = HERE + f"tz_isodensity_r{r:.1f}_sampled.csv"
	if os.path.exists(isof):
		available_iso[r] = pd.read_csv(isof)

# Also load R=3.5 from existing files as fallback
if 3.5 not in available_bondi:
	bf35 = HERE + "bondi_x1.15_sampled.csv"
	if os.path.exists(bf35):
		available_bondi[3.5] = pd.read_csv(bf35)
	else:
		# Try the sweep file
		bf35 = HERE + "bondi_x1.15_sampled.csv"
if 3.5 not in available_iso:
	isof35 = HERE + "tz_isodensity_sampled.csv"
	if os.path.exists(isof35):
		available_iso[3.5] = pd.read_csv(isof35)

print(f"Bondi x1.15 data available for R = {sorted(available_bondi.keys())}")
print(f"Isodensity data available for R = {sorted(available_iso.keys())}")

has_3d_comparison = bool(set(available_bondi) & set(available_iso))
radii_both = sorted(set(available_bondi) & set(available_iso))
radii_iso = sorted(available_iso.keys())

if not radii_iso:
	print("\nNo isodensity radius-sweep data found.")
	print("Run on server:  python analysis/radii/run_sampled_vbur.py --radius-sweep --isodensity")
	print("Run locally:    python analysis/radii/run_sampled_vbur.py --radius-sweep")
	sys.exit(0)

# ── Build 2D layer features (computed once, reused across radii) ────────────

sample_atoms = pd.read_csv(HERE + "sample_atoms.csv")
sample_by_mol = sample_atoms.groupby("mol_file")

print(f"\nBuilding 2D McGowan layer features from {len(sample_atoms)} samples...")

xyz_files = sorted(f for f in os.listdir(XYZ_DIR) if f.endswith('.xyz'))
layer_rows = []
failures = []

for i, f in enumerate(xyz_files):
	if (i + 1) % 100 == 0:
		print(f"  {i + 1}/{len(xyz_files)}")

	if f not in sample_by_mol.groups:
		continue

	path = os.path.join(XYZ_DIR, f)
	mol = xyz_to_mol(path)
	if mol is None:
		failures.append(f)
		continue

	mcg = per_atom_mcgowan(mol)
	dist_mat = rdmolops.GetDistanceMatrix(mol)
	group = sample_by_mol.get_group(f)

	for _, row in group.iterrows():
		center = row.atom1_idx - 1
		if center >= len(dist_mat):
			continue
		dists = dist_mat[center]
		mcg_layers = layer_features(mcg, dists, MAX_LAYERS) * MCGOWAN_TO_A3
		layer_rows.append({
			'mol_file': f,
			'atom1_idx': row.atom1_idx,
			'element': row.element,
			**{f'mcg_L{k}': mcg_layers[k] for k in range(MAX_LAYERS + 1)},
		})

layer_df = pd.DataFrame(layer_rows)
mcg_cols = [f'mcg_L{k}' for k in range(MAX_LAYERS + 1)]
print(f"Layer features: {len(layer_df)} samples ({len(failures)} failed bond perception)")

# ── Compute metrics at each radius ─────────────────────────────────────────

results = []

for r in sorted(set(radii_iso)):
	iso_df = available_iso[r]

	# Merge layer features with isodensity vbur
	merged = layer_df.merge(
		iso_df[["mol_file", "atom1_idx", "element", "vbur"]],
		on=["mol_file", "atom1_idx", "element"]
	).rename(columns={"vbur": "vbur_iso"})

	if len(merged) < N_FOLDS * 2:
		print(f"  R={r:.1f}: only {len(merged)} samples, skipping")
		continue

	X_mcg = merged[mcg_cols].values
	y_vbur = merged["vbur_iso"].values
	elements = merged["element"].values
	n_samples = len(merged)

	# Per-element CV regression
	mcg_cv = np.zeros(n_samples)
	for el in ELEMENT_ORDER:
		mask = elements == el
		if not mask.any() or mask.sum() < N_FOLDS:
			continue
		idx_el = np.where(mask)[0]
		mcg_cv[idx_el] = kfold_cv_regression(
			X_mcg[idx_el], y_vbur[idx_el], N_FOLDS, RNG_SEED
		)

	valid = mcg_cv != 0
	r2_2d = r2_cv(y_vbur[valid], mcg_cv[valid])
	mae_2d = np.mean(np.abs(y_vbur[valid] - mcg_cv[valid]))

	# 3D parity if Bondi data available
	r2_3d = np.nan
	rmse_3d = np.nan
	mae_3d = np.nan
	if r in available_bondi:
		bondi_df = available_bondi[r]
		m3d = merged[["mol_file", "atom1_idx", "element"]].merge(
			bondi_df[["mol_file", "atom1_idx", "element", "vbur"]],
			on=["mol_file", "atom1_idx", "element"]
		).rename(columns={"vbur": "vbur_bondi"})
		m3d = m3d.merge(
			iso_df[["mol_file", "atom1_idx", "element", "vbur"]],
			on=["mol_file", "atom1_idx", "element"]
		).rename(columns={"vbur": "vbur_iso"})
		if len(m3d) > 0:
			x3, y3 = m3d["vbur_iso"].values, m3d["vbur_bondi"].values
			r2_3d = r2_cv(x3, y3)
			rmse_3d = np.sqrt(np.mean((x3 - y3)**2))
			mae_3d = np.mean(np.abs(x3 - y3))

	row = {
		"R": r,
		"n": n_samples,
		"mean_vbur": y_vbur.mean(),
		"std_vbur": y_vbur.std(),
		"r2_2d": r2_2d,
		"mae_2d": mae_2d,
		"r2_3d": r2_3d,
		"rmse_3d": rmse_3d,
		"mae_3d": mae_3d,
	}
	results.append(row)
	print(f"  R={r:.1f} Å: n={n_samples}, mean %Vbur={y_vbur.mean():.1f}±{y_vbur.std():.1f}, "
	      f"2D R²_CV={r2_2d:.3f}, 3D R²={r2_3d:.3f}")

res_df = pd.DataFrame(results)
print(f"\n{res_df.to_string(index=False)}")

# ── Figure ──────────────────────────────────────────────────────────────────

col_3d = "#2176AE"
col_2d = "#E8553A"
col_dist = "#57A773"

n_panels = 2
if has_3d_comparison:
	n_panels = 3

fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
if n_panels == 1:
	axes = [axes]

panel_idx = 0

# Panel: 3D accuracy vs R (if data available)
if has_3d_comparison:
	ax = axes[panel_idx]
	r_arr = res_df["R"].values
	ax.plot(r_arr, res_df["rmse_3d"].values, "o-", color=col_3d, lw=2, ms=7,
	        label="RMSE (%)")
	ax.plot(r_arr, res_df["mae_3d"].values, "s--", color=col_3d, lw=1.5, ms=6,
	        alpha=0.7, label="MAE (%)")
	ax.set_xlabel("Sphere radius R (Å)", fontsize=11)
	ax.set_ylabel("Bondi ×1.15 vs Isodensity (%)", fontsize=11)
	ax.set_title("(a) 3D Accuracy vs Sphere Radius", fontsize=11, fontweight="bold")
	ax.legend(fontsize=9)
	ax.set_xticks(RADII)
	panel_idx += 1

# Panel: 2D R²_CV vs R
ax = axes[panel_idx]
r_arr = res_df["R"].values
ax.plot(r_arr, res_df["r2_2d"].values, "o-", color=col_2d, lw=2, ms=7)
ax.set_xlabel("Sphere radius R (Å)", fontsize=11)
ax.set_ylabel(r"McGowan per-element $R^2_{CV}$", fontsize=11)
label = "(b)" if has_3d_comparison else "(a)"
ax.set_title(f"{label} 2D Predictability vs Sphere Radius", fontsize=11, fontweight="bold")
ax.set_xticks(RADII)
# Annotate best R
best_idx = res_df["r2_2d"].idxmax()
best_r = res_df.loc[best_idx, "R"]
best_r2 = res_df.loc[best_idx, "r2_2d"]
ax.annotate(f"R={best_r:.1f} Å\n$R^2_{{CV}}$={best_r2:.3f}", fontsize=8,
            xy=(best_r, best_r2),
            xytext=(best_r + 0.3, best_r2 - 0.05),
            arrowprops=dict(arrowstyle="->", color=col_2d, lw=0.8))
panel_idx += 1

# Panel: %Vbur distribution vs R
ax = axes[panel_idx]
ax.errorbar(r_arr, res_df["mean_vbur"].values, yerr=res_df["std_vbur"].values,
            fmt="o-", color=col_dist, lw=2, ms=7, capsize=4)
ax.set_xlabel("Sphere radius R (Å)", fontsize=11)
ax.set_ylabel(r"Isodensity %$V_{bur}$", fontsize=11)
label = "(c)" if has_3d_comparison else "(b)"
ax.set_title(f"{label} %$V_{{bur}}$ Distribution vs Sphere Radius", fontsize=11, fontweight="bold")
ax.set_xticks(RADII)

plt.tight_layout()

for ext in ("png", "pdf"):
	out = HERE + f"radius_dependence.{ext}"
	fig.savefig(out, dpi=300, bbox_inches="tight")
	print(f"Saved {out}")
plt.show()
