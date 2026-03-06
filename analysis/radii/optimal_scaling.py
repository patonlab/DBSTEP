"""
Find the optimal Bondi radius scaling factor for reproducing isodensity
molecular volumes and buried volumes.

Uses actual DBSTEP runs at s = 1.00, 1.05–1.15 in steps of 0.01.
Molecular volume reference: isodensity_volumes.txt (atom1=1, 486 molecules).
Mol vol from sweep CSVs is deduplicated per molecule (mol_vol is independent
of atom1 choice).
Buried volume reference: isodensity_sampled.csv (1044 atom centers).

Run the sweep first:
    python analysis/radii/run_sampled_vbur.py --sweep

Panels:
  (a) RMSE vs scaling factor for mol vol and %V_bur (actual computed values)
  (b) Parity for mol vol at its optimal scaling
  (c) Parity for %V_bur at its optimal scaling, coloured by element
"""

import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

HERE = __file__.replace("optimal_scaling.py", "")

ELEMENT_COLORS = {
    "C":  "#555555",
    "H":  "#AAAAAA",
    "N":  "#4477BB",
    "O":  "#DD4444",
    "S":  "#DDAA22",
    "F":  "#44BB88",
    "Cl": "#BB44BB",
}
ELEMENT_ORDER = ["C", "H", "N", "O", "S", "F", "Cl"]

SWEEP_SCALES = [1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.20]


def parse_dbstep_file(filepath):
    data = {}
    with open(filepath) as f:
        for line in f:
            m = re.match(r'\s+mol_(\d+)_\d+\S*\s+\d+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)', line)
            if m:
                data[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return data


def mol_id_from_filename(fname):
    """Extract integer mol id from e.g. 'mol_0042_0001.xyz' → 42."""
    m = re.match(r'mol_(\d+)_', fname)
    return int(m.group(1)) if m else None


def calc_stats(x, y):
    slope, intercept, r, _, _ = stats.linregress(x, y)
    return r**2, np.mean(np.abs(y - x)), np.sqrt(np.mean((y - x)**2)), slope, intercept


# ── Reference data ────────────────────────────────────────────────────────────
ref_txt   = parse_dbstep_file(HERE + "isodensity_volumes.txt")
iso_sampled = pd.read_csv(HERE + "isodensity_sampled.csv")

# ── Load sweep CSVs ───────────────────────────────────────────────────────────
sweep = {}   # scale -> DataFrame with columns [mol_file, atom1_idx, element, mol_vol, vbur]
missing = []
for s in SWEEP_SCALES:
    fname = HERE + ("bondi_sampled.csv" if s == 1.00 else f"bondi_x{s:.2f}_sampled.csv")
    if not os.path.exists(fname):
        missing.append(s)
        continue
    sweep[s] = pd.read_csv(fname)

if missing:
    print(f"WARNING: missing sweep files for s = {missing}")
    print("Run:  python analysis/radii/run_sampled_vbur.py --sweep")

scales = sorted(sweep)
print(f"Scales available: {scales}")

# ── Compute RMSE / MAE for mol_vol and vbur at each scale ────────────────────
rmse_vol  = {}
rmse_vbur = {}
mae_vol   = {}
mae_vbur  = {}

for s, df in sweep.items():
    # Mol vol: deduplicate per molecule, match to isodensity reference
    mol_vol_df = df[["mol_file", "mol_vol"]].drop_duplicates("mol_file").copy()
    mol_vol_df["mol_id"] = mol_vol_df["mol_file"].map(mol_id_from_filename)
    mol_vol_df = mol_vol_df[mol_vol_df["mol_id"].isin(ref_txt)]
    mol_vol_df["iso_vol"] = mol_vol_df["mol_id"].map(lambda i: ref_txt[i][0])
    x_v, y_v = mol_vol_df["mol_vol"].values, mol_vol_df["iso_vol"].values
    rmse_vol[s] = np.sqrt(np.mean((x_v - y_v)**2))
    mae_vol[s]  = np.mean(np.abs(x_v - y_v))

    # Buried vol: merge with isodensity_sampled on (mol_file, atom1_idx, element)
    merged = df.merge(iso_sampled, on=["mol_file", "atom1_idx", "element"],
                      suffixes=("_b", "_iso"))
    x_b, y_b = merged["vbur_b"].values, merged["vbur_iso"].values
    rmse_vbur[s] = np.sqrt(np.mean((x_b - y_b)**2))
    mae_vbur[s]  = np.mean(np.abs(x_b - y_b))

    n_vol = len(mol_vol_df)
    n_bur = len(merged)
    print(f"  s={s:.2f}: mol_vol RMSE={rmse_vol[s]:.1f} Å³  MAE={mae_vol[s]:.1f} Å³  (n={n_vol})"
          f"  |  vbur RMSE={rmse_vbur[s]:.2f}%  MAE={mae_vbur[s]:.2f}%  (n={n_bur})")

s_opt_vol  = min(rmse_vol,  key=rmse_vol.get)
s_opt_vbur = min(rmse_vbur, key=rmse_vbur.get)
print(f"\nOptimal s — mol vol:  {s_opt_vol:.2f}  (RMSE = {rmse_vol[s_opt_vol]:.1f} Å³)")
print(f"Optimal s — %V_bur:  {s_opt_vbur:.2f}  (RMSE = {rmse_vbur[s_opt_vbur]:.2f} %)")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

col_vol  = "#2176AE"
col_vbur = "#E8553A"

# Panel (a): RMSE vs s — both mol vol and vbur
ax1 = axes[0]
s_arr = np.array(scales)
l1, = ax1.plot(s_arr, [rmse_vol[s]  for s in scales], color=col_vol,
               lw=2, marker="o", ms=6, label=r"Mol. Vol. ($\AA^3$)")
ax1.axvline(s_opt_vol,  color=col_vol,  ls=":", lw=1.5)
ax1.set_xlabel("Radius scaling factor (s)", fontsize=11)
ax1.set_ylabel(r"RMSE Mol. Vol. ($\AA^3$)", color=col_vol, fontsize=11)
ax1.tick_params(axis="y", labelcolor=col_vol)

ax2 = ax1.twinx()
l2, = ax2.plot(s_arr, [rmse_vbur[s] for s in scales], color=col_vbur,
               lw=2, marker="o", ms=6, label=r"%$V_{bur}$ (%)")
ax2.axvline(s_opt_vbur, color=col_vbur, ls=":", lw=1.5)
ax2.set_ylabel(r"RMSE %$V_{bur}$ (%)", color=col_vbur, fontsize=11)
ax2.tick_params(axis="y", labelcolor=col_vbur)

ax1.annotate(f"s = {s_opt_vol:.2f}", fontsize=8, color=col_vol,
             xy=(s_opt_vol,  rmse_vol[s_opt_vol]),
             xytext=(s_opt_vol  + 0.005, rmse_vol[s_opt_vol]  + (max(rmse_vol.values())  - min(rmse_vol.values()))  * 0.15),
             arrowprops=dict(arrowstyle="->", color=col_vol,  lw=0.8))
ax2.annotate(f"s = {s_opt_vbur:.2f}", fontsize=8, color=col_vbur,
             xy=(s_opt_vbur, rmse_vbur[s_opt_vbur]),
             xytext=(s_opt_vbur + 0.005, rmse_vbur[s_opt_vbur] + (max(rmse_vbur.values()) - min(rmse_vbur.values())) * 0.15),
             arrowprops=dict(arrowstyle="->", color=col_vbur, lw=0.8))
ax1.set_title("(a) RMSE vs Scaling Factor", fontsize=11, fontweight="bold")
ax1.legend(handles=[l1, l2], fontsize=8, loc="upper right")

# Panel (b): mol vol parity at optimal scaling
ax = axes[1]
df_opt_vol = sweep[s_opt_vol]
mol_vol_df = df_opt_vol[["mol_file", "mol_vol"]].drop_duplicates("mol_file").copy()
mol_vol_df["mol_id"] = mol_vol_df["mol_file"].map(mol_id_from_filename)
mol_vol_df = mol_vol_df[mol_vol_df["mol_id"].isin(ref_txt)]
mol_vol_df["iso_vol"] = mol_vol_df["mol_id"].map(lambda i: ref_txt[i][0])
x_v = mol_vol_df["mol_vol"].values
y_v = mol_vol_df["iso_vol"].values

ax.scatter(y_v, x_v, s=20, alpha=0.6, color=col_vol, edgecolors="none", rasterized=True)
all_vals = np.concatenate([x_v, y_v])
lo, hi = all_vals.min() * 0.9, all_vals.max() * 1.1
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
r2, mae, rmse_v, slope, intercept = calc_stats(y_v, x_v)
xfit = np.linspace(lo, hi, 200)
ax.plot(xfit, slope * xfit + intercept, "-", color=col_vol, lw=1.2, alpha=0.8)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
ax.set_xlabel(r"Isodensity Mol. Vol. ($\AA^3$)", fontsize=11)
ax.set_ylabel(fr"Bondi ×{s_opt_vol:.2f} Mol. Vol. ($\AA^3$)", fontsize=11)
ax.set_title(f"(b) Mol. Vol. at s = {s_opt_vol:.2f}", fontsize=11, fontweight="bold")
ax.text(0.05, 0.95,
        f"$R^2$ = {r2:.3f}\nMAE = {mae:.1f} Å³\nRMSE = {rmse_v:.1f} Å³\nn = {len(x_v)}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

# Panel (c): vbur parity at optimal scaling, coloured by element
ax = axes[2]
df_opt = sweep[s_opt_vbur].merge(iso_sampled, on=["mol_file", "atom1_idx", "element"],
                                  suffixes=("_b", "_iso"))
vbur_b   = df_opt["vbur_b"].values
vbur_iso = df_opt["vbur_iso"].values
els      = df_opt["element"].values

for el in ELEMENT_ORDER:
    mask = els == el
    if mask.any():
        ax.scatter(vbur_iso[mask], vbur_b[mask], s=20, alpha=0.65,
                   color=ELEMENT_COLORS[el], edgecolors="none",
                   label=el, zorder=3, rasterized=True)
all_vals = np.concatenate([vbur_iso, vbur_b])
lo, hi = all_vals.min() * 0.9, all_vals.max() * 1.1
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
r2, mae, rmse_v, slope, intercept = calc_stats(vbur_iso, vbur_b)
xfit = np.linspace(lo, hi, 200)
ax.plot(xfit, slope * xfit + intercept, "-", color=col_vbur, lw=1.2, alpha=0.8)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
ax.set_xlabel(r"Isodensity %$V_{bur}$", fontsize=11)
ax.set_ylabel(fr"Bondi ×{s_opt_vbur:.2f} %$V_{{bur}}$", fontsize=11)
ax.set_title(f"(c) %$V_{{bur}}$ at s = {s_opt_vbur:.2f}", fontsize=11, fontweight="bold")
ax.text(0.05, 0.95,
        f"$R^2$ = {r2:.3f}\nMAE = {mae:.2f}%\nRMSE = {rmse_v:.2f}%\nn = {len(vbur_b)}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
legend_handles = [mpatches.Patch(color=ELEMENT_COLORS[el], label=el)
                  for el in ELEMENT_ORDER if el in els]
ax.legend(handles=legend_handles, title="Center atom", fontsize=8,
          title_fontsize=8, loc="lower right")

plt.tight_layout()
for ext in ("png", "pdf"):
    out = HERE + f"optimal_scaling.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
plt.show()
