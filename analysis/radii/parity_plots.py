import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

HERE = __file__.replace("parity_plots.py", "")

# ── Element colours ──────────────────────────────────────────────────────────
ELEMENT_COLORS = {
    "C":  "#555555",
    "H":  "#AAAAAA",
    "N":  "#4477BB",
    "O":  "#DD4444",
    "S":  "#DDAA22",
    "F":  "#44BB88",
    "Cl": "#BB44BB",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_dbstep_file(filepath):
    """Parse a DBSTEP text output file → dict of {mol_id: (Mol_Vol, %V_Bur)}."""
    data = {}
    with open(filepath) as f:
        for line in f:
            m = re.match(r'\s+mol_(\d+)_\d+\S*\s+\d+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)', line)
            if m:
                data[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return data


def load_sampled(bondi_path, charry_path, sambvca_path, iso_path):
    """Load and inner-join the four sampled CSV files on (mol_file, atom1_idx)."""
    key = ["mol_file", "atom1_idx", "element"]
    b  = pd.read_csv(bondi_path).rename(columns={"mol_vol": "mol_vol_b",  "vbur": "vbur_b"})
    c  = pd.read_csv(charry_path).rename(columns={"mol_vol": "mol_vol_c",  "vbur": "vbur_c"})
    s  = pd.read_csv(sambvca_path).rename(columns={"mol_vol": "mol_vol_s",  "vbur": "vbur_s"})
    iso = pd.read_csv(iso_path).rename(columns={"mol_vol": "mol_vol_iso", "vbur": "vbur_iso"})
    df = b.merge(c, on=key).merge(s, on=key).merge(iso, on=key)
    return df


def calc_stats(x, y):
    slope, intercept, r, _, _ = stats.linregress(x, y)
    r2   = r ** 2
    mae  = np.mean(np.abs(y - x))
    rmse = np.sqrt(np.mean((y - x) ** 2))
    return r2, mae, rmse, slope, intercept


def parity_panel(ax, x, y, color, title, xlabel, ylabel, unit, elements=None):
    """Draw a single parity panel, optionally coloured by element."""
    if elements is not None:
        for el in ELEMENT_COLORS:
            mask = elements == el
            if mask.any():
                ax.scatter(x[mask], y[mask], s=18, alpha=0.65,
                           color=ELEMENT_COLORS[el], edgecolors="none",
                           label=el, zorder=3, rasterized=True)
    else:
        ax.scatter(x, y, s=18, alpha=0.65, color=color,
                   edgecolors="none", zorder=3, rasterized=True)

    all_vals = np.concatenate([x, y])
    lo, hi = all_vals.min() * 0.9, all_vals.max() * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)

    r2, mae, rmse, slope, intercept = calc_stats(x, y)
    xfit = np.linspace(lo, hi, 200)
    ax.plot(xfit, slope * xfit + intercept, "-", color=color, lw=1.2, alpha=0.8)

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold")

    stats_text = f"$R^2$ = {r2:.3f}\nMAE = {mae:.1f} {unit}\nRMSE = {rmse:.1f} {unit}\nn = {len(x)}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
            va="top", bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))


# ── Load data ─────────────────────────────────────────────────────────────────

ref    = parse_dbstep_file(HERE + "tz_isodensity_volumes.txt")
bondi  = parse_dbstep_file(HERE + "bondi_volumes.txt")
charry = parse_dbstep_file(HERE + "charry_volumes.txt")
sambvca = parse_dbstep_file(HERE + "sambvca_volumes.txt")

common_ids = sorted(set(ref) & set(bondi) & set(charry) & set(sambvca))
print(f"Molecular volume comparison: {len(common_ids)} molecules")

ref_vol    = np.array([ref[i][0]     for i in common_ids])
bondi_vol  = np.array([bondi[i][0]   for i in common_ids])
charry_vol = np.array([charry[i][0]  for i in common_ids])
sambvca_vol = np.array([sambvca[i][0] for i in common_ids])

df = load_sampled(
    HERE + "bondi_sampled.csv",
    HERE + "charry_sampled.csv",
    HERE + "sambvca_sampled.csv",
    HERE + "tz_isodensity_sampled.csv",
)
print(f"Buried volume comparison: {len(df)} atom samples")
elements = df["element"]

# ── Figure ────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 0 — molecular volume (coloured uniformly, same as before)
COLORS = {"Bondi": "#2176AE", "Charry": "#E8553A", "SambVca": "#57A773"}
SUBTITLES = {
    "Bondi":   "Bondi (1.0×, +H)",
    "Charry":  "Charry-Tkatchenko (1.0×, +H)",
    "SambVca": "Bondi (1.17×, no H)",
}

for ax, x, y, method in [
    (axes[0, 0], ref_vol,  bondi_vol,   "Bondi"),
    (axes[0, 1], ref_vol,  charry_vol,  "Charry"),
    (axes[0, 2], ref_vol,  sambvca_vol, "SambVca"),
]:
    parity_panel(
        ax, x, y,
        color=COLORS[method],
        title=f"Mol. Volume — {SUBTITLES[method]}",
        xlabel=r"Isodensity Mol. Vol. ($\AA^3$)",
        ylabel=f"{SUBTITLES[method]} ($\\AA^3$)",
        unit=r"$\AA^3$",
    )

# Row 1 — buried volume, coloured by element
for ax, x, y, method in [
    (axes[1, 0], df["vbur_iso"], df["vbur_b"],  "Bondi"),
    (axes[1, 1], df["vbur_iso"], df["vbur_c"],  "Charry"),
    (axes[1, 2], df["vbur_iso"], df["vbur_s"],  "SambVca"),
]:
    parity_panel(
        ax, x.values, y.values,
        color=COLORS[method],
        title=f"%$V_{{bur}}$ — {SUBTITLES[method]}",
        xlabel=r"Isodensity %$V_{bur}$",
        ylabel=f"{SUBTITLES[method]} %$V_{{bur}}$",
        unit="%",
        elements=elements,
    )

# Shared element legend below bottom row
legend_handles = [
    mpatches.Patch(color=ELEMENT_COLORS[el], label=el)
    for el in ELEMENT_COLORS
    if el in df["element"].values
]
fig.legend(handles=legend_handles, title="Center atom", loc="lower center",
           ncol=len(legend_handles), fontsize=9, title_fontsize=9,
           bbox_to_anchor=(0.5, 0.0), frameon=True)

plt.tight_layout(rect=[0, 0.04, 1, 1])

for ext in ("png", "pdf"):
    out = HERE + f"parity_plots.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
plt.show()
