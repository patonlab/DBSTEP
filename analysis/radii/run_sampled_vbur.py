"""
Run buried volume calculations for all (mol_file, atom1_idx) pairs in
sample_atoms.csv and write results to CSV files.

Modes:
  (default)      Three VDW conditions:
                   Bondi 1.0x +H  → bondi_sampled.csv
                   Charry 1.0x +H → charry_sampled.csv
                   Bondi 1.17x -H → sambvca_sampled.csv

  --isodensity   Isodensity reference (cube files on server):
                   set CUBE_DIR below, then copy script + sample_atoms.csv
                   to server and run with --isodensity
                   → isodensity_sampled.csv

  --sweep        Bondi +H at scaling factors 1.05–1.10 in steps of 0.01
                   → bondi_x1.05_sampled.csv … bondi_x1.10_sampled.csv

Usage:
    python analysis/radii/run_sampled_vbur.py
    python analysis/radii/run_sampled_vbur.py --isodensity
    python analysis/radii/run_sampled_vbur.py --sweep
"""

import os
import csv
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dbstep.Dbstep import dbstep

HERE = os.path.dirname(os.path.abspath(__file__))
XYZ_DIR = os.path.join(HERE, "xyz_files")
SAMPLE_CSV = os.path.join(HERE, "sample_atoms.csv")

# Set this to the directory containing cube files on the server
CUBE_DIR = os.path.join(HERE, "cube_files")

# Scaling factors for the sweep
SWEEP_SCALES = [1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.20]

VDW_CONDITIONS = [
    {
        "name": "Bondi",
        "output": os.path.join(HERE, "bondi_sampled.csv"),
        "file_dir": XYZ_DIR,
        "file_fn": lambda stem: stem + ".xyz",
        "kwargs": {"radii": "bondi", "scalevdw": 1.0, "noH": False},
    },
    {
        "name": "Charry-Tkatchenko",
        "output": os.path.join(HERE, "charry_sampled.csv"),
        "file_dir": XYZ_DIR,
        "file_fn": lambda stem: stem + ".xyz",
        "kwargs": {"radii": "charry-tkatchenko", "scalevdw": 1.0, "noH": False},
    },
    {
        "name": "SambVca (Bondi x1.17, no H)",
        "output": os.path.join(HERE, "sambvca_sampled.csv"),
        "file_dir": XYZ_DIR,
        "file_fn": lambda stem: stem + ".xyz",
        "kwargs": {"radii": "bondi", "scalevdw": 1.17, "noH": True},
    },
]

ISODENSITY_CONDITION = {
    "name": "Isodensity (reference)",
    "output": os.path.join(HERE, "isodensity_sampled.csv"),
    "file_dir": CUBE_DIR,
    "file_fn": lambda stem: stem + "_medium.cube",
    "kwargs": {"surface": "density", "isoval": 0.0016},
}


def sweep_conditions():
    return [
        {
            "name": f"Bondi x{s:.2f} +H",
            "output": os.path.join(HERE, f"bondi_x{s:.2f}_sampled.csv"),
            "file_dir": XYZ_DIR,
            "file_fn": lambda stem: stem + ".xyz",
            "kwargs": {"radii": "bondi", "scalevdw": s, "noH": False},
        }
        for s in SWEEP_SCALES
    ]


def xyz_stem(mol_file):
    """Return the bare stem from a mol_file entry, e.g. 'mol_0002_0001'."""
    return os.path.splitext(mol_file)[0]


def read_samples(csv_path):
    samples = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            samples.append((row["mol_file"], int(row["atom1_idx"]), row["element"]))
    return samples


def run_condition(samples, condition):
    name = condition["name"]
    output_path = condition["output"]
    file_dir = condition["file_dir"]
    file_fn = condition["file_fn"]
    kwargs = condition["kwargs"]

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Output:  {output_path}")
    print(f"{'='*60}")

    rows = []
    errors = []
    for i, (mol_file, atom1_idx, element) in enumerate(samples, 1):
        input_path = os.path.join(file_dir, file_fn(xyz_stem(mol_file)))
        if not os.path.exists(input_path):
            errors.append(f"Missing: {input_path}")
            continue

        try:
            result = dbstep(input_path, volume=True, atom1=atom1_idx, quiet=True, **kwargs)
            rows.append({
                "mol_file": mol_file,
                "atom1_idx": atom1_idx,
                "element": element,
                "mol_vol": round(result.occ_vol, 3),
                "vbur": round(result.bur_vol, 3),
            })
        except Exception as e:
            errors.append(f"{mol_file} atom={atom1_idx}: {e}")

        if i % 100 == 0:
            print(f"  {i}/{len(samples)} done ...")

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["mol_file", "atom1_idx", "element", "mol_vol", "vbur"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Wrote {len(rows)} rows to {output_path}")
    if errors:
        print(f"  {len(errors)} errors:")
        for e in errors[:10]:
            print(f"    {e}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")


def main():
    isodensity = "--isodensity" in sys.argv
    sweep      = "--sweep" in sys.argv

    samples = read_samples(SAMPLE_CSV)
    print(f"Loaded {len(samples)} samples from {SAMPLE_CSV}")

    if isodensity:
        run_condition(samples, ISODENSITY_CONDITION)
    elif sweep:
        for condition in sweep_conditions():
            if os.path.exists(condition["output"]):
                print(f"  Skipping {condition['name']} — {condition['output']} already exists")
                continue
            run_condition(samples, condition)
    else:
        for condition in VDW_CONDITIONS:
            run_condition(samples, condition)

    print("\nDone.")


if __name__ == "__main__":
    main()
