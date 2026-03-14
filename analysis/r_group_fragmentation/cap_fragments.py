"""
Cap terminal BRICS fragments with capping groups.

Reads the fragment distribution CSV, filters to fragments appearing >= 3 times,
replaces the dummy atom (*) with a capping group, deduplicates via canonical
SMILES, and writes the SMILES rooted so the capping group appears first.

Supported capping groups:
  - phenyl (c1ccccc1)
  - tert-butyl (C(C)(C)C)
  - cis-4-tert-butylcyclohexyl  (fragment cis to tBu)
  - trans-4-tert-butylcyclohexyl (fragment trans to tBu)
  - para cyano-phenyl (c1ccc(C#N)cc1)
  - meta cyano-phenyl (c1cc(C#N)ccc1)

Usage:
  python cap_fragments.py --input 250k_rndm_zinc_drugs_clean_3_brics_distribution.csv --min_count 3

Outputs:
  <stem>_capped_phenyl.csv
  <stem>_capped_tbutyl.csv
  <stem>_capped_cis_tbucy.csv
  <stem>_capped_trans_tbucy.csv
"""

import argparse
import os

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, EnumerateStereoisomers
from rdkit.Chem.EnumerateStereoisomers import StereoEnumerationOptions
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

_UNCHARGER = rdMolStandardize.Uncharger()

CAPS = {
    "phenyl": "c1ccccc1",
    "tbutyl": "C(C)(C)C",  # atom 0 is the central C — ReplaceSubstructs bonds fragment here
    "cis_tbucy": "C1CCC(CC1)C(C)(C)C",   # 4-tBu-cyclohexyl, atom 0 = C1
    "trans_tbucy": "C1CCC(CC1)C(C)(C)C",  # same SMILES; stereo assigned after replacement
    "para_cnphenyl": "c1ccc(C#N)cc1",     # para cyano-phenyl, atom 0 = ipso C
    "meta_cnphenyl": "c1cc(C#N)ccc1",     # meta cyano-phenyl, atom 0 = ipso C
}

# Substructure queries for finding cap groups in the product (for rooting)
_CAP_QUERIES = {
    "phenyl": Chem.MolFromSmarts("c1ccccc1"),
    "tbutyl": Chem.MolFromSmarts("[CH3][CX4]([CH3])[CH3]"),  # matches tert-butyl in product
    # 4-tBu-cyclohexyl: CH ring atoms with tBu on one, 4 CH2 between them
    "tbucy": Chem.MolFromSmarts("[CH1]1[CH2][CH2][CH1]([CX4]([CH3])([CH3])[CH3])[CH2][CH2]1"),
    "para_cnphenyl": Chem.MolFromSmarts("c1ccc(C#N)cc1"),
    "meta_cnphenyl": Chem.MolFromSmarts("c1cc(C#N)ccc1"),
}

_DUMMY = Chem.MolFromSmiles("[*]")

def _find_cap_atoms(mol, cap_name):
    """Find the attachment atom, fragment atom, and all cap group heavy atoms.

    For phenyl: the unsubstituted phenyl ring's ipso carbon (1 external neighbor)
                and all 6 ring carbons.
    For tbutyl: the quaternary carbon bonded to 3 methyls and the fragment,
                and all 4 carbons (central + 3 methyls).

    Returns (attach_idx, frag_atom_idx, cap_heavy_indices) where all are 0-based.
      - attach_idx: cap group's attachment atom (ipso C or quaternary C)
      - frag_atom_idx: first heavy atom of the fragment bonded to attach_idx
      - cap_heavy_indices: all heavy atom indices belonging to the cap group
    These heavy-atom indices are stable when hydrogens are added.
    Returns (None, None, None) on failure.
    """
    query_key = "tbucy" if cap_name in ("cis_tbucy", "trans_tbucy") else cap_name
    # para/meta cyano-phenyl use their own SMARTS queries directly
    query = _CAP_QUERIES[query_key]
    matches = mol.GetSubstructMatches(query)
    if not matches:
        return None, None, None, None

    def _frag_neighbor(mol, ipso, cap_set):
        """Find the non-cap heavy-atom neighbor of the ipso atom."""
        for nbr in mol.GetAtomWithIdx(ipso).GetNeighbors():
            if nbr.GetIdx() not in cap_set and nbr.GetAtomicNum() > 1:
                return nbr.GetIdx()
        return None

    if cap_name in ("phenyl", "para_cnphenyl", "meta_cnphenyl"):
        # Find the (cyano-)phenyl ring where exactly 1 atom has an external non-cap neighbor
        for match in matches:
            cap_set = set(match)
            ipso = None
            n_external = 0
            for idx in match:
                for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
                    if nbr.GetIdx() not in cap_set:
                        n_external += 1
                        ipso = idx
                        break
            if n_external == 1 and ipso is not None:
                frag_atom = _frag_neighbor(mol, ipso, cap_set)
                return ipso, frag_atom, sorted(match)
        # Fallback: first match, first atom with external neighbor
        match = matches[0]
        cap_set = set(match)
        for idx in match:
            for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
                if nbr.GetIdx() not in cap_set:
                    frag_atom = _frag_neighbor(mol, idx, cap_set)
                    return idx, frag_atom, sorted(match)
    elif cap_name == "tbutyl":
        # The quaternary carbon (bonded to 3 CH3 + fragment)
        match = matches[0]
        cap_set = set(match)
        for idx in match:
            atom = mol.GetAtomWithIdx(idx)
            if atom.GetDegree() == 4:  # quaternary C
                frag_atom = _frag_neighbor(mol, idx, cap_set)
                return idx, frag_atom, sorted(match)
    elif cap_name in ("cis_tbucy", "trans_tbucy"):
        # 4-tBu-cyclohexyl: C1 (attachment) has a non-cap external neighbor
        for match in matches:
            cap_set = set(match)
            # match[0] is C1 (first [CH1] in SMARTS), match[3] is C4
            # C1 should have exactly 1 external non-cap neighbor (the fragment)
            c1_candidates = []
            for idx in match:
                atom = mol.GetAtomWithIdx(idx)
                if atom.GetTotalNumHs(includeNeighbors=False) == 1:  # CH in ring
                    ext = _frag_neighbor(mol, idx, cap_set)
                    if ext is not None:
                        c1_candidates.append((idx, ext))
            # C1 is the CH with a non-cap, non-tBu external neighbor
            for c1, frag_atom in c1_candidates:
                return c1, frag_atom, sorted(match)
    return None, None, None


def _find_tbucy_ring_centers(mol):
    """Find C1 (fragment-bearing) and C4 (tBu-bearing) in a 4-tBu-cyclohexyl product.

    Returns (ring_set, c1_idx, c4_idx) or (None, None, None) on failure.
    """
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if len(ring) != 6 or not all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in ring):
            continue
        ring_set = set(ring)
        c1 = c4 = None
        for idx in ring:
            atom = mol.GetAtomWithIdx(idx)
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() in ring_set or nbr.GetAtomicNum() <= 1:
                    continue
                # tBu quaternary C: degree 4, bonded to 4 carbons
                if (nbr.GetAtomicNum() == 6 and nbr.GetDegree() == 4 and
                        sum(1 for n in nbr.GetNeighbors() if n.GetAtomicNum() == 6) == 4):
                    c4 = idx
                elif c1 is None:
                    c1 = idx
        if c1 is not None and c4 is not None:
            return ring_set, c1, c4
    return None, None, None


def _assign_tbucy_stereo(mol, cap_name):
    """Assign cis or trans stereochemistry to 4-tBu-cyclohexyl product.

    Uses EnumerateStereoisomers to generate both diastereomers, then selects
    the correct one by comparing ChiralTags at C1 and C4:
      - same ChiralTag   → trans (substituents on opposite faces)
      - different ChiralTag → cis  (substituents on same face)
    """
    ring_set, c1, c4 = _find_tbucy_ring_centers(mol)
    if c1 is None or c4 is None:
        return None

    opts = StereoEnumerationOptions()
    opts.onlyUnassigned = True
    opts.unique = True
    try:
        isomers = list(EnumerateStereoisomers.EnumerateStereoisomers(mol, options=opts))
    except (ValueError, RuntimeError):
        return None

    want_cis = (cap_name == "cis_tbucy")

    for iso in isomers:
        c1_tag = iso.GetAtomWithIdx(c1).GetChiralTag()
        c4_tag = iso.GetAtomWithIdx(c4).GetChiralTag()
        tags_same = (c1_tag == c4_tag)
        # same ChiralTag → trans; different → cis
        is_cis = not tags_same
        if is_cis == want_cis:
            return iso

    # Fallback: return first isomer if only 1 generated (symmetric fragment)
    return isomers[0] if isomers else None


def cap_fragment(frag_smi, cap_name):
    """Replace the dummy atom (*) in a fragment with a capping group.

    Returns (canonical_smiles, attach_atom_idx) rooted at the capping group,
    or (None, None) on failure. attach_atom_idx is the 0-based index of the
    cap group's attachment atom in the canonical SMILES atom ordering — this
    index is stable when hydrogens are added.
    """
    frag = Chem.MolFromSmiles(frag_smi)
    if frag is None:
        return None, None, None, None

    cap = Chem.MolFromSmiles(CAPS[cap_name])
    if cap is None:
        return None, None, None, None

    # ReplaceSubstructs replaces the dummy atom with the cap group
    products = AllChem.ReplaceSubstructs(frag, _DUMMY, cap)
    if not products:
        return None, None, None, None

    mol = products[0]
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None, None, None, None

    # Neutralize net-charged species (e.g. [NH3+]→NH2, [O-]→OH)
    # Leaves internally balanced groups like nitro [N+](=O)[O-] untouched
    mol = _UNCHARGER.uncharge(mol)

    # Drop permanently charged species (quaternary N, pyridinium, etc.)
    if Chem.rdmolops.GetFormalCharge(mol) != 0:
        return None, None, None, None

    # For 4-tBu-cyclohexyl caps, assign cis/trans stereochemistry at C1 and C4
    if cap_name in ("cis_tbucy", "trans_tbucy"):
        mol = _assign_tbucy_stereo(mol, cap_name)
        if mol is None:
            return None, None, None, None

    # Drop molecules with unassigned stereocenters (bridged bicyclics where
    # the cap attachment creates a new undefined stereocenter)
    from rdkit.Chem import FindMolChiralCenters
    centers = FindMolChiralCenters(mol, includeUnassigned=True)
    if any(label == "?" for _, label in centers):
        return None, None, None, None

    # Drop molecules with >= 5 rotatable bonds (keeps conformational search
    # tractable — excludes < 4% of fragments)
    from rdkit.Chem import rdMolDescriptors
    if rdMolDescriptors.CalcNumRotatableBonds(mol) >= 5:
        return None, None, None, None

    # Find attachment atom, fragment atom, and cap group atoms
    attach, frag_atom, cap_heavy = _find_cap_atoms(mol, cap_name)

    # Root the SMILES at a cap-group atom far from the attachment point so the
    # cap writes out fully before the fragment (e.g. "c1ccccc1C" not "Cc1ccccc1")
    if cap_heavy and attach is not None:
        from rdkit.Chem import rdmolops as _rdmolops
        dmat = _rdmolops.GetDistanceMatrix(mol)
        root = max(cap_heavy, key=lambda i: dmat[attach][i])
    else:
        root = 0

    smi = Chem.MolToSmiles(mol, rootedAtAtom=root)

    # Re-parse the canonical SMILES to get indices in the final atom ordering
    # (MolToSmiles reorders atoms)
    final_mol = Chem.MolFromSmiles(smi)
    if final_mol is None:
        return None, None, None, None, None
    final_attach, final_frag_atom, final_cap_heavy = _find_cap_atoms(final_mol, cap_name)

    return smi, final_attach, final_frag_atom, final_cap_heavy


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True,
                        help="Fragment distribution CSV (from brics_fragments_from_csv.py)")
    parser.add_argument("--min_count", type=int, default=3,
                        help="Minimum total_count to include a fragment (default: 3)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: same as input)")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_csv(args.input)
    print(f"Read {len(df):,} unique fragments from {args.input}")

    df = df[df["total_count"] >= args.min_count].reset_index(drop=True)
    print(f"Filtered to {len(df):,} fragments with total_count >= {args.min_count}")

    stem = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(out_dir, exist_ok=True)

    for cap_name, cap_smi in CAPS.items():
        print(f"\nCapping with {cap_name} ({cap_smi})...")

        records = []
        seen = set()
        n_fail = 0

        for _, row in df.iterrows():
            frag_smi = row["fragment_smiles"]
            capped, attach_idx, frag_atom_idx, cap_atoms = cap_fragment(frag_smi, cap_name)
            if capped is None:
                n_fail += 1
                continue
            if capped in seen:
                continue
            seen.add(capped)
            records.append({
                "smiles": capped,
                "attach_atom_idx": attach_idx,
                "frag_atom_idx": frag_atom_idx,
                "cap_atoms": ",".join(str(i) for i in cap_atoms) if cap_atoms else "",
                "fragment_smiles": frag_smi,
                "total_count": row["total_count"],
                "molecule_count": row["molecule_count"],
            })

        out_df = pd.DataFrame(records)
        out_df.insert(0, "name", [f"fragment_{i+1:05d}" for i in range(len(out_df))])
        outpath = os.path.join(out_dir, f"{stem}_capped_{cap_name}.csv")
        out_df.to_csv(outpath, index=False)

        n_input = len(df) - n_fail
        n_dupes = n_input - len(out_df)
        print(f"  Capped successfully : {n_input:,}")
        print(f"  Failed              : {n_fail:,}")
        print(f"  Duplicates removed  : {n_dupes:,}")
        print(f"  Unique molecules    : {len(out_df):,}")
        print(f"  Saved -> {outpath}")

    print("\nDone.")


if __name__ == "__main__":
    main()
