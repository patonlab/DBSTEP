"""Generate ala5.pdb: penta-alanine with all hydrogens plus two waters and a sodium ion (HETATM).

Run with the analysis dependency group (needs RDKit):  uv run --group analysis python tests/pdb_files/make_ala5.py
Not collected by pytest (filename does not start with test_).
"""

import os

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def main():
	mol = Chem.AddHs(Chem.MolFromSequence("AAAAA"))
	AllChem.EmbedMolecule(mol, randomSeed=7)
	AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
	conf = mol.GetConformer()

	# give hydrogens the residue info of the heavy atom they are attached to
	h_per_residue = {}
	for atom in mol.GetAtoms():
		if atom.GetAtomicNum() == 1:
			heavy = atom.GetNeighbors()[0].GetPDBResidueInfo()
			key = (heavy.GetChainId(), heavy.GetResidueNumber())
			h_per_residue[key] = h_per_residue.get(key, 0) + 1
			info = Chem.AtomPDBResidueInfo()
			info.SetResidueName(heavy.GetResidueName())
			info.SetResidueNumber(heavy.GetResidueNumber())
			info.SetChainId(heavy.GetChainId())
			info.SetIsHeteroAtom(False)
			info.SetName(f"H{h_per_residue[key]}".ljust(4))
			atom.SetMonomerInfo(info)

	lines, serial = [], 0
	atoms = sorted(mol.GetAtoms(), key=lambda a: (a.GetPDBResidueInfo().GetResidueNumber(), a.GetAtomicNum() == 1, a.GetIdx()))
	for atom in atoms:
		info, pos = atom.GetPDBResidueInfo(), conf.GetAtomPosition(atom.GetIdx())
		serial += 1
		name = info.GetName()
		if len(name.strip()) < 4 and not name.startswith(" "):
			name = (" " + name.strip()).ljust(4)
		element = atom.GetSymbol().upper()
		lines.append(f"ATOM  {serial:5d} {name:4s} {info.GetResidueName():>3s} {info.GetChainId():1s}{info.GetResidueNumber():4d}    {pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00  0.00          {element:>2s}  ")
	lines.append(f"TER   {serial + 1:5d}      ALA A   5")

	coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
	center = coords.mean(axis=0)
	ca3 = next(a for a in mol.GetAtoms() if a.GetPDBResidueInfo().GetResidueNumber() == 3 and a.GetPDBResidueInfo().GetName().strip() == "CA")
	p_ca3 = np.array(list(conf.GetAtomPosition(ca3.GetIdx())))
	direction = (p_ca3 - center) / np.linalg.norm(p_ca3 - center)

	def away(dist):
		return p_ca3 + dist * direction

	hets = [("HOH", "O", away(3.2), 101), ("HOH", "O", away(9.0), 102), (" NA", "NA", away(4.5) + np.array([0, 2.5, 0]), 201)]
	for resname, element, pos, resseq in hets:
		serial += 1
		name = f"{element:>2s}  " if len(element) == 2 else f" {element:<3s}"
		lines.append(f"HETATM{serial:5d} {name:4s} {resname:>3s} A{resseq:4d}    {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00          {element:>2s}  ")
	lines.append("END")

	out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ala5.pdb")
	with open(out, "w") as f:
		f.write("\n".join(lines) + "\n")
	print(f"wrote {out} ({serial} atoms)")


if __name__ == "__main__":
	main()
