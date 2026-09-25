"""Generate ala5_traj.pdb: 10 MODELs derived from ala5.pdb with a known trend.

Frame k (0-9): the whole system is translated by 0.7*k Angstrom along x (results must be
translation invariant), and the first water (HOH 101) is moved from 4.8 to 3.0 Angstrom away from
CA of residue 3 in steps of 0.2 Angstrom, so %V_bur around A:3 rises monotonically with the frame
index unless --nowater is used. Needs only numpy:  uv run python tests/pdb_files/make_ala5_traj.py
Not collected by pytest (filename does not start with test_).
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
	lines = [line for line in open(os.path.join(HERE, "ala5.pdb")).read().splitlines() if line.startswith(("ATOM", "HETATM"))]
	coords = np.array([[float(line[30:38]), float(line[38:46]), float(line[46:54])] for line in lines])
	names = [line[12:16].strip() for line in lines]
	resseq = [int(line[22:26]) for line in lines]
	ca3 = next(i for i, (n, r) in enumerate(zip(names, resseq)) if n == "CA" and r == 3)
	water = next(i for i, (n, r) in enumerate(zip(names, resseq)) if n == "O" and r == 101)
	protein = np.array([line.startswith("ATOM") for line in lines])
	center = coords[protein].mean(axis=0)
	direction = coords[ca3] - center
	direction /= np.linalg.norm(direction)

	out = []
	for k in range(10):
		frame = coords.copy()
		frame[water] = coords[ca3] + (4.8 - 0.2 * k) * direction
		frame[:, 0] += 0.7 * k
		out.append(f"MODEL     {k + 1:4d}")
		for line, (x, y, z) in zip(lines, frame):
			out.append(line[:30] + f"{x:8.3f}{y:8.3f}{z:8.3f}" + line[54:])
		out.append("ENDMDL")
	out.append("END")
	path = os.path.join(HERE, "ala5_traj.pdb")
	with open(path, "w") as f:
		f.write("\n".join(out) + "\n")
	print(f"wrote {path} (10 models)")


if __name__ == "__main__":
	main()
