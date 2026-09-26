# -*- coding: UTF-8 -*-
import os
import sys

from dbstep import parse_data

"""
trajectory

Frames of multi-structure inputs: multi-frame XYZ, multi-record SDF and multi-MODEL PDB files.
Each frame is measured by its own dbstep run (with the crop recomputed, since neighbours move),
and --frames selects which frames to run.
"""

# functions returning the structure boundaries of a multi-structure file, by extension
FRAME_COUNTERS = {
	".xyz": parse_data.get_xyz_structures,
	".sdf": parse_data.get_sdf_structures,
	".mol": parse_data.get_sdf_structures,
	".pdb": parse_data.get_pdb_models,
	".ent": parse_data.get_pdb_models,
}


def count_frames(file):
	"""Number of structures (frames) in a file; 1 for single-structure formats."""
	if not isinstance(file, str):
		return 1
	_, ext = os.path.splitext(file)
	if ext not in FRAME_COUNTERS:
		return 1
	return max(1, len(FRAME_COUNTERS[ext](file)))


def parse_frames(spec, n_frames):
	"""Frame indices selected by a --frames value.

	The value uses Python slice semantics on the 0-based frame index: 'start:stop:stride' with any
	part omitted ('::10' every tenth frame, '100:' from frame 100 on, ':50' the first fifty), a
	single index, negative indices from the end, or a comma-separated list of these. A list or
	tuple of integers is accepted from Python. Unset means every frame.

	Returns:
		list of 0-based frame indices in the order given
	"""
	if spec is False or spec is None or (isinstance(spec, str) and spec.strip() == ""):
		return list(range(n_frames))
	tokens = [str(t).strip() for t in spec] if isinstance(spec, (list, tuple)) else [t.strip() for t in str(spec).split(",")]
	indices = []
	for token in tokens:
		if not token:
			continue
		try:
			if ":" in token:
				parts = token.split(":")
				if len(parts) > 3:
					raise ValueError
				indices.extend(range(n_frames)[slice(*[int(p) if p.strip() else None for p in parts])])
			else:
				indices.append(int(token))
		except ValueError:
			sys.exit(f"   Can't read frames '{spec}'. Use start:stop:stride (0-based, Python slice rules), e.g. --frames 0:1000:10")
	selected = []
	for index in indices:
		frame = index + n_frames if index < 0 else index
		if not 0 <= frame < n_frames:
			sys.exit(f"   Frame {index} is out of range (file has {n_frames} frame{'s' if n_frames != 1 else ''})")
		selected.append(frame)
	if not selected:
		sys.exit(f"   --frames {spec} selects no frames")
	return selected


def frame_indices(file, options):
	"""Frames to run for a file: a list of 0-based indices, or [None] for a single-structure file without --frames."""
	n_frames = count_frames(file)
	frames = getattr(options, "frames", False)
	unset = frames is False or frames is None or (isinstance(frames, str) and frames.strip() == "")
	if n_frames == 1 and unset:
		return [None]
	return parse_frames(frames, n_frames)
