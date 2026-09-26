# -*- coding: UTF-8 -*-

"""
pymol_plugin

DBSTEP inside PyMOL: measure the structures you have open, pick atoms with PyMOL selections and see
the Sterimol axes, buried-volume spheres, residue contributions and cone angles drawn on the molecule.

Setup (once, in the Python that PyMOL uses): ``pip install dbstep``. Then in PyMOL, or in your
.pymolrc:

	import dbstep.pymol_plugin

(or ``run /path/to/dbstep/pymol_plugin.py``; ``python -m dbstep.pymol_plugin`` prints that path).

Commands (all arguments are PyMOL selections or plain values, ``help dbstep_sterimol`` in PyMOL):

	dbstep_sterimol atom1, atom2 [, atom3, radii=bondi, measure=classic, grid=0.05, scale=1.0, noH=0, selection=, name=sterimol]
	dbstep_vbur     atom1 [, radius=3.5, radii=bondi, scale=1.0, noH=0, selection=, decompose=1, grid=0.05, name=vbur]
	dbstep_cone     metal [, ligand, radii=cpk, scale=1.0, selection=, name=cone]
	dbstep_vdw      object [, radii=bondi, scale=1.0, transparency=0.6]
	dbstep_ensemble files, atom1, atom2 [, radii=bondi, temperature=298.15, window=, tag=, vbur=0, radius=3.5, group=ensemble]
	dbstep_conformers folder [, pattern=*.pdb, group=conformers]
	dbstep_style
"""

import glob
import math
import os
import sys

import numpy as np

try:
	from pymol import cmd
	from pymol.cgo import ALPHA, BEGIN, COLOR, CONE, CYLINDER, END, LINEWIDTH, LINE_LOOP, SPHERE, VERTEX
except ImportError:  # imported outside PyMOL (tests, --help): the helpers still work, nothing is registered
	cmd = None

from dbstep import Dbstep, calculator, ensemble, trajectory
from dbstep import radii as radii_sets
from dbstep.parse_data import Structure

COLORS = {"L": (0.25, 0.45, 1.0), "bmin": (0.15, 0.75, 0.25), "bmax": (0.95, 0.2, 0.2), "sphere": (0.95, 0.45, 0.8), "cone": (1.0, 0.7, 0.15), "axis": (0.4, 0.4, 0.4)}
_ICODE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


class PluginError(Exception):
	pass


def _flag(value):
	return str(value).strip().lower() in ("1", "true", "yes", "on")


def _object_of(selection):
	"""The single PyMOL object a selection belongs to."""
	objects = sorted({obj for obj, _ in cmd.index(selection)})
	if not objects:
		raise PluginError("selection '{}' matches no atoms".format(selection))
	if len(objects) > 1:
		raise PluginError("selection '{}' spans several objects ({}); pick one".format(selection, ", ".join(objects)))
	return objects[0]


def _element(atom):
	symbol = (atom.symbol or "").strip()
	if symbol:
		return symbol[0].upper() + symbol[1:].lower()
	name = atom.name.strip("0123456789 ")
	return name[:1].upper() if name else "X"


def _structure(object_name, selection="", extra=()):
	"""In-memory Structure of `object_name` restricted to `selection` (whole object when empty).

	Atoms of the `extra` selections (atom1, atom2, ...) are always part of the structure; when they
	fall outside `selection` they are kept as zero-radius ghosts, so `selection="not resi 45"` with
	atom1 in residue 45 measures the environment of the residue, like --exclude-self.

	Returns:
		(Structure, list of PyMOL atom indices in structure order)
	"""
	kept = "({})".format(object_name) if not selection else "(({}) and ({}))".format(object_name, selection)
	pieces = [kept] + ["({})".format(e) for e in extra if e]
	model = cmd.get_model(" or ".join(pieces))
	if not model.atom:
		raise PluginError("no atoms in '{}'".format(object_name))
	inside = {a.index for a in cmd.get_model(kept).atom}
	atoms, coords, ids = [], [], []
	meta = {key: [] for key in ("record", "name", "resname", "chain", "resseq", "icode", "element", "resid")}
	for a in model.atom:
		element = _element(a)
		atoms.append(element if a.index in inside else "Bq")
		coords.append(a.coord)
		ids.append(a.index)
		resi = str(a.resi)
		number = resi.rstrip(_ICODE_LETTERS)
		meta["record"].append("HETATM" if a.hetatm else "ATOM")
		meta["name"].append(a.name)
		meta["resname"].append(a.resn)
		meta["chain"].append(a.chain)
		meta["resseq"].append(int(number) if number.lstrip("-").isdigit() else 0)
		meta["icode"].append(resi[len(number):])
		meta["element"].append(element)
		meta["resid"].append("{}:{}".format(a.chain, resi))
	return Structure(atoms, coords, object_name, meta), ids


def _positions(ids, selection):
	"""1-based positions (in structure order) of the atoms of a selection."""
	if not selection:
		return []
	positions = []
	for _, index in cmd.index(selection):
		if index not in ids:
			raise PluginError("atom {} of '{}' is not part of the measured structure".format(index, selection))
		positions.append(ids.index(index) + 1)
	if not positions:
		raise PluginError("selection '{}' matches no atoms".format(selection))
	return positions


def _run(structure, **kwargs):
	"""Run DBSTEP on a Structure, printing its table to the PyMOL console."""
	Dbstep.dbstep._column_header_printed = False
	Dbstep.dbstep._file_col_width = max(20, len(structure.name) + 2)
	try:
		return Dbstep.dbstep(structure, **kwargs)
	except SystemExit as error:  # DBSTEP reports problems through sys.exit; keep PyMOL alive
		raise PluginError(str(error).strip()) from None


def to_original(run):
	"""Function mapping points of DBSTEP's aligned frame (atom1 at the origin, axis along z) back to the
	coordinates of the object the run was made from."""
	rotation, translation = calculator.rigid_transform(run.coords, run.aligned_coords)

	def convert(points):
		return (np.asarray(points, dtype=float).reshape(-1, 3) - translation) @ rotation

	return convert


# --- CGO helpers -----------------------------------------------------------------------------------------


def cgo_cylinder(start, end, radius, color):
	return [CYLINDER, *map(float, start), *map(float, end), float(radius), *color, *color]


def cgo_circle(points, color, width=3.0):
	obj = [LINEWIDTH, float(width), BEGIN, LINE_LOOP, COLOR, *color]
	for point in points:
		obj += [VERTEX, *map(float, point)]
	obj.append(END)
	return obj


def circle_points(centre, e1, e2, radius, n=90):
	angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
	return [np.asarray(centre) + radius * (math.cos(a) * np.asarray(e1) + math.sin(a) * np.asarray(e2)) for a in angles]


def sterimol_cgo(run):
	"""CGO list drawing L (blue), Bmin (green) and Bmax (red) of a Sterimol run on the original coordinates."""
	convert = to_original(run)
	origin, z_end = convert([[0, 0, 0], [0, 0, run.L]])
	e1, e2 = convert([[1, 0, 0], [0, 1, 0]]) - origin
	obj = cgo_cylinder(origin, z_end, 0.08, COLORS["L"])
	vectors = run.sterimol_vectors() or {}
	for key in ("bmax", "bmin"):
		value = run.Bmax if key == "bmax" else run.Bmin
		if value is False:
			continue
		if key in vectors:
			tip = convert([vectors[key]])[0]
			obj += cgo_cylinder(origin, tip, 0.08, COLORS[key])
		obj += cgo_circle(circle_points(origin, e1, e2, float(value)), COLORS[key])
	return obj


def sphere_cgo(centre, radius, color=COLORS["sphere"], alpha=0.35):
	return [ALPHA, float(alpha), COLOR, *color, SPHERE, *map(float, centre), float(radius)]


def cone_cgo(apex, axis, half_angle_deg, length, color=COLORS["cone"], alpha=0.35):
	"""Cone of half angle `half_angle_deg` around `axis` from `apex`, drawn to slant length `length`
	(also for angles beyond 90 degrees, where the rim lies behind the apex)."""
	axis = np.asarray(axis, dtype=float)
	axis = axis / np.linalg.norm(axis)
	half = math.radians(half_angle_deg)
	rim_centre = np.asarray(apex, dtype=float) + length * math.cos(half) * axis
	rim_radius = length * math.sin(half)
	return [ALPHA, float(alpha), CONE, *map(float, apex), *map(float, rim_centre), 0.0, float(rim_radius), *color, *color, 1.0, 1.0]


def _load(name, obj):
	cmd.delete(name)
	cmd.load_cgo(obj, name)


# --- commands -----------------------------------------------------------------------------------------------


def dbstep_style():
	"""Display settings for clean steric figures (white background, soft specular light, orthoscopic view)."""
	cmd.bg_color("white")
	cmd.set("ray_opaque_background", 0)
	cmd.set("specular", 0.25)
	cmd.set("spec_power", 300)
	cmd.set("spec_reflect", 0.5)
	cmd.set("antialias", 1)
	cmd.set("orthoscopic", 1)
	cmd.set("cgo_transparency", 0.0)
	cmd.set("transparency", 0.5)


def dbstep_vdw(object_name, radii="bondi", scale=1.0, transparency=0.6):
	"""Show a translucent van der Waals copy `<object>_vdw` of an object with DBSTEP's radii (bondi, charry-tkatchenko or cpk)."""
	structure, ids = _structure(object_name)
	values = radii_sets.for_atoms(structure.atoms, structure.coords, str(radii).lower()) * float(scale)
	name = "{}_vdw".format(object_name)
	cmd.delete(name)
	cmd.copy(name, object_name)
	cmd.alter(name, "vdw=table.get(index, vdw)", space={"table": {index: float(r) for index, r in zip(ids, values)}})
	cmd.rebuild()
	cmd.hide("everything", name)
	cmd.show("spheres", name)
	cmd.set("sphere_scale", 1.0, name)
	cmd.set("sphere_transparency", float(transparency), name)
	print(" dbstep: {} van der Waals radii ({}) shown as {}".format(len(values), radii_sets.radii_label(str(radii).lower()), name))
	return dict(zip(ids, values))


def dbstep_sterimol(atom1, atom2, atom3="", radii="bondi", measure="classic", grid=0.05, scale=1.0, noH=0, selection="", name="sterimol"):
	"""Sterimol L, Bmin and Bmax from atom1 along the atom1-atom2 axis, drawn on the object as CGO `name`.

	`selection` restricts the measured atoms (default: the whole object of atom1); atom3 fixes the roll.
	"""
	obj = _object_of(atom1)
	structure, ids = _structure(obj, selection, (atom1, atom2, atom3))
	i1, i2 = _positions(ids, atom1)[0], _positions(ids, atom2)
	i3 = _positions(ids, atom3)[0] if atom3 else False
	run = _run(structure, atom1=i1, atom2=i2, atom3=i3, sterimol=True, measure=str(measure), radii=str(radii).lower(), grid=float(grid), scalevdw=float(scale), noH=_flag(noH))
	_load(name, sterimol_cgo(run))
	print(" dbstep: L = {:.2f}, Bmin = {:.2f}, Bmax = {:.2f} Ang ({} radii) -> CGO '{}'".format(run.L, run.Bmin, run.Bmax, radii, name))
	return run


def dbstep_vbur(atom1, radius=3.5, radii="bondi", scale=1.0, noH=0, selection="", decompose=1, grid=0.05, name="vbur"):
	"""Percent buried volume in a sphere of `radius` around atom1, drawn as CGO `name`.

	`selection` restricts the atoms that can fill the sphere (e.g. "polymer and not resi 45" measures the
	pocket around residue 45 without waters and without the residue itself). With `decompose` the
	B-factor of every residue is set to its contribution and the contributing residues are coloured by it.
	"""
	obj = _object_of(atom1)
	structure, ids = _structure(obj, selection, (atom1,))
	i1 = _positions(ids, atom1)[0]
	run = _run(structure, atom1=i1, volume=True, r=float(radius), radii=str(radii).lower(), grid=float(grid), scalevdw=float(scale), noH=_flag(noH), cutoff="auto", decompose=_flag(decompose))
	centre = cmd.get_model(atom1).atom[0].coord
	_load(name, sphere_cgo(centre, float(radius)))
	if run.contributions:
		cmd.alter(obj, "b=0.0")
		for label, value in run.contributions.items():
			chain, resi = label.split()[0].split(":", 1)
			cmd.alter("{} and chain '{}' and resi {}".format(obj, chain, resi), "b={}".format(float(value)))
		cmd.spectrum("b", "white_red", "{} and b > 0".format(obj), minimum=0.0)
		top = sorted(run.contributions.items(), key=lambda item: -item[1])[:5]
		print(" dbstep: %V_bur = {:.2f} (R = {:.2f} Ang); top contributors: {}".format(run.bur_vol, float(radius), ", ".join("{} {:.1f}".format(k, v) for k, v in top)))
	else:
		print(" dbstep: %V_bur = {:.2f} (R = {:.2f} Ang) -> CGO '{}'".format(run.bur_vol, float(radius), name))
	return run


def dbstep_cone(metal, ligand="", radii="cpk", scale=1.0, selection="", name="cone"):
	"""Tolman cone angle of the ligand bound to `metal` through the `ligand` atoms (ring atoms or the donor
	atom; auto-detected when empty), with the ligand's Sterimol parameters from the metal. Draws the cone,
	the metal-centroid distance and the Sterimol axes."""
	obj = _object_of(metal)
	structure, ids = _structure(obj, selection, (metal, ligand))
	i1 = _positions(ids, metal)[0]
	i2 = _positions(ids, ligand) if ligand else None
	kwargs = {"atom2": i2} if i2 else {}
	run = _run(structure, atom1=i1, cone=True, radii=str(radii).lower(), scalevdw=float(scale), **kwargs)
	apex = np.asarray(cmd.get_model(metal).atom[0].coord)
	axis_atoms = [ids[i - 1] for i in run.atom2]
	centroid = np.mean([a.coord for a in cmd.get_model("{} and index {}".format(obj, "+".join(str(i) for i in axis_atoms))).atom], axis=0)
	length = max(run.metal_centroid, run.L if run.L else 0.0)
	_load(name, cone_cgo(apex, centroid - apex, run.cone_angle / 2.0, length) + sterimol_cgo(run))
	cmd.pseudoatom("{}_centroid".format(name), pos=list(map(float, centroid)))
	cmd.delete("{}_distance".format(name))
	cmd.distance("{}_distance".format(name), "{}_centroid".format(name), metal)
	print(" dbstep: cone angle {:.2f} deg, metal-centroid {:.3f} Ang, ligand of {} atoms (L {:.2f}, Bmin {:.2f}, Bmax {:.2f}) -> CGO '{}'".format(
		run.cone_angle, run.metal_centroid, len(run.ligand_atoms), run.L, run.Bmin, run.Bmax, name))
	return run


def _expand(files):
	names = []
	for pattern in str(files).replace(";", ",").split(","):
		pattern = pattern.strip()
		if pattern:
			matches = sorted(glob.glob(os.path.expanduser(pattern)))
			names.extend(matches or [pattern])
	return names


def _pdb_string(atoms, coords):
	lines = []
	for i, (atom, (x, y, z)) in enumerate(zip(atoms, coords), start=1):
		element = "" if atom == "Bq" else atom
		lines.append("HETATM{:5d} {:<4} UNK A   1    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00          {:>2}".format(i % 100000, (element or "X")[:4], x, y, z, element[:2]))
	return "\n".join(lines + ["END"]) + "\n"


def _load_run(name, path, run):
	"""Load a conformer into PyMOL: the file itself when PyMOL reads the format, else the measured coordinates."""
	cmd.delete(name)
	try:
		cmd.load(path, name)
		if cmd.count_atoms(name) == 0:
			raise ValueError("empty")
	except Exception:
		cmd.delete(name)
		cmd.read_pdbstr(_pdb_string(run.atoms, run.coords), name)


def dbstep_ensemble(files, atom1, atom2, radii="bondi", temperature=298.15, window="", tag="", vbur=0, radius=3.5, group="ensemble"):
	"""Boltzmann-weighted Sterimol parameters (and %V_bur with vbur=1) over conformers: one file per
	conformer (QM outputs, energies read from them) or one multi-structure file (SDF with energy fields,
	multi-frame xyz). `files` is a glob or a comma-separated list; atom1/atom2 are 1-based atom indices.
	The conformers are loaded into PyMOL, titled with their population and grouped under `group`."""
	names = _expand(files)
	if not names:
		raise PluginError("no files match '{}'".format(files))
	kwargs = {"atom1": int(atom1), "atom2": [int(a) for a in str(atom2).split("+") if a] or int(atom2), "sterimol": True, "radii": str(radii).lower(), "quiet": True}
	if _flag(vbur):
		kwargs.update(volume=True, r=float(radius))
	if len(names) == 1 and trajectory.count_frames(names[0]) > 1:
		runs = Dbstep.all_frames(names[0], **kwargs)
	else:
		runs = [Dbstep.dbstep(f, **kwargs) for f in names]
	summary = ensemble.boltzmann_average(runs, tag=tag or None, temperature=float(temperature), window=float(window) if str(window).strip() else None, label=group)
	populations = [r.population for r in runs]
	print(" {:<28} {:>10} {:>8} {:>8} {:>8} {:>8}".format("structure", "E_rel", "L", "Bmin", "Bmax", "pop"))
	for r in runs:
		label = r.results[0]["structure"] or r.results[0]["file"]
		print(" {:<28} {:>10.2f} {:>8.2f} {:>8.2f} {:>8.2f} {:>8.3f}{}".format(label, r.energy_rel, r.L, r.Bmin, r.Bmax, r.population, "" if r.in_window else "  (outside window)"))
	row = summary[0]
	print(" {:<28} {:>10} {:>8.2f} {:>8.2f} {:>8.2f}".format("Boltzmann average", "", row["L"], row["bmin"], row["bmax"]))
	# load into PyMOL: one object per file, or one object with states for a multi-structure file
	cmd.delete(group)
	if len(names) == 1 and len(runs) > 1:
		cmd.load(names[0], group)
		for state, r in enumerate(runs, start=1):
			cmd.set_title(group, state, "p = {:.3f}".format(r.population))
	else:
		best = max(populations) or 1.0
		members = []
		for f, r in zip(names, runs):
			member = os.path.basename(f).split(".")[0]
			_load_run(member, f, r)
			cmd.set_title(member, 1, "p = {:.3f}".format(r.population))
			cmd.set("stick_transparency", 0.85 * (1.0 - r.population / best), member)
			members.append(member)
		cmd.group(group, " ".join(members))
	return runs, summary


def dbstep_conformers(folder, pattern="*.pdb", group="conformers"):
	"""Load every structure file of a folder into PyMOL under one group (AddConformers of wSterimol)."""
	files = sorted(glob.glob(os.path.join(os.path.expanduser(folder), pattern)))
	if not files:
		raise PluginError("no files matching {} in {}".format(pattern, folder))
	members = []
	for f in files:
		member = os.path.basename(f).split(".")[0]
		cmd.load(f, member)
		members.append(member)
	cmd.group(group, " ".join(members))
	print(" dbstep: loaded {} structures into group '{}'".format(len(members), group))
	return members


COMMANDS = {
	"dbstep_style": dbstep_style,
	"dbstep_vdw": dbstep_vdw,
	"dbstep_sterimol": dbstep_sterimol,
	"dbstep_vbur": dbstep_vbur,
	"dbstep_cone": dbstep_cone,
	"dbstep_ensemble": dbstep_ensemble,
	"dbstep_conformers": dbstep_conformers,
}


def register():
	"""Register the dbstep_* commands with PyMOL (done on import inside PyMOL)."""
	if cmd is None:
		raise ImportError("PyMOL is not importable from this Python; install dbstep into PyMOL's Python and import there")
	for name, function in COMMANDS.items():
		cmd.extend(name, function)
	cmd.auto_arg[0]["dbstep_vdw"] = [cmd.object_sc, "object", ""]
	cmd.auto_arg[0]["dbstep_conformers"] = [cmd.object_sc, "folder", ""]


def __init_plugin__(app=None):
	"""Entry point for PyMOL's plugin manager."""
	register()


if cmd is not None:
	register()

if __name__ == "__main__":
	print("DBSTEP PyMOL plugin: {}".format(os.path.abspath(__file__)))
	print("In PyMOL (with dbstep installed in its Python): import dbstep.pymol_plugin")
	print("or add to .pymolrc:  run {}".format(os.path.abspath(__file__)))
	sys.exit(0)
