"""The PyMOL plugin, run against headless open-source PyMOL (skipped when PyMOL is not installed)."""

import numpy as np
import pytest

pymol = pytest.importorskip("pymol")
from pymol import cmd  # noqa: E402

from dbstep import Dbstep, pymol_plugin as plugin  # noqa: E402

RH = "tests/metal_files/RhCpMe5Cl2PMe3.xyz"
ALA5 = "tests/pdb_files/ala5.pdb"
ETHER = "tests/sdf_files/ether_conformers.sdf"


@pytest.fixture(autouse=True)
def fresh_session():
	cmd.reinitialize()
	yield
	cmd.reinitialize()


def test_commands_are_registered():
	for name in plugin.COMMANDS:
		assert name in cmd.keyword


def test_sterimol_matches_cli_and_draws_on_the_object():
	cmd.load("dbstep/data/Et.xyz", "et")
	run = plugin.dbstep_sterimol("et and id 2", "et and id 5")
	ref = Dbstep.dbstep("dbstep/data/Et.xyz", atom1=2, atom2=5, sterimol=True, quiet=True)
	assert (run.L, run.Bmin, run.Bmax) == pytest.approx((ref.L, ref.Bmin, ref.Bmax))
	assert "sterimol" in cmd.get_names("all")
	# the L cylinder starts at atom1 and points towards atom2 in the object's own frame
	obj = plugin.sterimol_cgo(run)
	start, end = np.array(obj[1:4]), np.array(obj[4:7])
	xyz = cmd.get_coords("et")
	assert np.allclose(start, xyz[1], atol=1e-5)
	direction = xyz[4] - xyz[1]
	assert np.dot(end - start, direction / np.linalg.norm(direction)) == pytest.approx(run.L, abs=1e-5)
	# grid mode and a restricting selection also work
	grid = plugin.dbstep_sterimol("et and id 2", "et and id 5", measure="grid", grid=0.1, selection="not id 1", name="grid_sterimol")
	assert grid.n_atoms_total == 7 and "grid_sterimol" in cmd.get_names("all")


def test_vbur_matches_residue_mode_and_colours_contributions():
	cmd.load(ALA5, "ala")
	run = plugin.dbstep_vbur("/ala//A/3/CA", grid=0.1)
	ref = Dbstep.dbstep(ALA5, residue="A:3", volume=True, decompose=True, grid=0.1, quiet=True)
	assert run.bur_vol == pytest.approx(ref.bur_vol, abs=1e-9)
	assert run.contributions == pytest.approx(ref.contributions, abs=1e-9)
	assert cmd.get_model("ala and resi 3 and name CA").atom[0].b == pytest.approx(ref.contributions["A:3 ALA"])
	assert cmd.get_model("ala and resi 101").atom[0].b == pytest.approx(ref.contributions["A:101 HOH"])
	assert "vbur" in cmd.get_names("all")
	# excluding the residue through the selection reproduces --exclude-self, waters through "polymer"
	env = plugin.dbstep_vbur("/ala//A/3/CA", grid=0.1, selection="polymer and not resi 3", decompose=0)
	ref_env = Dbstep.dbstep(ALA5, residue="A:3", volume=True, exclude_self=True, nowater=True, nohet=True, grid=0.1, quiet=True)
	assert env.bur_vol == pytest.approx(ref_env.bur_vol, abs=1e-9)
	assert list(env.atoms).count("Bq") == 1


def test_cone_reproduces_the_legacy_value():
	cmd.load(RH, "rh")
	run = plugin.dbstep_cone("rh and elem Rh")
	assert run.cone_angle == pytest.approx(173.97, abs=0.01) and run.metal_centroid == pytest.approx(1.833, abs=0.001)
	assert {"cone", "cone_centroid", "cone_distance"} <= set(cmd.get_names("all"))
	assert np.allclose(cmd.get_coords("cone_centroid")[0], cmd.get_coords("rh and index 3+4+5+24+25").mean(axis=0), atol=1e-4)
	phosphine = plugin.dbstep_cone("rh and elem Rh", "rh and elem P", name="pme3")
	assert 110 < phosphine.cone_angle < 120 and len(phosphine.ligand_atoms) == 13
	obj = plugin.cone_cgo([0, 0, 0], [0, 0, 1], 60.0, 2.0)
	assert obj[6:9] == pytest.approx([0, 0, 1.0]) and obj[10] == pytest.approx(2.0 * np.sin(np.radians(60))) and obj[14:17] == pytest.approx([1.0, 0.7, 0.15])


def test_vdw_copy_carries_cpk_radii():
	cmd.load(RH, "rh")
	table = plugin.dbstep_vdw("rh", radii="cpk")
	assert "rh_vdw" in cmd.get_object_list() and cmd.count_atoms("rh_vdw") == 41
	radii = {a.index: a.vdw for a in cmd.get_model("rh_vdw").atom}
	assert radii[6] == pytest.approx(1.50, abs=1e-3)  # methyl carbon (sp3)
	assert radii[7] == pytest.approx(1.00, abs=1e-3) and table[7] == pytest.approx(1.0)  # H
	assert radii[2] == pytest.approx(1.80, abs=1e-3) and radii[17] == pytest.approx(1.40, abs=1e-3)  # Cl, P
	assert radii[1] == pytest.approx(2.0, abs=1e-3)  # Rh has no CPK type: Bondi fallback
	plugin.dbstep_vdw("rh", radii="bondi", scale=1.17)
	assert {a.index: a.vdw for a in cmd.get_model("rh_vdw").atom}[7] == pytest.approx(1.09 * 1.17, abs=1e-3)


def test_ensemble_over_files_and_over_a_multi_structure_file():
	runs, summary = plugin.dbstep_ensemble("tests/qm_files/pentane_*.out.gz", 1, 3, radii="cpk", temperature=298, window=1.0)
	assert len(runs) == 9 and summary[0]["L"] + 0.4 == pytest.approx(6.34, abs=0.015)
	assert sum(not r.in_window for r in runs) == 2
	assert "ensemble" in cmd.get_names("group_objects") and cmd.count_atoms("pentane_18") == 17
	assert cmd.get_title("pentane_18", 1).startswith("p = 0.27")
	runs, summary = plugin.dbstep_ensemble(ETHER, 3, 2, group="ether")
	assert len(runs) == 3 and cmd.count_states("ether") == 3 and cmd.get_title("ether", 1).startswith("p = 0.92")


def test_conformers_and_style_and_errors(tmp_path):
	for i in range(3):
		(tmp_path / f"c{i}.xyz").write_text(open("dbstep/data/Et.xyz").read())
	members = plugin.dbstep_conformers(str(tmp_path), pattern="*.xyz")
	assert members == ["c0", "c1", "c2"] and "conformers" in cmd.get_names("group_objects")
	plugin.dbstep_style()
	assert cmd.get_setting_int("orthoscopic") == 1
	with pytest.raises(plugin.PluginError, match="matches no atoms"):
		plugin.dbstep_sterimol("c0 and id 99", "c0 and id 2")
	with pytest.raises(plugin.PluginError, match="must not include"):
		plugin.dbstep_cone("c0 and id 1", "c0 and id 1")
	cmd.load("dbstep/data/Me.xyz", "me")
	with pytest.raises(plugin.PluginError, match="several objects"):
		plugin.dbstep_sterimol("c0 or me", "c0 and id 2")
