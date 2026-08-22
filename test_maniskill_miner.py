"""Mining ManiSkill shards into runtime assets (step 12).

The strongest check here is that the emitted JSON loads through the real
runtime loaders -- an asset the miner is happy with but load_affordance_set
rejects would only fail at training time.
"""

import json
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scenegraph.adapters.interaction_events import EE_KEY
from scenegraph.core.affordance import load_affordance_set
from scenegraph.core.relation_rules import REQUIRED_BIN_RELATIONS
from scenegraph.core.whitelist import load_whitelist
from scenegraph.tools import build_maniskill_assets as miner

IDENT = [1.0, 0.0, 0.0, 0.0]


def _pose(xyz, quat=IDENT):
    return list(xyz) + list(quat)


def _sample(payload, frame=0):
    return {"frame": frame, "payload": payload}


def _grasp(n=300):
    return [_sample({"force": 1.0, "gripper_width": 0.04,
                     "tcp_pose": _pose([0.0, 0.0, 0.05]),
                     "obj_pose": _pose([0.0, 0.0, 0.0])}) for _ in range(n)]


def _obj_contact(n=300):
    return [_sample({"force": 1.0,
                     "key_a": "actor:cubeA", "key_b": "actor:cubeB",
                     "pose_a": _pose([0.0, 0.0, 0.04]),
                     "pose_b": _pose([0.0, 0.0, 0.0]),
                     "anchor_a_local": [0.0, 0.0, -0.02],
                     "normal_a_local": [0.0, 0.0, -1.0],
                     "anchor_b_local": [0.0, 0.0, 0.02],
                     "normal_b_local": [0.0, 0.0, 1.0]}) for _ in range(n)]


def _support(n=300, jitter=0.0):
    out = []
    rng = np.random.default_rng(0)
    for _ in range(n):
        off = rng.uniform(-jitter, jitter, size=2) if jitter else [0.0, 0.0]
        out.append(_sample({
            "force": 1.0, "key_a": "actor:cubeB", "key_b": "actor:cubeA",
            "pose_a": _pose([0.0, 0.0, 0.0]),
            "pose_b": _pose([float(off[0]), float(off[1]), 0.04]),
        }))
    return out


def _contain(n=300):
    return [_sample({"force": 1.0, "axial": 0.05, "hole_half_width": 0.02,
                     "hole_pose": _pose([0.1, 0.0, 0.1]),
                     "container_pose": _pose([0.0, 0.0, 0.0]),
                     "key_pose": _pose([0.1, 0.0, 0.1]),
                     "containee_pose": _pose([0.05, 0.0, 0.1])})
            for _ in range(n)]


class MinerTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.shards = self.tmp / "evidence"
        self.configs = self.tmp / "configs"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_shard(self, env_id, samples, **kw):
        d = self.shards / env_id
        d.mkdir(parents=True, exist_ok=True)
        payload = {
            "_schema_version": 1, "env_id": env_id, "target": 300,
            "episodes": 300, "samples": samples,
            "seen_counts": {k: len(v) for k, v in samples.items()},
            "presence": {k: 1.0 for k in samples},
            "excluded": kw.get("excluded", {}),
            "symmetry": kw.get("symmetry", {}),
            "capability": kw.get("capability"),
            "bin_stats": kw.get("bin_stats", {
                "planar_distance": 0.8, "height_offset": 0.4,
                "planar_distance_change": 0.05,
                "height_offset_change": 0.03}),
            "complete": list(samples),
            "incomplete": [], "late": {},
        }
        with open(d / f"shard_{kw.get('shard', 0):03d}.pkl", "wb") as f:
            pickle.dump(payload, f)

    def mine(self, env_id):
        miner.main(["--env-id", env_id, "--shards", str(self.shards),
                    "--configs", str(self.configs)])
        aff = json.loads(
            (self.configs / "affordances" / f"{env_id}.json").read_text())
        wl = json.loads((self.configs / "subtask_whitelists" / env_id
                         / "task_all.json").read_text())
        return aff, wl


class AssetShapeTest(MinerTestBase):
    def _stack(self, **kw):
        self.write_shard("StackCube-v1", {
            f"grasp / {EE_KEY} / actor:cubeA": _grasp(),
            f"contact / {EE_KEY} / actor:cubeA": _grasp(),
            "contact / actor:cubeA / actor:cubeB": _obj_contact(),
            "support / actor:cubeB / actor:cubeA": _support(jitter=0.03),
        }, **kw)
        return self.mine("StackCube-v1")

    def test_affordances_load_through_the_runtime_loader(self):
        self._stack()
        path = self.configs / "affordances" / "StackCube-v1.json"
        aff_set = load_affordance_set(str(path))
        self.assertFalse(aff_set.is_empty())

    def test_whitelist_loads_through_the_runtime_loader(self):
        self._stack()
        path = (self.configs / "subtask_whitelists" / "StackCube-v1"
                / "task_all.json")
        wl = load_whitelist(str(path))
        self.assertEqual(wl.subtask, "task")
        self.assertEqual(wl.task_group, "StackCube-v1")
        self.assertTrue(wl.contains("actor:cubeA"))
        self.assertTrue(wl.contains("actor:cubeB"))

    def test_every_required_bin_relation_is_calibrated(self):
        _, wl = self._stack()
        for relation in REQUIRED_BIN_RELATIONS:
            self.assertTrue(wl["bin_edges"].get(relation), relation)

    def test_interaction_types_come_from_evidence(self):
        _, wl = self._stack()
        a = wl["members"]["actor:cubeA"]["interaction_types"]
        self.assertIn("grasp", a)
        self.assertIn("contact", a)
        self.assertIn("support", a)
        self.assertNotIn("contain", a)

    def test_paired_contact_components_are_index_aligned(self):
        aff, _ = self._stack()
        a = aff["objects"]["actor:cubeA"]["contact_components"]
        b = aff["objects"]["actor:cubeB"]["contact_components"]
        self.assertEqual(len(a), len(b))
        self.assertEqual(len(a), 300)

    def test_support_footprint_carries_the_spread(self):
        aff, _ = self._stack()
        comp = aff["objects"]["actor:cubeB"]["support_components"][0]
        self.assertGreater(comp["footprint_radius"], 0.01)
        self.assertEqual(comp["surface_normal"], [0.0, 0.0, 1.0])

    def test_supporter_gets_surface_and_supported_gets_bottom(self):
        aff, _ = self._stack()
        self.assertIn("support_components", aff["objects"]["actor:cubeB"])
        self.assertIn("bottom_components", aff["objects"]["actor:cubeA"])


class ExclusionTest(MinerTestBase):
    def test_incomplete_buckets_never_reach_the_asset(self):
        self.write_shard("PickCube-v1", {
            f"grasp / {EE_KEY} / actor:cube": _grasp(),
            f"contact / {EE_KEY} / actor:brush": _grasp(n=4),
        })
        _, wl = self.mine("PickCube-v1")
        self.assertIn("actor:cube", wl["members"])
        self.assertNotIn("actor:brush", wl["members"])

    def test_incidental_buckets_never_reach_the_asset(self):
        brush = f"contact / {EE_KEY} / actor:cubeB"
        self.write_shard("StackCube-v1", {
            f"grasp / {EE_KEY} / actor:cubeA": _grasp(),
            brush: _grasp(),
        }, excluded={brush: 0.04})
        _, wl = self.mine("StackCube-v1")
        self.assertNotIn("actor:cubeB", wl["members"])

    def test_stale_shard_without_bin_stats_is_refused(self):
        self.write_shard("PickCube-v1", {
            f"grasp / {EE_KEY} / actor:cube": _grasp()}, bin_stats={})
        with self.assertRaises(SystemExit):
            self.mine("PickCube-v1")


class OrientationTest(MinerTestBase):
    def test_dominant_orientation_wins_over_force_sign_noise(self):
        self.write_shard("PegInsertionSide-v1", {
            f"grasp / {EE_KEY} / actor:peg": _grasp(),
            "support / actor:box / actor:peg": _support(300),
            "support / actor:peg / actor:box": _support(300)[:5],
        })
        aff, _ = self.mine("PegInsertionSide-v1")
        self.assertIn("support_components", aff["objects"]["actor:box"])
        self.assertNotIn("support_components",
                         aff["objects"].get("actor:peg", {}))

    def test_a_genuinely_close_split_raises(self):
        self.write_shard("Ambiguous-v1", {
            f"grasp / {EE_KEY} / actor:x": _grasp(),
            "support / actor:x / actor:y": _support(300),
            "support / actor:y / actor:x": _support(300)[:280],
        })
        with self.assertRaises(SystemExit):
            self.mine("Ambiguous-v1")


class SphereTest(MinerTestBase):
    def test_spherical_grasp_is_radial_not_pose_based(self):
        self.write_shard("PlaceSphere-v1", {
            f"grasp / {EE_KEY} / actor:sphere": _grasp(),
        }, symmetry={"actor:sphere": {"symmetry": "spherical",
                                      "radius": 0.02,
                                      "orientation_invariant": True}})
        aff, _ = self.mine("PlaceSphere-v1")
        comps = aff["objects"]["actor:sphere"]["grasp_components"]
        self.assertEqual(len(comps), 1)
        self.assertTrue(comps[0]["orientation_invariant"])
        self.assertAlmostEqual(comps[0]["radial_offset"], 0.05, places=6)

    def test_non_spherical_keeps_one_component_per_sample(self):
        self.write_shard("PickCube-v1", {
            f"grasp / {EE_KEY} / actor:cube": _grasp()})
        aff, _ = self.mine("PickCube-v1")
        self.assertEqual(
            len(aff["objects"]["actor:cube"]["grasp_components"]), 300)


class ContainTest(MinerTestBase):
    def test_container_and_containee_components(self):
        self.write_shard("PegInsertionSide-v1", {
            f"grasp / {EE_KEY} / actor:peg": _grasp(),
            "contain / actor:box / actor:peg": _contain(),
        }, capability="peg_hole")
        aff, wl = self.mine("PegInsertionSide-v1")
        entry = aff["objects"]["actor:box"]["contain_components"][0]
        self.assertAlmostEqual(entry["opening_radius"], 0.02)
        self.assertIn("key_components", aff["objects"]["actor:peg"])
        self.assertIn("contain",
                      wl["members"]["actor:peg"]["interaction_types"])


class MergeTest(MinerTestBase):
    def test_shards_merge_and_maxima_take_the_max(self):
        key = f"grasp / {EE_KEY} / actor:cube"
        self.write_shard("PickCube-v1", {key: _grasp(150)}, shard=0,
                         bin_stats={"planar_distance": 0.5,
                                    "height_offset": 0.2})
        self.write_shard("PickCube-v1", {key: _grasp(150)}, shard=1,
                         bin_stats={"planar_distance": 0.9,
                                    "height_offset": 0.1})
        merged = miner.load_shards("PickCube-v1", self.shards)
        self.assertEqual(len(merged["samples"][key]), 300)
        self.assertEqual(merged["episodes"], 600)
        self.assertAlmostEqual(merged["bin_stats"]["planar_distance"], 0.9)
        self.assertAlmostEqual(merged["bin_stats"]["height_offset"], 0.2)


if __name__ == "__main__":
    unittest.main()
