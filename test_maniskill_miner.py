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
            "_schema_version": kw.get("schema", 3), "env_id": env_id,
            "target": 300,
            "episodes": kw.get("episodes", 300), "samples": samples,
            "seen_counts": {k: len(v) for k, v in samples.items()},
            "presence": {k: 1.0 for k in samples},
            "episode_presence": kw.get(
                "episode_presence", {k: kw.get("episodes", 300) for k in samples}),
            "traces": kw.get("traces", []),
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


class FinalPresenceTest(MinerTestBase):
    """A bucket can pass the freeze gate and drift below it by the end."""

    def _drifted(self):
        drifter = "support / actor:cube / actor:tool"
        self.write_shard("PullCubeTool-v1", {
            f"grasp / {EE_KEY} / actor:tool": _grasp(),
            drifter: _support(),
        })
        # Freeze kept it; the full run says 16%. Expressed as a count, which
        # is what the shard carries -- the rate is recomputed on merge.
        path = next((self.shards / "PullCubeTool-v1").glob("*.pkl"))
        with open(path, "rb") as f:
            shard = pickle.load(f)
        shard["episode_presence"][drifter] = 48      # of 300
        shard["presence"][drifter] = 0.16
        with open(path, "wb") as f:
            pickle.dump(shard, f)
        return drifter

    def test_drifted_bucket_is_dropped_by_the_miner(self):
        self._drifted()
        _, wl = self.mine("PullCubeTool-v1")
        self.assertNotIn("actor:cube", wl["members"])

    def test_it_would_have_been_mined_without_the_recheck(self):
        drifter = self._drifted()
        merged = miner.load_shards("PullCubeTool-v1", self.shards)
        self.assertIn(drifter, miner.usable_buckets(merged, 300, 0.0))
        self.assertNotIn(drifter, miner.usable_buckets(merged, 300, 0.2))


class SilentLossTest(MinerTestBase):
    """A bucket with evidence but no components means a payload field was
    lost upstream. Emitting anyway yields a runtime that scores the relation
    unobserved forever."""

    def test_contact_without_anchors_is_refused(self):
        stripped = [_sample({k: v for k, v in s["payload"].items()
                             if not k.startswith("anchor_")})
                    for s in _obj_contact()]
        self.write_shard("StackCube-v1", {
            f"grasp / {EE_KEY} / actor:cubeA": _grasp(),
            "contact / actor:cubeA / actor:cubeB": stripped,
        })
        with self.assertRaises(SystemExit):
            self.mine("StackCube-v1")

    def test_support_without_endpoint_keys_is_refused(self):
        stripped = [_sample({k: v for k, v in s["payload"].items()
                             if k not in ("key_a", "key_b")})
                    for s in _support()]
        self.write_shard("StackCube-v1", {
            f"grasp / {EE_KEY} / actor:cubeA": _grasp(),
            "support / actor:cubeB / actor:cubeA": stripped,
        })
        with self.assertRaises(SystemExit):
            self.mine("StackCube-v1")

    def test_contain_without_hole_pose_is_refused(self):
        stripped = [_sample({k: v for k, v in s["payload"].items()
                             if k != "hole_pose"}) for s in _contain()]
        self.write_shard("PegInsertionSide-v1", {
            f"grasp / {EE_KEY} / actor:peg": _grasp(),
            "contain / actor:box / actor:peg": stripped,
        })
        with self.assertRaises(SystemExit):
            self.mine("PegInsertionSide-v1")


class PairedComparisonTest(MinerTestBase):
    """An object touching two things must not match one pair's anchors
    against the other's."""

    def test_components_carry_their_partner(self):
        self.write_shard("StackCube-v1", {
            f"grasp / {EE_KEY} / actor:cubeA": _grasp(),
            "contact / actor:cubeA / actor:cubeB": _obj_contact(),
        })
        aff, _ = self.mine("StackCube-v1")
        a = aff["objects"]["actor:cubeA"]["contact_components"]
        self.assertTrue(all(c["partner"] == "actor:cubeB" for c in a))
        b = aff["objects"]["actor:cubeB"]["contact_components"]
        self.assertTrue(all(c["partner"] == "actor:cubeA" for c in b))


class QuantileBinTest(MinerTestBase):
    """Equal-width bins over a max collapse a bimodal distribution; the table
    origin sits ~0.9m below its own surface, so heights are bimodal."""

    def _bimodal(self):
        import random
        rng = random.Random(0)
        # Half the pairs are object-object (near zero), half object-table.
        return [rng.gauss(0.03, 0.01) for _ in range(500)] + \
               [rng.gauss(0.92, 0.02) for _ in range(500)]

    def _mine_with(self, samples, height_max=0.4):
        self.write_shard("StackCube-v1", {
            f"grasp / {EE_KEY} / actor:cubeA": _grasp()},
            bin_stats={"planar_distance": 0.8, "height_offset": height_max,
                       "planar_distance_change": 0.05,
                       "height_offset_change": 0.03})
        path = next((self.shards / "StackCube-v1").glob("*.pkl"))
        with open(path, "rb") as f:
            shard = pickle.load(f)
        shard["bin_samples"] = samples
        with open(path, "wb") as f:
            pickle.dump(shard, f)
        return self.mine("StackCube-v1")[1]["bin_edges"]

    def _labels(self, edges, values):
        from scenegraph.core.relation_rules import SPATIAL_LABELS, bin_label
        labels = SPATIAL_LABELS["height-offset"]
        return [bin_label(v, edges, labels) for v in values]

    def test_quantile_edges_resolve_within_the_object_mode(self):
        import numpy as np
        edges = self._mine_with(
            {"height_offset": np.asarray(self._bimodal(), dtype=np.float32)}
        )["height-offset"]
        # A stacked cube (0.02) and an adjacent one (0.05) must differ.
        stacked, adjacent = self._labels(edges, [0.02, 0.05])
        self.assertNotEqual(stacked, adjacent)

    def test_max_derived_edges_collapse_the_object_mode(self):
        """Why this needed fixing: StackCube measured height_offset max 1.134,
        because the table origin sits 0.92m below its own surface."""
        edges = self._mine_with({}, height_max=1.134)["height-offset"]
        stacked, adjacent = self._labels(edges, [0.02, 0.05])
        self.assertEqual(stacked, adjacent)
        self.assertEqual(stacked, "level")

    def test_degenerate_statistic_keeps_the_max_scale(self):
        import numpy as np
        flat = np.zeros(500, dtype=np.float32)
        edges = self._mine_with({"height_offset": flat})["height-offset"]
        self.assertGreater(max(edges), 0.0)

    def test_too_few_samples_keeps_the_max_scale(self):
        """A thin reservoir gives unstable quantiles, so fall back."""
        import numpy as np
        tiny = np.asarray(self._bimodal()[:50], dtype=np.float32)
        got = self._mine_with({"height_offset": tiny})["height-offset"]
        self.assertEqual(got, self._mine_with({})["height-offset"])

    def test_reservoirs_concatenate_across_shards(self):
        import numpy as np
        from scenegraph.adapters.interaction_events import BinStats
        b = BinStats()
        for _ in range(3):
            b.observe({"a": [0, 0, 0, 1, 0, 0, 0],
                       "b": [1, 0, 0.5, 1, 0, 0, 0]}, 0)
        self.assertEqual(len(b.reservoir()["height_offset"]), 3)

    def test_missing_quantiles_fall_back(self):
        edges = self._mine_with({})
        self.assertTrue(edges["planar-distance"])
        self.assertTrue(edges["height-offset"])


class ShardMergeTest(MinerTestBase):
    """Cross-shard merging. Bites the moment collection is sharded."""

    def test_old_shards_are_rejected_not_silently_mined(self):
        self.write_shard("PickCube-v1", {f"grasp / {EE_KEY} / actor:cube": _grasp()},
                         schema=2)
        with self.assertRaises(SystemExit) as cm:
            miner.load_shards("PickCube-v1", self.shards)
        self.assertIn("Re-collect", str(cm.exception))

    def test_traces_stay_one_entry_per_episode(self):
        a = [(("grasp", "ee", "actor:cube", 5, 20),)]
        b = [(("contact", "ee", "actor:cube", 1, 4),
              ("grasp", "ee", "actor:cube", 5, 20))]
        key = f"grasp / {EE_KEY} / actor:cube"
        self.write_shard("PickCube-v1", {key: _grasp()}, shard=0, traces=a)
        self.write_shard("PickCube-v1", {key: _grasp()}, shard=1, traces=b)
        merged = miner.load_shards("PickCube-v1", self.shards)
        self.assertEqual(len(merged["traces"]), 2)
        self.assertEqual(merged["traces"][1], b[0])

    def test_presence_merges_by_count_not_last_shard(self):
        """Two shards, present in one of them. The rate is 50%, not whatever
        the last shard happened to write."""
        key = f"grasp / {EE_KEY} / actor:cube"
        rare = "support / actor:cube / actor:tool"
        self.write_shard("PullCubeTool-v1", {key: _grasp(), rare: _support()},
                         shard=0, episodes=100,
                         episode_presence={key: 100, rare: 100})
        self.write_shard("PullCubeTool-v1", {key: _grasp(), rare: _support()},
                         shard=1, episodes=100,
                         episode_presence={key: 100, rare: 0})
        merged = miner.load_shards("PullCubeTool-v1", self.shards)
        self.assertEqual(merged["episodes"], 200)
        self.assertAlmostEqual(merged["presence"][rare], 0.5)
        self.assertAlmostEqual(merged["presence"][key], 1.0)
        # 50% clears a 20% gate and fails a 60% one. Last-shard-wins would
        # have read 0% and dropped it from both.
        self.assertIn(rare, miner.usable_buckets(merged, 300, 0.2))
        self.assertNotIn(rare, miner.usable_buckets(merged, 300, 0.6))


class EpisodeAlignmentTest(MinerTestBase):
    def test_every_stream_keeps_its_episode_index(self):
        """Interactions, predicates and scalars must stay on the same row --
        a phase mined from episode k is checked against episode k's predicates.
        """
        key = f"grasp / {EE_KEY} / actor:cube"
        ep0 = {"interactions": (("grasp", "ee", "actor:cube", 5, 20),),
               "predicates": {"is_grasped": ((5, 20),)},
               "scalars": {}, "kinds": {"is_grasped": "predicate"}, "frames": 30}
        ep1 = {"interactions": (("grasp", "ee", "actor:cube", 8, 25),),
               "predicates": {"is_grasped": ((9, 25),)},
               "scalars": {}, "kinds": {"is_grasped": "predicate"}, "frames": 40}
        self.write_shard("PickCube-v1", {key: _grasp()}, shard=0, traces=[ep0])
        self.write_shard("PickCube-v1", {key: _grasp()}, shard=1, traces=[ep1])
        merged = miner.load_shards("PickCube-v1", self.shards)
        self.assertEqual(len(merged["traces"]), 2)
        for rec in merged["traces"]:
            on = rec["interactions"][0][3]
            self.assertAlmostEqual(rec["predicates"]["is_grasped"][0][0], on,
                                   delta=1)


class SpatialOnlyMemberTest(MinerTestBase):
    """A goal marker has no collision geometry, so it produces no bucket at
    all -- but a task whose success is "the object reaches *there*" has nowhere
    to point without it."""

    def _mine_with_goal(self, symmetry):
        self.write_shard("PickCube-v1", {
            f"grasp / {EE_KEY} / actor:cube": _grasp(),
            "support / actor:table / actor:cube": _support(),
        }, symmetry=symmetry)
        return self.mine("PickCube-v1")

    def test_a_non_interacting_entity_becomes_a_member(self):
        _, wl = self._mine_with_goal({"actor:goal_site": {"symmetry": "none"}})
        self.assertIn("actor:goal_site", wl["members"])

    def test_it_carries_no_interaction_types(self):
        """Empty types keep the physical and affordance families away from it;
        only planar-distance and height-offset reach a spatial member."""
        _, wl = self._mine_with_goal({"actor:goal_site": {"symmetry": "none"}})
        entry = wl["members"]["actor:goal_site"]
        self.assertEqual(entry["interaction_types"], [])
        self.assertEqual(entry["roles"], ["spatial"])

    def test_it_gets_no_affordance_components(self):
        aff, _ = self._mine_with_goal({"actor:goal_site": {"symmetry": "none"}})
        self.assertNotIn("actor:goal_site", aff["objects"])

    def test_interacting_entities_keep_their_own_roles(self):
        """The pass only fills gaps; it must not overwrite a real member."""
        _, wl = self._mine_with_goal({
            "actor:goal_site": {"symmetry": "none"},
            "actor:cube": {"symmetry": "none"},
        })
        self.assertNotIn("spatial", wl["members"]["actor:cube"]["roles"])
        self.assertIn("grasp", wl["members"]["actor:cube"]["interaction_types"])

    def test_a_task_with_no_extra_entities_is_unchanged(self):
        _, wl = self._mine_with_goal({"actor:cube": {"symmetry": "none"}})
        self.assertEqual(sorted(wl["members"]), ["actor:cube", "actor:table"])
