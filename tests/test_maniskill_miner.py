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
from scenegraph.core.relation_rules import required_bin_keys
from scenegraph.core.spatial_metrics import (
    EE_OBJECT_SCOPE,
    OBJECT_OBJECT_SCOPE,
    spatial_bin_key,
    stat_key,
)
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


def _support(n=300, jitter=0.0, sup="actor:cubeB", sub="actor:cubeA"):
    """``sup`` carries ``sub``. Contact anchors sit on the supporter's top."""
    out = []
    rng = np.random.default_rng(0)
    for _ in range(n):
        off = rng.uniform(-jitter, jitter, size=2) if jitter else [0.0, 0.0]
        out.append(_sample({
            "force": 1.0, "key_a": sup, "key_b": sub,
            "pose_a": _pose([0.0, 0.0, 0.0]),
            "pose_b": _pose([float(off[0]), float(off[1]), 0.04]),
            "anchor_a_local": [float(off[0]), float(off[1]), 0.02],
            "normal_a_local": [0.0, 0.0, 1.0],
            "anchor_b_local": [0.0, 0.0, -0.02],
            "normal_b_local": [0.0, 0.0, -1.0],
        }))
    return out


def _contain(n=300):
    return [_sample({"force": 1.0, "axial": 0.05, "hole_half_width": 0.02,
                     "hole_pose": _pose([0.1, 0.0, 0.1]),
                     "container_pose": _pose([0.0, 0.0, 0.0]),
                     "key_pose": _pose([0.1, 0.0, 0.1]),
                     "containee_pose": _pose([0.05, 0.0, 0.1])})
            for _ in range(n)]


def _ee_stats(planar=0.8, height=0.4):
    return {
        stat_key(EE_OBJECT_SCOPE, "planar-distance"): planar,
        stat_key(EE_OBJECT_SCOPE, "height-offset"): height,
        stat_key(EE_OBJECT_SCOPE, "planar-distance") + "_change": 0.05,
        stat_key(EE_OBJECT_SCOPE, "height-offset") + "_change": 0.03,
    }


def _pose_pairs(a="actor:cubeA", b="actor:cubeB", n=8):
    """Two objects drifting apart, so both object scales are non-degenerate."""
    out = []
    for i in range(n):
        pose_a = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        pose_b = [0.05 * i, 0.0, 0.02 * i, 1.0, 0.0, 0.0, 0.0]
        prev_b = [0.05 * (i - 1), 0.0, 0.02 * (i - 1), 1.0, 0.0, 0.0, 0.0]
        out.append({
            "key_a": a, "key_b": b,
            "pose_a": pose_a, "pose_b": pose_b,
            "prev_pose_a": pose_a if i else None,
            "prev_pose_b": prev_b if i else None,
        })
    return out


class MinerTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.shards = self.tmp / "evidence"
        self.configs = self.tmp / "configs"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_shard(self, env_id, samples, **kw):
        # Object-object scales come from reprojected pose pairs, so a
        # shard without them calibrates nothing on that scope.
        d = self.shards / env_id
        d.mkdir(parents=True, exist_ok=True)
        payload = {
            "_schema_version": kw.get("schema", 4), "env_id": env_id,
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
            "bin_stats": kw.get("bin_stats", _ee_stats()),
            "bin_pose_pairs": kw.get("bin_pose_pairs", _pose_pairs()),
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
        for key in required_bin_keys({"object_object_spatial": True}):
            self.assertTrue(wl["bin_edges"].get(key), key)

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
            "support / actor:box / actor:peg": _support(
                300, sup="actor:box", sub="actor:peg"),
            "support / actor:peg / actor:box": _support(
                300, sup="actor:box", sub="actor:peg")[:5],
        })
        aff, _ = self.mine("PegInsertionSide-v1")
        self.assertIn("support_components", aff["objects"]["actor:box"])
        self.assertNotIn("support_components",
                         aff["objects"].get("actor:peg", {}))

    def test_a_genuinely_close_split_raises(self):
        self.write_shard("Ambiguous-v1", {
            f"grasp / {EE_KEY} / actor:x": _grasp(),
            "support / actor:x / actor:y": _support(
                300, sup="actor:x", sub="actor:y"),
            "support / actor:y / actor:x": _support(
                300, sup="actor:x", sub="actor:y")[:280],
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
                         bin_stats=_ee_stats(0.5, 0.2))
        self.write_shard("PickCube-v1", {key: _grasp(150)}, shard=1,
                         bin_stats=_ee_stats(0.9, 0.1))
        merged = miner.load_shards("PickCube-v1", self.shards)
        self.assertEqual(len(merged["samples"][key]), 300)
        self.assertEqual(merged["episodes"], 600)
        pd = stat_key(EE_OBJECT_SCOPE, "planar-distance")
        ho = stat_key(EE_OBJECT_SCOPE, "height-offset")
        self.assertAlmostEqual(merged["bin_stats"][pd], 0.9)
        self.assertAlmostEqual(merged["bin_stats"][ho], 0.2)


if __name__ == "__main__":
    unittest.main()


class FinalPresenceTest(MinerTestBase):
    """A bucket can pass the freeze gate and drift below it by the end."""

    def _drifted(self):
        drifter = "support / actor:cube / actor:tool"
        self.write_shard("PullCubeTool-v1", {
            f"grasp / {EE_KEY} / actor:tool": _grasp(),
            drifter: _support(sup="actor:cube", sub="actor:tool"),
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
    """Only unsigned distance may use distribution quantiles.

    Height and temporal change have signed vocabulary, so their edges must be
    symmetric around zero even when the observed scene distribution is not.
    """

    def _bimodal(self):
        import random
        rng = random.Random(0)
        # Half the pairs are object-object (near zero), half object-table.
        return [rng.gauss(0.03, 0.01) for _ in range(500)] + \
               [rng.gauss(0.92, 0.02) for _ in range(500)]

    def _mine_with(self, samples, height_max=0.4):
        self.write_shard("StackCube-v1", {
            f"grasp / {EE_KEY} / actor:cubeA": _grasp()},
            bin_stats=_ee_stats(0.8, height_max))
        path = next((self.shards / "StackCube-v1").glob("*.pkl"))
        with open(path, "rb") as f:
            shard = pickle.load(f)
        shard["bin_samples"] = samples
        with open(path, "wb") as f:
            pickle.dump(shard, f)
        return self.mine("StackCube-v1")[1]["bin_edges"]

    def _labels(self, relation, edges, values):
        from scenegraph.core.relation_rules import (
            CHANGE_LABELS, SPATIAL_LABELS, bin_label,
        )
        labels = (CHANGE_LABELS if relation.endswith("-change")
                  else SPATIAL_LABELS[relation])
        return [bin_label(v, edges, labels) for v in values]

    @staticmethod
    def _ee(relation):
        return spatial_bin_key(EE_OBJECT_SCOPE, relation)

    def test_planar_distance_uses_quantiles(self):
        import numpy as np
        values = np.linspace(0.01, 0.80, 1000, dtype=np.float32)
        edges = self._mine_with(
            {stat_key(EE_OBJECT_SCOPE, "planar-distance"): values},
        )[self._ee("planar-distance")]
        expected = [float(np.quantile(values, p)) for p in miner._EDGE_PROBS]
        np.testing.assert_allclose(edges, expected)

    def test_signed_edges_stay_symmetric_however_skewed_the_samples(self):
        """Skew may set the scale; it may never move zero out of the middle."""
        import numpy as np
        values = np.asarray(self._bimodal(), dtype=np.float32)
        edges = self._mine_with(
            {stat_key(EE_OBJECT_SCOPE, "height-offset"): values},
        )[self._ee("height-offset")]
        self.assertAlmostEqual(edges[0], -edges[3])
        self.assertAlmostEqual(edges[1], -edges[2])
        self.assertEqual(self._labels("height-offset", edges, [0.0]), ["level"])

    def test_a_second_mode_is_not_an_outlier(self):
        """Half the data is the run's range, not a failure to discard."""
        import numpy as np
        values = np.asarray(self._bimodal(), dtype=np.float32)
        edges = self._mine_with(
            {stat_key(EE_OBJECT_SCOPE, "height-offset"): values},
        )[self._ee("height-offset")]
        # The 0.92 mode still reaches the outer bins.
        self.assertGreater(edges[3], 0.4)

    def test_a_thin_tail_does_not_set_the_signed_scale(self):
        """PullCubeTool drops its cube in ~9% of frames.

        Those 0.88m readings are true and useless: taken as the maximum they
        put every on-table height in one bin.
        """
        import numpy as np
        rng = np.random.default_rng(0)
        real = rng.uniform(-0.3, 0.3, size=940)
        fallen = np.full(60, -5.0)
        values = np.concatenate([real, fallen]).astype(np.float32)
        edges = self._mine_with(
            {stat_key(EE_OBJECT_SCOPE, "height-offset"): values},
        )[self._ee("height-offset")]
        self.assertLess(edges[3], 1.0, "the tail set the scale")
        self.assertGreater(edges[3], 0.1, "the real range was clipped away")
        self.assertAlmostEqual(edges[0], -edges[3])

    def test_zero_is_level_and_stable(self):
        edges = self._mine_with({})
        self.assertEqual(
            self._labels("height-offset",
                         edges[self._ee("height-offset")], [0.0]),
            ["level"],
        )
        for scope in (EE_OBJECT_SCOPE, OBJECT_OBJECT_SCOPE):
            for relation in ("planar-distance", "height-offset"):
                key = spatial_bin_key(scope, relation) + "-change"
                self.assertEqual(
                    self._labels(relation + "-change", edges[key], [0.0]),
                    ["stable"], key,
                )

    def test_degenerate_statistic_keeps_the_max_scale(self):
        import numpy as np
        flat = np.zeros(500, dtype=np.float32)
        edges = self._mine_with(
            {stat_key(EE_OBJECT_SCOPE, "planar-distance"): flat},
        )[self._ee("planar-distance")]
        self.assertGreater(max(edges), 0.0)

    def test_too_few_samples_keeps_the_max_scale(self):
        """A thin reservoir gives unstable quantiles, so fall back."""
        import numpy as np
        tiny = np.asarray(self._bimodal()[:50], dtype=np.float32)
        got = self._mine_with(
            {stat_key(EE_OBJECT_SCOPE, "planar-distance"): tiny}
        )[self._ee("planar-distance")]
        self.assertEqual(got, self._mine_with({})[self._ee("planar-distance")])

    def test_reservoirs_concatenate_across_shards(self):
        import numpy as np
        from scenegraph.adapters.interaction_events import BinStats
        b = BinStats()
        for _ in range(3):
            b.observe({"ee": [0, 0, 0, 1, 0, 0, 0],
                       "b": [1, 0, 0.5, 1, 0, 0, 0]}, 0)
        self.assertEqual(
            len(b.reservoir()[stat_key(EE_OBJECT_SCOPE, "height-offset")]), 3)

    def test_change_statistics_use_runtime_horizon(self):
        import numpy as np
        from scenegraph.adapters.interaction_events import BinStats
        b = BinStats(horizon=5)
        for frame in range(6):
            b.observe({
                "ee": [0, 0, 0, 1, 0, 0, 0],
                "b": [10 - frame, 0, 0, 1, 0, 0, 0],
            }, frame)
        changes = b.reservoir()[
            stat_key(EE_OBJECT_SCOPE, "planar-distance") + "_change"]
        np.testing.assert_allclose(changes, [-5.0])

    def test_missing_quantiles_fall_back(self):
        edges = self._mine_with({})
        self.assertTrue(edges[self._ee("planar-distance")])
        self.assertTrue(edges[self._ee("height-offset")])


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
        tool = _support(sup="actor:cube", sub="actor:tool")
        self.write_shard("PullCubeTool-v1", {key: _grasp(), rare: tool},
                         shard=0, episodes=100,
                         episode_presence={key: 100, rare: 100})
        self.write_shard("PullCubeTool-v1", {key: _grasp(), rare: tool},
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
            "support / actor:table / actor:cube": _support(
                sup="actor:table", sub="actor:cube"),
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


class StructuralSurfaceClassificationTest(unittest.TestCase):
    """Telling a tabletop from a bin.

    The extents below are the ones the collector actually reported for the
    shipped tasks, so this is a regression against measured geometry rather
    than against a guess.
    """

    # Reported by the collector for the shipped tasks. The tabletop is
    # identical in every one of them -- they share a TableSceneBuilder -- and
    # six independent actors sit on the other side of the threshold.
    EXTENTS = {
        "actor:table-workspace": [1.209, 0.6045, 0.4598],
        "actor:bin": [0.025, 0.025, 0.0075],
        "actor:box_with_hole": [0.1234, 0.1234, 0.1234],
        "actor:peg": [0.1234, 0.022, 0.022],
        "actor:sphere": [0.02, 0.02, 0.02],
        "actor:charger": [0.028, 0.015, 0.012],
        "actor:receptacle": [0.01, 0.05, 0.05],
    }
    MEMBERS = {
        "actor:table-workspace": {"interaction_types": ["contact", "support"]},
        "actor:bin": {"interaction_types": ["contact", "support"]},
        "actor:box_with_hole": {"interaction_types": ["contact", "support",
                                                      "contain"]},
        "actor:peg": {"interaction_types": ["contact", "grasp", "support"]},
        "actor:sphere": {"interaction_types": ["contact", "grasp", "support"]},
        "actor:charger": {"interaction_types": ["contact", "grasp", "support"]},
        "actor:receptacle": {"interaction_types": ["contact", "support",
                                                   "contain"]},
    }

    def _classify(self, extents=None, members=None):
        return miner.structural_surfaces(
            {"extents": dict(extents if extents is not None else self.EXTENTS)},
            dict(members if members is not None else self.MEMBERS),
        )

    def test_only_the_table_is_structural(self):
        self.assertEqual(sorted(self._classify()), ["actor:table-workspace"])

    def test_the_bin_is_not(self):
        """It supports the sphere and is kinematic, exactly like the table.
        Only its size says otherwise."""
        self.assertNotIn("actor:bin", self._classify())

    def test_the_decision_has_room_either_side(self):
        """A threshold that only just separates them would be a coincidence.
        The table clears it 2x and the next largest object misses it 2.4x."""
        table = min(self.EXTENTS["actor:table-workspace"][:2])
        largest_other = max(
            min(v[:2]) for k, v in self.EXTENTS.items()
            if k != "actor:table-workspace")
        self.assertGreater(table / miner.STRUCTURAL_SURFACE_MIN_HALF_EXTENT, 2)
        self.assertGreater(
            miner.STRUCTURAL_SURFACE_MIN_HALF_EXTENT / largest_other, 2)

    def test_an_object_moved_by_a_tool_is_still_a_manipuland(self):
        """PullCubeTool grasps the tool, never the cube. The cube is dragged
        by it, so a grasp-only rule misses the one object the task is about
        and the miner refused to write the asset at all."""
        members = {
            "actor:cube": {"interaction_types": ["contact", "support"]},
            "actor:l_shape_tool": {"interaction_types": ["contact", "grasp"]},
            "actor:table-workspace": {"interaction_types": ["contact",
                                                            "support"]},
        }
        buckets = ["support / actor:table-workspace / actor:cube",
                   "grasp / ee / actor:l_shape_tool"]
        families = miner.object_families(
            members, buckets, {"actor:table-workspace"})
        self.assertEqual(families["actor:cube"], _MANIPULAND)
        self.assertEqual(miner.ambiguous_families(families), [])

    def test_being_supported_does_not_outrank_being_a_holder(self):
        """A bin resting on a table is still a receptacle. Rule order is what
        decides, and the holder rule comes first."""
        members = {
            "actor:bin": {"interaction_types": ["contact", "support"]},
            "actor:sphere": {"interaction_types": ["contact", "support"]},
        }
        buckets = ["support / actor:bin / actor:sphere",
                   "support / actor:table-workspace / actor:bin"]
        families = miner.object_families(members, buckets, set())
        self.assertEqual(families["actor:bin"], _RECEPTACLE)

    def test_an_object_that_neither_rests_nor_holds_is_still_ambiguous(self):
        """The guard has to keep biting, or the new rule has just made
        everything a manipuland."""
        members = {"actor:mystery": {"interaction_types": ["contact"]}}
        families = miner.object_families(members, [], set())
        self.assertEqual(miner.ambiguous_families(families),
                         ["actor:mystery"])

    def test_a_long_thin_object_is_not_a_surface(self):
        """The smaller horizontal extent decides, so a rail as long as a table
        is still not something you can place things anywhere on."""
        rail = {"actor:rail": [1.5, 0.02, 0.02]}
        members = {"actor:rail": {"interaction_types": ["support"]}}
        self.assertEqual(self._classify(rail, members), {})

    def test_a_large_object_that_supports_nothing_is_scenery(self):
        members = {"actor:table-workspace": {"interaction_types": ["contact"]}}
        self.assertEqual(self._classify(members=members), {})

    def test_flatness_is_not_the_test(self):
        """The tabletop actor includes its legs -- 0.46m vertically against
        0.60m horizontally -- so an aspect-ratio rule would reject the one
        object this exists to find."""
        table = self.EXTENTS["actor:table-workspace"]
        self.assertLess(min(table[:2]) / table[2], 1.5)
        self.assertIn("actor:table-workspace", self._classify())

    def test_unreadable_geometry_is_not_treated_as_small(self):
        """A missing measurement is not evidence of absence. Defaulting it to
        'not a surface' reinstates the ~0.9m origin error silently."""
        extents = dict(self.EXTENTS)
        del extents["actor:table-workspace"]
        self.assertEqual(self._classify(extents), {})
        self.assertIn(
            "actor:table-workspace",
            miner.unclassified_supporters(
                {"extents": extents}, dict(self.MEMBERS)),
        )

    def test_a_classified_member_is_not_reported_unclassified(self):
        self.assertEqual(
            miner.unclassified_supporters(
                {"extents": dict(self.EXTENTS)}, dict(self.MEMBERS)),
            [],
        )

    def test_the_reason_is_recorded_for_review(self):
        reason = self._classify()["actor:table-workspace"]
        self.assertIn("0.605", reason)
        self.assertIn("0.3", reason)


from scenegraph.adapters.interaction_events import (
    KIND_EE_OBJECT as _EE,
    KIND_OBJECT_REGION as _REGION,
    KIND_OBJECT_SITE as _SITE,
)
from scenegraph.core.spatial_metrics import (
    FAMILY_MANIPULAND as _MANIPULAND,
    FAMILY_RECEPTACLE as _RECEPTACLE,
    FAMILY_STRUCTURAL as _STRUCTURAL,
    OBJECT_REGION_PLANAR_KEY as _REGION_PD,
    OBJECT_SITE_HEIGHT_KEY as _SITE_HO,
    OBJECT_SITE_PLANAR_KEY as _SITE_PD,
    ee_family_bin_key as _fkey,
)


class SiteDeclarationSourceTest(unittest.TestCase):
    """Reviewed declarations are input; mined assets are output.

    Reading them from the same directory meant a pilot writing to /tmp looked
    for its declarations there, found none, and silently mined a Peg asset
    with no hole site at all -- the live geometry collected and thrown away.
    """

    SITE_SAMPLE = {"kind": "object-site", "src_key": "actor:peg",
                   "dst_key": "spatial:hole_site",
                   "src_pose": [0, 0, 0, 1, 0, 0, 0],
                   "dst_pose": [0.1, 0, 0, 1, 0, 0, 0],
                   "prev_src_pose": None, "prev_dst_pose": None}
    MEMBERS = {"actor:peg": {"interaction_types": ["grasp"]}}

    def _sites_dir(self, declared):
        import json as _json
        tmp = Path(tempfile.mkdtemp())
        if declared is not None:
            with open(tmp / "PegInsertionSide-v1.json", "w") as handle:
                _json.dump({"env_id": "PegInsertionSide-v1",
                            "sites": declared}, handle)
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        return tmp

    def test_the_shipped_declarations_are_found_by_default(self):
        for env in ("PickCube-v1", "PegInsertionSide-v1", "PullCubeTool-v1"):
            self.assertTrue(
                miner.declared_sites(env, miner.SITES_DIR), env)

    def test_a_task_with_no_declaration_file_has_no_sites(self):
        self.assertEqual(
            miner.declared_sites("PlaceSphere-v1", miner.SITES_DIR), {})

    def test_unclaimed_site_evidence_fails_loudly(self):
        """Not silently mined away, which is what happened."""
        with self.assertRaises(SystemExit) as ctx:
            miner.site_declarations(
                {"bin_keyed_pairs": [self.SITE_SAMPLE]}, self.MEMBERS,
                "PegInsertionSide-v1", self._sites_dir(None))
        self.assertIn("no declaration claims", str(ctx.exception))

    def test_a_declaration_with_no_evidence_fails_loudly(self):
        declared = {"spatial:hole_site": {
            "site_type": "surface", "subject": "actor:peg"}}
        with self.assertRaises(SystemExit) as ctx:
            miner.site_declarations(
                {"bin_keyed_pairs": []}, self.MEMBERS,
                "PegInsertionSide-v1", self._sites_dir(declared))
        self.assertIn("no object-site samples", str(ctx.exception))

    def test_a_declaration_naming_an_absent_member_fails(self):
        declared = {"spatial:hole_site": {
            "site_type": "surface", "subject": "actor:ghost"}}
        with self.assertRaises(SystemExit):
            miner.site_declarations(
                {"bin_keyed_pairs": [self.SITE_SAMPLE]}, self.MEMBERS,
                "PegInsertionSide-v1", self._sites_dir(declared))

    def test_matching_declaration_and_evidence_passes(self):
        declared = {"spatial:hole_site": {
            "site_type": "surface", "subject": "actor:peg"}}
        out = miner.site_declarations(
            {"bin_keyed_pairs": [self.SITE_SAMPLE]}, self.MEMBERS,
            "PegInsertionSide-v1", self._sites_dir(declared))
        self.assertEqual(sorted(out), ["spatial:hole_site"])

    def test_a_virtual_site_gets_a_spatial_vocabulary_member(self):
        members = miner.admit_site_members(self.MEMBERS, {
            "spatial:hole_site": {"site_type": "surface"},
        })
        self.assertEqual(members["spatial:hole_site"], {
            "roles": ["spatial"],
            "interaction_types": [],
            "kind": "spatial",
        })

    def test_a_real_goal_marker_keeps_its_mined_member(self):
        original = {
            "actor:goal_site": {
                "roles": ["spatial"],
                "interaction_types": [],
                "kind": "actor",
                "family": "goal-marker",
            }
        }
        members = miner.admit_site_members(original, {
            "actor:goal_site": {"site_type": "point"},
        })
        self.assertEqual(members["actor:goal_site"], original["actor:goal_site"])


class KeyedCalibrationTest(unittest.TestCase):
    """Scales that only the keyed reservoir can produce.

    The ranges below are the ones actually recorded in the PlaceSphere
    rollout, so what is under test is that the miner separates them rather
    than that the numbers are plausible.
    """

    TABLE = "actor:table-workspace"
    # The shipped PlaceSphere asset: the tabletop is 0.92m above its own
    # origin, and its mined normal points down.
    SURFACE = {"anchor": [0.0, 0.0, 0.92], "outward_normal": [0.0, 0.0, -1.0]}
    FAMILIES = {"actor:sphere": _MANIPULAND, "actor:bin": _RECEPTACLE,
                TABLE: _STRUCTURAL}

    def _pose(self, x=0.0, y=0.0, z=0.0):
        return [x, y, z, 1.0, 0.0, 0.0, 0.0]

    def _keyed(self, kind, src_key, dst_key, src, dst):
        return {"kind": kind, "src_key": src_key, "dst_key": dst_key,
                "src_pose": src, "dst_pose": dst,
                "prev_src_pose": None, "prev_dst_pose": None}

    def _merged(self):
        rng = np.random.default_rng(0)
        pairs = []
        for _ in range(300):
            pairs.append(self._keyed(
                _EE, "ee", "actor:sphere",
                self._pose(z=rng.uniform(0.001, 0.135)), self._pose()))
            pairs.append(self._keyed(
                _EE, "ee", "actor:bin",
                self._pose(z=rng.uniform(0.019, 0.153)), self._pose()))
            # Recorded against the table ORIGIN, which is what the collector
            # ships: 0.94-1.07m, because the origin sits under its own top.
            pairs.append(self._keyed(
                _EE, "ee", self.TABLE,
                self._pose(z=rng.uniform(0.941, 1.075)), self._pose()))
            pairs.append(self._keyed(
                _SITE, "actor:peg", "spatial:hole_site",
                self._pose(x=-rng.uniform(0.0, 0.30),
                           z=rng.uniform(-0.05, 0.05)), self._pose()))
            pairs.append(self._keyed(
                _REGION, "actor:cube", "spatial:pull_goal_region",
                self._pose(x=rng.uniform(0.55, 0.95)), self._pose()))
        return {"bin_keyed_pairs": pairs, "bin_pose_pairs": [],
                "bin_stats": {}, "bin_samples": {}}

    SITES = {
        "spatial:hole_site": {"subject": "actor:peg", "site_type": "surface"},
        "spatial:pull_goal_region": {"subject": "actor:cube",
                                     "site_type": "region"},
    }

    def _edges(self, objects=None, sites=None):
        if objects is None:
            objects = {self.TABLE: {"reference_surface": dict(self.SURFACE)}}
        return miner._bin_edges(
            self._merged(), objects, self.FAMILIES,
            self.SITES if sites is None else sites)

    def test_the_table_does_not_set_the_manipuland_scale(self):
        """The whole failure: one shared scale gave every end-effector height
        in every task a +/-0.206m deadband, so nothing the gripper did to the
        sphere ever left 'level'."""
        edges = self._edges()
        self.assertLess(edges[_fkey(_MANIPULAND)][2], 0.05)

    def test_each_family_gets_its_own_scale(self):
        edges = self._edges()
        for family in (_MANIPULAND, _RECEPTACLE, _STRUCTURAL):
            self.assertIn(_fkey(family), edges)

    def test_the_structural_scale_is_reprojected_onto_the_surface(self):
        """The reservoir ships end-effector-to-table-ORIGIN heights, because
        nothing knows the table is a surface until its extents are read and
        its anchors mined. Calibrating on those while the runtime labels
        surface-relative heights is the drift spatial_metrics exists to stop:
        it would put a 0.15m clearance inside a 0.21m deadband."""
        edges = self._edges()
        self.assertLess(edges[_fkey(_STRUCTURAL)][2], 0.10)

    def test_without_a_reference_surface_the_structural_scale_is_dropped(self):
        """Not silently calibrated on origins instead."""
        edges = self._edges(objects={})
        self.assertNotIn(_fkey(_STRUCTURAL), edges)

    def test_the_site_ladder_gets_both_relations(self):
        edges = self._edges()
        self.assertIn(_SITE_PD, edges)
        self.assertIn(_SITE_HO, edges)

    def test_the_region_gets_planar_and_no_height(self):
        """A disc around the robot base has a horizontal extent and no height
        target at all."""
        edges = self._edges()
        self.assertIn(_REGION_PD, edges)
        self.assertNotIn("object-region-height-offset", edges)

    def test_the_region_scale_spans_the_pull(self):
        """The cube starts ~0.9m out and succeeds inside 0.6m. On the
        object-object scale, which tops out near 0.31m for this task, every
        frame of the pull would read 'very-far'."""
        edges = self._edges()
        self.assertGreater(max(edges[_REGION_PD]), 0.6)

    def test_an_undeclared_region_calibrates_nothing(self):
        """The collector records region samples against every movable actor,
        because it cannot know which one a task drags into its goal. Only a
        reviewed declaration turns those into a scale -- otherwise PlaceSphere,
        which has no goal region at all, would be given one."""
        edges = self._edges(sites={})
        self.assertNotIn(_REGION_PD, edges)
        self.assertNotIn(_SITE_PD, edges)

    def test_only_the_declared_subject_calibrates_the_region(self):
        edges = self._edges(sites={
            "spatial:pull_goal_region": {"subject": "actor:other",
                                         "site_type": "region"}})
        self.assertNotIn(_REGION_PD, edges)

    def test_the_superseded_shared_height_scale_is_dropped(self):
        """Nothing reads it once families exist, and leaving it in the asset
        invites something to."""
        self.assertNotIn("ee-object-height-offset", self._edges())

    def test_an_unclassified_member_contributes_to_no_family(self):
        edges = miner._bin_edges(
            self._merged(),
            {self.TABLE: {"reference_surface": dict(self.SURFACE)}},
            {"actor:sphere": _MANIPULAND}, self.SITES,
        )
        self.assertIn(_fkey(_MANIPULAND), edges)
        self.assertNotIn(_fkey(_RECEPTACLE), edges)
