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

    EXTENTS = {
        "actor:table-workspace": [1.209, 0.6045, 0.4598],
        "actor:bin": [0.025, 0.025, 0.0075],
        "actor:box_with_hole": [0.1234, 0.1234, 0.1234],
        "actor:peg": [0.1234, 0.022, 0.022],
        "actor:sphere": [0.02, 0.02, 0.02],
    }
    MEMBERS = {
        "actor:table-workspace": {"interaction_types": ["contact", "support"]},
        "actor:bin": {"interaction_types": ["contact", "support"]},
        "actor:box_with_hole": {"interaction_types": ["contact", "support",
                                                      "contain"]},
        "actor:peg": {"interaction_types": ["contact", "grasp", "support"]},
        "actor:sphere": {"interaction_types": ["contact", "grasp", "support"]},
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
