"""Schedule-candidate mining from successful-episode traces.

Traces are synthesised with the shape the real StackCube run produced: the cubes
resting on the table from frame zero, the gripper reaching cube A, grasping it,
then cube A coming to rest on cube B.
"""

import json
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

from scenegraph.tools import mine_maniskill_schedules as miner


def _episode(seed=0, grasp=True, stack=True, drift=0):
    """One StackCube-shaped record. ``drift`` shifts the later phase."""
    interactions = [
        ("contact", "actor:cubeA", "actor:table-workspace", 0, 34 + drift),
        ("support", "actor:table-workspace", "actor:cubeA", 0, 34 + drift),
        ("contact", "actor:cubeB", "actor:table-workspace", 0, 99),
        ("support", "actor:table-workspace", "actor:cubeB", 0, 99),
        ("contact", "ee", "actor:cubeA", 34 + drift, 78 + drift),
    ]
    predicates = {"is_robot_static": ((0, 5), (90, 99))}
    if grasp:
        interactions.append(("grasp", "ee", "actor:cubeA", 35 + drift, 78 + drift))
        predicates["is_grasped"] = ((35 + drift, 78 + drift),)
    if stack:
        interactions.append(
            ("contact", "actor:cubeA", "actor:cubeB", 76 + drift, 99))
        interactions.append(
            ("support", "actor:cubeB", "actor:cubeA", 76 + drift, 99))
        predicates["is_cubeA_on_cubeB"] = ((77 + drift, 99),)
        predicates["success"] = ((90, 99),)
    return {
        "interactions": tuple(sorted(interactions, key=lambda e: (e[3], e[4], e))),
        "predicates": predicates,
        "scalars": {"elapsed_steps": ([0, 1, 2], [0.0, 1.0, 2.0])},
        "kinds": {},
        "frames": 100,
    }


class MinerTestBase(unittest.TestCase):
    ENV = "StackCube-v1"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.shards = self.tmp / "shards"
        self.configs = self.tmp / "configs"
        self._write_assets()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_assets(self, **overrides):
        aff = self.configs / "affordances"
        wl = self.configs / "subtask_whitelists" / self.ENV
        aff.mkdir(parents=True, exist_ok=True)
        wl.mkdir(parents=True, exist_ok=True)
        objects = {
            "actor:cubeA": {"grasp_components": [{}], "contact_components": [{}],
                            "bottom_components": [{}]},
            "actor:cubeB": {"contact_components": [{}], "support_components": [{}],
                            "bottom_components": [{}]},
            "actor:table-workspace": {"support_components": [{}]},
        }
        objects.update(overrides.get("objects", {}))
        members = {
            "actor:cubeA": {"interaction_types": ["contact", "grasp", "support"]},
            "actor:cubeB": {"interaction_types": ["contact", "support"]},
            "actor:table-workspace": {"interaction_types": ["contact", "support"]},
        }
        with open(aff / f"{self.ENV}.json", "w") as f:
            json.dump({"objects": objects}, f)
        with open(wl / "task_all.json", "w") as f:
            json.dump({"members": members,
                       "bin_edges": overrides.get(
                           "bin_edges",
                           {"planar-distance": [0.1, 0.2, 0.3, 0.4],
                            "height-offset": [-0.1, 0.0, 0.1, 0.2]})}, f)

    def _write_shard(self, traces, schema=3):
        d = self.shards / self.ENV
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "shard_000.pkl", "wb") as f:
            pickle.dump({
                "_schema_version": schema, "env_id": self.ENV, "target": 300,
                "episodes": len(traces), "samples": {}, "seen_counts": {},
                "presence": {}, "episode_presence": {}, "excluded": {},
                "symmetry": {}, "capability": None, "bin_stats": {},
                "complete": [], "incomplete": [], "late": {},
                "traces": traces,
            }, f)

    def _bundle(self, traces, **kw):
        self._write_shard(traces)
        merged = miner.load_shards(self.ENV, self.shards)
        return miner.build_bundle(
            self.ENV, merged, self.configs,
            kw.get("min_presence", miner.MILESTONE_PRESENCE),
            kw.get("min_order", miner.ORDER_CONFIDENCE))


class PresenceTest(MinerTestBase):
    def test_a_milestone_in_every_episode_is_a_candidate(self):
        bundle = self._bundle([_episode() for _ in range(200)])
        entry = bundle["milestones"]["grasp / ee / actor:cubeA"]
        self.assertTrue(entry["is_candidate"])
        self.assertEqual((entry["episodes"], entry["of"]), (200, 200))

    def test_one_missed_episode_in_two_hundred_survives(self):
        """99.5% is the case the gate exists for: a single flicker of the grasp
        detector must not delete the task's defining milestone."""
        traces = [_episode() for _ in range(199)] + [_episode(grasp=False)]
        bundle = self._bundle(traces)
        entry = bundle["milestones"]["grasp / ee / actor:cubeA"]
        self.assertTrue(entry["is_candidate"])
        self.assertEqual(entry["n_missing"], 1)
        self.assertEqual(entry["missing_episodes"], [199])

    def test_a_genuinely_intermittent_milestone_is_not_a_candidate(self):
        traces = [_episode(stack=i % 2 == 0) for i in range(200)]
        bundle = self._bundle(traces)
        self.assertFalse(
            bundle["milestones"]["support / actor:cubeB / actor:cubeA"]["is_candidate"])

    def test_missing_episodes_are_exported_but_truncated(self):
        traces = [_episode(grasp=i >= 100) for i in range(200)]
        entry = self._bundle(traces)["milestones"]["grasp / ee / actor:cubeA"]
        self.assertEqual(entry["n_missing"], 100)
        self.assertEqual(len(entry["missing_episodes"]), 32)

    def test_initial_conditions_are_flagged_not_dropped(self):
        """Cubes on the table from frame zero are the scene, not an
        achievement -- but the refinement pass decides that, not the miner."""
        bundle = self._bundle([_episode() for _ in range(200)])
        table = bundle["milestones"]["support / actor:table-workspace / actor:cubeB"]
        self.assertTrue(table["is_candidate"])
        self.assertTrue(table["initial_condition"])
        stack = bundle["milestones"]["support / actor:cubeB / actor:cubeA"]
        self.assertFalse(stack["initial_condition"])


class OrderingTest(MinerTestBase):
    def test_a_consistent_order_becomes_a_constraint(self):
        bundle = self._bundle([_episode(drift=i % 5) for i in range(200)])
        pair = self._find(bundle, "grasp / ee / actor:cubeA",
                          "support / actor:cubeB / actor:cubeA")
        self.assertEqual(pair["verdict"], "before")
        self.assertEqual(pair["rate_a_before_b"], 1.0)

    def test_a_coin_flip_is_reported_ambiguous_not_resolved(self):
        traces = []
        for i in range(200):
            record = _episode()
            interactions = list(record["interactions"])
            # Flip which of the two table contacts starts first.
            for k, entry in enumerate(interactions):
                if entry[:3] == ("contact", "actor:cubeB", "actor:table-workspace"):
                    interactions[k] = (*entry[:3], 1 if i % 2 else 0, entry[4])
            record["interactions"] = tuple(interactions)
            traces.append(record)
        bundle = self._bundle(traces)
        pair = self._find(bundle, "contact / actor:cubeA / actor:table-workspace",
                          "contact / actor:cubeB / actor:table-workspace")
        self.assertEqual(pair["verdict"], "ambiguous")
        self.assertAlmostEqual(pair["rate_a_before_b"], 0.5, places=2)

    def test_simultaneous_onsets_are_their_own_verdict(self):
        """Two milestones that always start together have no order, and that
        is different from an order that varies: simultaneous milestones belong
        in one phase, an ambiguous pair is a genuine branch."""
        bundle = self._bundle([_episode() for _ in range(200)])
        pair = self._find(bundle, "contact / actor:cubeA / actor:table-workspace",
                          "support / actor:table-workspace / actor:cubeA")
        self.assertEqual(pair["same_frame"], 200)
        self.assertEqual(pair["rate_same_frame"], 1.0)
        self.assertEqual(pair["verdict"], "simultaneous")

    @staticmethod
    def _find(bundle, a, b):
        for entry in bundle["ordering"]:
            if {entry["a"], entry["b"]} == {a, b}:
                return entry
        raise AssertionError(f"no ordering entry for {a} / {b}")


class PhaseAndRoleTest(MinerTestBase):
    def test_phases_follow_the_pair_and_the_clock(self):
        bundle = self._bundle([_episode() for _ in range(200)])
        pairs = [tuple(p["pair"]) for p in bundle["proposed_phases"]]
        self.assertIn(("ee", "actor:cubeA"), pairs)
        self.assertIn(("actor:cubeA", "actor:cubeB"), pairs)
        self.assertLess(pairs.index(("ee", "actor:cubeA")),
                        pairs.index(("actor:cubeA", "actor:cubeB")))

    def test_the_grasped_object_is_proposed_as_movable(self):
        roles = self._bundle([_episode() for _ in range(200)])["proposed_roles"]
        self.assertEqual(roles["movable"], ["actor:cubeA"])
        self.assertEqual(roles["destination_candidates"], ["actor:cubeB"])
        self.assertFalse(roles["ambiguous"])

    def test_the_surface_it_started_on_is_not_a_destination(self):
        """cubeA touches the table from frame zero. That is where it began,
        not somewhere it was taken."""
        roles = self._bundle([_episode() for _ in range(200)])["proposed_roles"]
        self.assertNotIn("actor:table-workspace", roles["destination_candidates"])

    def test_two_grasped_objects_are_flagged_not_guessed(self):
        traces = []
        for _ in range(200):
            record = _episode()
            record["interactions"] = record["interactions"] + (
                ("grasp", "ee", "actor:cubeB", 80, 90),)
            traces.append(record)
        roles = self._bundle(traces)["proposed_roles"]
        self.assertTrue(roles["ambiguous"])
        self.assertEqual(len(roles["movable"]), 2)


class PredicateAgreementTest(MinerTestBase):
    def test_the_matching_predicate_is_found_by_overlap_not_by_name(self):
        bundle = self._bundle([_episode() for _ in range(200)])
        best = bundle["detector_agreement"]["grasp / ee / actor:cubeA"][0]
        self.assertEqual(best["predicate"], "is_grasped")
        self.assertEqual(best["median_onset_delta"], 0.0)
        self.assertEqual(best["median_span_iou"], 1.0)

    def test_disagreement_is_measured_not_corrected(self):
        """A detector two frames early stays two frames early in the bundle."""
        traces = []
        for _ in range(200):
            record = _episode()
            record["predicates"]["is_grasped"] = ((37, 78),)
            traces.append(record)
        best = self._bundle(traces)["detector_agreement"][
            "grasp / ee / actor:cubeA"][0]
        self.assertEqual(best["median_onset_delta"], 2.0)
        self.assertLess(best["onset_within_1_frame"], 0.5)

    def test_a_toggling_predicate_is_flagged(self):
        bundle = self._bundle([_episode() for _ in range(200)])
        self.assertTrue(bundle["environment_predicates"]["is_robot_static"]["toggles"])
        self.assertFalse(bundle["environment_predicates"]["is_grasped"]["toggles"])


class ClauseInventoryTest(MinerTestBase):
    def test_only_relations_with_components_behind_them_are_scorable(self):
        inventory = self._bundle([_episode() for _ in range(10)])["scorable_clauses"]
        ee_cube = inventory["scorable"]["ee / actor:cubeA"]
        self.assertTrue(ee_cube["grasp-compatibility"])
        # cubeB is never grasped, so a grasp clause on it would score zero
        # forever and read as "unobserved" while doing so.
        self.assertFalse(inventory["scorable"]["ee / actor:cubeB"]["grasp-compatibility"])

    def test_support_compatibility_needs_both_halves(self):
        inventory = self._bundle([_episode() for _ in range(10)])["scorable_clauses"]
        pair = inventory["scorable"]["actor:cubeA / actor:cubeB"]
        self.assertTrue(pair["support-compatibility"])
        # The table has a surface but nothing declares a bottom against cubeB.
        self.assertFalse(
            inventory["scorable"]["actor:cubeB / actor:table-workspace"][
                "contain-compatibility"])

    def test_missing_spatial_bins_are_reported(self):
        self._write_assets(bin_edges={"planar-distance": [0.1, 0.2, 0.3, 0.4]})
        inventory = self._bundle([_episode() for _ in range(10)])["scorable_clauses"]
        self.assertTrue(inventory["spatial_bins"]["planar-distance"])
        self.assertFalse(inventory["spatial_bins"]["height-offset"])

    def test_mining_a_schedule_without_assets_fails_loudly(self):
        shutil.rmtree(self.configs)
        with self.assertRaises(SystemExit) as cm:
            self._bundle([_episode() for _ in range(10)])
        self.assertIn("Mine the task's assets first", str(cm.exception))


class BundleTest(MinerTestBase):
    def test_end_to_end_writes_a_task_file_and_a_manifest(self):
        self._write_shard([_episode() for _ in range(200)])
        out = self.tmp / "bundle"
        miner.main(["--env-id", self.ENV, "--shards", str(self.shards),
                    "--configs", str(self.configs), "--out", str(out)])
        manifest = json.load(open(out / "manifest.json"))
        self.assertIn(self.ENV, manifest["tasks"])
        self.assertEqual(manifest["gates"]["milestone_presence"], 0.99)
        task = json.load(open(out / f"{self.ENV}.json"))
        self.assertEqual(task["successful_episodes"], 200)
        self.assertTrue(task["example_traces"])

    def test_the_bundle_is_json_serialisable_end_to_end(self):
        """It is handed to a refinement pass as a file, so nothing in it may be
        a tuple key, a numpy scalar or a set."""
        bundle = self._bundle([_episode() for _ in range(50)])
        json.loads(json.dumps(bundle))

    def test_traceless_shards_are_rejected(self):
        self._write_shard([])
        merged = miner.load_shards(self.ENV, self.shards)
        with self.assertRaises(SystemExit):
            miner.build_bundle(self.ENV, merged, self.configs, 0.99, 0.95)


if __name__ == "__main__":
    unittest.main()


class SpatialDestinationTest(MinerTestBase):
    """A goal marker appears in no bucket and therefore in no trace, so the
    milestone-driven role proposal can never see it."""

    def _with_goal(self):
        self._write_assets()
        aff = json.load(open(self.configs / "affordances" / f"{self.ENV}.json"))
        wl_path = self.configs / "subtask_whitelists" / self.ENV / "task_all.json"
        wl = json.load(open(wl_path))
        wl["members"]["actor:goal_site"] = {"interaction_types": []}
        json.dump(wl, open(wl_path, "w"))
        return self._bundle([_episode() for _ in range(50)])

    def test_it_is_reported_as_spatial_only(self):
        bundle = self._with_goal()
        self.assertEqual(bundle["scorable_clauses"]["spatial_only"],
                         ["actor:goal_site"])

    def test_it_becomes_a_destination_candidate(self):
        bundle = self._with_goal()
        self.assertIn("actor:goal_site",
                      bundle["proposed_roles"]["destination_candidates"])

    def test_only_spatial_relations_are_scorable_against_it(self):
        scorable = self._with_goal()["scorable_clauses"]["scorable"]
        pair = scorable["actor:cubeA / actor:goal_site"]
        self.assertTrue(pair["planar-distance"])
        self.assertFalse(pair["contact"])
        self.assertFalse(pair["support"])

    def test_an_interacting_member_is_not_listed_spatial_only(self):
        bundle = self._with_goal()
        self.assertNotIn("actor:cubeA", bundle["scorable_clauses"]["spatial_only"])
