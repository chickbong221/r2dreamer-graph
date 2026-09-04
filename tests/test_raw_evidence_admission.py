"""Raw evidence keeps what it saw; the runtime asset keeps what it can measure.

The MS-HAB tuna-can demonstrations touched a sofa and a fridge door on the way
to the target. Neither is grasped, neither holds anything, neither is an
extended surface, so no height-family rule reaches either one -- and the mine
stopped, four files into nine, holding a directory that looked finished.

The two claims are separate and both have to hold:

* ``full-evidence`` is a *record*. Deleting the sofa to make the mine finish
  would throw away evidence to avoid an error message, and inventing a family
  for it would give a token two meanings. It is kept, and marked.
* ``target-supporters`` is what a training run loads. A member it admits that
  nothing can scale still stops the pipeline -- the marking is a record, not a
  waiver.

The interesting case is the one where those two disagree, which is why the
negative test below builds a member that is both unresolved *and* a direct
supporter of the target: pruning keeps it, so pruning has to refuse.
"""

import json
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path

from scenegraph.core import families as rules
from scenegraph.core.affordance import load_affordance_set
from scenegraph.core.spatial_metrics import (
    FAMILY_MANIPULAND,
    FAMILY_STRUCTURAL,
)
from scenegraph.tools.build_subtask_whitelists import (
    MEMBERSHIP_FULL_EVIDENCE,
    MEMBERSHIP_TARGET_SUPPORTERS,
    _WhitelistBuilder,
)
from scenegraph.core.sites import SITE_EE_REST
from scenegraph.core.whitelist import load_whitelist
from scenegraph.tools import (
    build_subtask_whitelists,
    build_union_whitelist,
    prune_whitelists,
)
from scenegraph.tools.build_subtask_whitelists import MIN_ROLLOUT_SCHEMA
from scenegraph.tools.prepare_assets import _unmeasurable_members
from scenegraph.tools.prune_whitelists import prune_payload

TUNA = "actor:007_tuna_fish_can"
COUNTER = "link:kitchen_counter-0/body"
# The two the tuna-can demonstrations actually dragged into the evidence.
SOFA = "actor:frl_apartment_sofa"
DOOR = "link:fridge-0/top_door"

FAMILY_STATS = ("ee_manipuland_height_offset",
                "ee_receptacle_height_offset",
                "ee_structural_surface_height_offset",
                "ee_goal_marker_height_offset")


def _extent(half):
    return {"half_extents": list(half), "extent_status": "ok"}


def _affordance_payload():
    """The counter's mined support plane, which a surface is measured against.

    Without it the miner has nothing to measure a structural surface from,
    and the actor origin it would otherwise fall back to sits ~0.9m below the
    counter's own top.
    """
    payload = {"_schema_version": 4, "objects": {
        TUNA: {"grasp_components": []},
        COUNTER: {"support_components": [
            {"surface_anchor": [0.0, 0.0, 0.45],
             "surface_normal": [0.0, 0.0, -1.0], "partner": TUNA},
        ]},
    }}
    payload["objects"][COUNTER]["reference_surface"] = (
        rules.reference_surface_from_supports(payload["objects"][COUNTER]))
    return payload


def _affordances():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "aff.json"
        path.write_text(json.dumps(_affordance_payload()))
        return load_affordance_set(str(path))


class _Fixture(unittest.TestCase):
    """One grasped target on a counter, plus furniture the arm only touched."""

    def _rollout(self, brushed=True):
        interacted = [
            {"key": TUNA, "kind": "actor", "name": "env-0_007_tuna_fish_can-0",
             "grasped": True, "max_ee_force": 3.0},
        ]
        extents = {TUNA: _extent([0.04, 0.04, 0.05]),
                   COUNTER: _extent([0.95, 0.42, 0.45])}
        entities = {
            TUNA: {"pose": [0.0, 0.0, 0.46, 1.0, 0.0, 0.0, 0.0],
                   "kind": "actor", "name": "env-0_007_tuna_fish_can-0"},
            COUNTER: {"pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                      "kind": "link", "name": "env-0_body"},
        }
        if brushed:
            interacted += [
                {"key": SOFA, "kind": "actor",
                 "name": "env-0_frl_apartment_sofa-0", "grasped": False,
                 "max_ee_force": 1.4},
                {"key": DOOR, "kind": "link", "name": "env-0_top_door",
                 "grasped": False, "max_ee_force": 0.9},
            ]
            extents[SOFA] = _extent([0.90, 0.45, 0.40])
            # The door's collision geometry never read. It is not a supporter,
            # so the extent gate does not reach it either way.
            extents[DOOR] = {"extent_status": "no-collision-shapes"}
            # The sofa sits well below the gripper's whole sweep, so its
            # end-effector offsets (1.18m..1.80m) fall outside every offset
            # the target produces (0.02m..0.64m). One leaked sample would
            # move a family quantile visibly, which is what the calibration
            # test below reads. Both stay near the target in plan, so the
            # object-object statistics -- which are mined over every recorded
            # entity, membership or not -- keep their usual magnitudes.
            entities[SOFA] = {"pose": [0.10, 0.10, -0.70, 1.0, 0.0, 0.0, 0.0],
                              "kind": "actor",
                              "name": "env-0_frl_apartment_sofa-0"}
            entities[DOOR] = {"pose": [0.10, -0.10, 0.75, 1.0, 0.0, 0.0, 0.0],
                              "kind": "link", "name": "env-0_top_door"}
        return {
            "target_key": TUNA,
            "interacted": interacted,
            "supports": [
                {"supported_key": TUNA,
                 "supporter": {"key": COUNTER, "kind": "link",
                               "name": "env-0_body"}},
            ],
            "extents": extents,
            "pose_samples": [
                {"tcp_pose": [0.0, 0.0, z] + [1.0, 0.0, 0.0, 0.0],
                 "entities": {k: dict(v) for k, v in entities.items()}}
                for z in (1.10, 1.00, 0.90, 0.80, 0.66, 0.55, 0.48)
            ],
            "bin_samples": {
                "ee_object_planar_distance": [0.4, 0.2],
                "ee_object_height_offset": [0.02, 0.55],
            },
        }

    def _payload(self, policy=MEMBERSHIP_FULL_EVIDENCE, brushed=True,
                 rollout=None):
        builder = _WhitelistBuilder(
            "pick", TUNA, task_group="tidy_house",
            membership_policy=policy, affordance_set=_affordances())
        builder.absorb(rollout if rollout is not None
                       else self._rollout(brushed))
        return builder.payload()


class RawEvidenceIsPreservedTest(_Fixture):
    """The mine finishes, and nothing was invented to make it finish."""

    def test_the_mine_no_longer_stops(self):
        self._payload()      # the failure this replaces was a ValueError here

    def test_the_brushed_past_furniture_is_still_in_the_evidence(self):
        members = self._payload()["members"]
        self.assertIn(SOFA, members)
        self.assertIn(DOOR, members)

    def test_neither_was_given_a_family(self):
        """A fallback family is the same bug wearing a different token."""
        members = self._payload()["members"]
        self.assertIsNone(members[SOFA].get("family"))
        self.assertIsNone(members[DOOR].get("family"))

    def test_each_carries_the_reason_nothing_classified_it(self):
        members = self._payload()["members"]
        for key in (SOFA, DOOR):
            with self.subTest(member=key):
                self.assertIn(rules.UNRESOLVED_FIELD, members[key])
                self.assertIn("height-family",
                              members[key][rules.UNRESOLVED_FIELD])

    def test_the_payload_indexes_them_in_one_place(self):
        self.assertEqual(sorted(self._payload()["_unresolved_members"]),
                         sorted([DOOR, SOFA]))

    def test_the_target_and_its_counter_are_classified_as_before(self):
        members = self._payload()["members"]
        self.assertEqual(members[TUNA]["family"], FAMILY_MANIPULAND)
        self.assertEqual(members[COUNTER]["family"], FAMILY_STRUCTURAL)
        self.assertTrue(members[COUNTER]["structural_surface"])

    def test_a_clean_rollout_records_nothing_unresolved(self):
        """The marking appears because of the evidence, not by default."""
        payload = self._payload(brushed=False)
        self.assertNotIn("_unresolved_members", payload)
        self.assertNotIn(rules.UNRESOLVED_FIELD, payload["members"][TUNA])


class UnresolvedMembersDoNotCalibrateTest(_Fixture):
    """An unresolved member has no scale, so it contributes to none.

    Scoped to the per-family end-effector scales on purpose. The
    object-object statistics are mined from the pose trace over every entity
    the collector recorded, membership or not, and that is unchanged here.
    """

    def _family_stats(self, **kwargs):
        stats = self._payload(**kwargs)["bin_stats_robust"]
        return {k: v for k, v in stats.items() if k.startswith("ee_")
                and any(k.startswith(f) for f in FAMILY_STATS)}

    def test_the_family_scales_are_identical_with_and_without_them(self):
        """The sofa sits 4m below the gripper and the door 5m above it, so a
        single leaked sample would move the quantile by metres."""
        self.assertEqual(self._family_stats(brushed=True),
                         self._family_stats(brushed=False))

    def test_those_scales_were_actually_calibrated(self):
        """Otherwise the equality above would hold by both being empty."""
        self.assertTrue(self._family_stats(brushed=True))

    def test_the_sampler_is_never_handed_an_unresolved_member(self):
        builder = _WhitelistBuilder(
            "pick", TUNA, task_group="tidy_house",
            membership_policy=MEMBERSHIP_FULL_EVIDENCE,
            affordance_set=_affordances())
        builder.absorb(self._rollout())
        seen = {}

        def spy(families):
            seen.update(families)
            return {}

        builder._mine_family_height_samples = spy
        builder.payload()
        self.assertNotIn(SOFA, seen)
        self.assertNotIn(DOOR, seen)
        self.assertEqual(seen.get(TUNA), FAMILY_MANIPULAND)


class PruningProducesAValidRuntimeAssetTest(_Fixture):
    """End to end: evidence in, runtime asset out, and it validates."""

    def _pruned(self):
        return prune_payload(self._payload(), MEMBERSHIP_TARGET_SUPPORTERS)

    def test_the_furniture_is_gone(self):
        members = self._pruned()["members"]
        self.assertNotIn(SOFA, members)
        self.assertNotIn(DOOR, members)

    def test_the_target_and_its_supporter_remain(self):
        self.assertEqual(sorted(self._pruned()["members"]),
                         sorted([COUNTER, TUNA]))

    def test_nothing_in_it_is_unresolved(self):
        self.assertEqual(rules.runtime_blockers(self._pruned()["members"]), {})

    def test_the_stale_index_does_not_survive_the_prune(self):
        """It named members this file no longer has."""
        self.assertNotIn("_unresolved_members", self._pruned())

    def test_the_written_asset_passes_the_final_gate(self):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: None)
        (directory / "pick_007_tuna_fish_can.json").write_text(
            json.dumps(self._pruned()))
        self.assertEqual(_unmeasurable_members(directory), [])

    def test_the_raw_evidence_still_holds_what_the_prune_dropped(self):
        """Pruning reads the raw payload; it must not edit it."""
        raw = self._payload()
        prune_payload(raw, MEMBERSHIP_TARGET_SUPPORTERS)
        self.assertIn(SOFA, raw["members"])
        self.assertIn(DOOR, raw["members"])


class RuntimeAdmissionStillRefusesTest(_Fixture):
    """The negative half: an unresolved member that survives must fail.

    Without this, "raw evidence is allowed to carry unresolved members" is one
    edit away from "unresolved members are allowed", and the ~0.9m origin
    error walks back in through the stage that was supposed to stop it.
    """

    def _unreadable_supporter(self):
        """The counter's collision geometry never read.

        It supports the target, so ``target-supporters`` keeps it -- which is
        what makes it the case the raw/runtime split has to get right.
        """
        rollout = self._rollout()
        rollout["extents"][COUNTER] = {"extent_status": "no-collision-shapes"}
        return rollout

    def test_the_raw_mine_records_it_rather_than_refusing(self):
        payload = self._payload(rollout=self._unreadable_supporter())
        self.assertIn("collision extent",
                      payload["members"][COUNTER][rules.UNRESOLVED_FIELD])

    def test_pruning_it_into_a_runtime_asset_refuses(self):
        raw = self._payload(rollout=self._unreadable_supporter())
        with self.assertRaises(ValueError) as ctx:
            prune_payload(raw, MEMBERSHIP_TARGET_SUPPORTERS)
        self.assertIn(COUNTER, str(ctx.exception))

    def test_copying_it_through_unpruned_refuses_too(self):
        """``full-evidence`` names the membership rule, not an exemption: the
        file still lands in the directory a run loads from."""
        raw = self._payload(rollout=self._unreadable_supporter())
        with self.assertRaises(ValueError):
            prune_payload(raw, MEMBERSHIP_FULL_EVIDENCE)

    def test_mining_the_runtime_shape_directly_refuses(self):
        """The miner can write that shape itself, and must answer the same."""
        with self.assertRaises(ValueError) as ctx:
            self._payload(policy=MEMBERSHIP_TARGET_SUPPORTERS,
                          rollout=self._unreadable_supporter())
        self.assertIn("collision extent", str(ctx.exception))

    def test_the_runtime_policy_never_admitted_the_furniture(self):
        """Which is why mining that shape from the *good* rollout passes: the
        sofa is not rejected there, it was never a member."""
        members = self._payload(policy=MEMBERSHIP_TARGET_SUPPORTERS)["members"]
        self.assertEqual(sorted(members), sorted([COUNTER, TUNA]))

    def test_a_flagged_member_is_a_blocker_however_it_got_there(self):
        """Read off the file, so a hand-edited asset answers the same."""
        members = {
            TUNA: {"roles": ["interacted"], "interaction_types": ["grasp"],
                   "family": FAMILY_MANIPULAND},
            SOFA: {"roles": ["interacted"], "interaction_types": ["contact"],
                   rules.UNRESOLVED_FIELD: rules.UNRESOLVED_NO_FAMILY},
        }
        self.assertEqual(sorted(rules.runtime_blockers(members)), [SOFA])

    def test_a_families_aware_asset_may_not_omit_one(self):
        """No flag at all, just a missing family in an asset that classifies
        everything else -- the shape a hand edit produces."""
        members = {
            TUNA: {"roles": ["interacted"], "interaction_types": ["grasp"],
                   "family": FAMILY_MANIPULAND},
            SOFA: {"roles": ["interacted"], "interaction_types": ["contact"]},
        }
        self.assertEqual(sorted(rules.runtime_blockers(members)), [SOFA])

    def test_a_legacy_asset_that_classifies_nothing_is_left_alone(self):
        """It keeps the single shared scale, exactly as the runtime does."""
        members = {
            TUNA: {"roles": ["interacted"], "interaction_types": ["grasp"]},
            COUNTER: {"roles": ["support"], "interaction_types": ["support"]},
        }
        self.assertEqual(rules.runtime_blockers(members), {})

    def test_a_declared_site_is_not_expected_to_carry_a_family(self):
        """It has no body to be above; it is measured on the ee-site scales."""
        members = {
            TUNA: {"roles": ["interacted"], "interaction_types": ["grasp"],
                   "family": FAMILY_MANIPULAND},
            "spatial:ee_rest_site": {"roles": ["spatial"],
                                     "interaction_types": []},
        }
        self.assertEqual(rules.runtime_blockers(members), {})


class TheRuntimeLoaderRefusesRawAssetsTest(_Fixture):
    """Last line, and the only one that sees what a run actually loaded.

    ``whitelists.dir`` is a path in a config file, and
    ``subtask_whitelists_raw`` differs from ``subtask_whitelists`` by four
    characters. Pointed at the wrong one, everything parses.
    """

    def _write(self, payload):
        directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, directory, ignore_errors=True)
        path = directory / "pick_007_tuna_fish_can.json"
        path.write_text(json.dumps(payload))
        return str(path)

    def test_loading_a_raw_asset_refuses(self):
        path = self._write(self._payload())
        with self.assertRaises(ValueError) as ctx:
            load_whitelist(path)
        self.assertIn(SOFA, str(ctx.exception))

    def test_the_refusal_says_which_directory_to_use(self):
        path = self._write(self._payload())
        with self.assertRaises(ValueError) as ctx:
            load_whitelist(path)
        self.assertIn("pruned", str(ctx.exception))

    def test_the_pruned_asset_loads(self):
        path = self._write(
            prune_payload(self._payload(), MEMBERSHIP_TARGET_SUPPORTERS))
        whitelist = load_whitelist(path)
        self.assertEqual(whitelist.families[TUNA], FAMILY_MANIPULAND)
        self.assertEqual(whitelist.families[COUNTER], FAMILY_STRUCTURAL)

    def test_a_raw_asset_with_nothing_unresolved_loads(self):
        """The mark is what is refused, not the membership policy: a clean
        full-evidence file is still a valid whitelist."""
        path = self._write(self._payload(brushed=False))
        self.assertTrue(load_whitelist(path).families)


class PipelineOnDiskTest(_Fixture):
    """The same run the server makes: pickles in, runtime assets out.

    The payload-level tests above pin the rules; this one pins that the two
    command-line stages actually carry them, because the failure being fixed
    was a directory on disk holding four files of nine.
    """

    SECOND = "actor:003_cracker_box"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.success = self.tmp / "robot_success_states"
        self.raw = self.tmp / "raw" / "tidy_house"
        self.runtime = self.tmp / "runtime" / "tidy_house"
        self.affordances = self.tmp / "affordances.json"
        self.affordances.write_text(json.dumps(_affordance_payload()))

    def _second_target(self):
        """A second object in the same scene, so a refusal has something to
        take down with it."""
        rollout = self._rollout(brushed=False)
        rollout["target_key"] = self.SECOND
        rollout["interacted"][0]["key"] = self.SECOND
        rollout["supports"][0]["supported_key"] = self.SECOND
        rollout["extents"][self.SECOND] = rollout["extents"].pop(TUNA)
        for snap in rollout["pose_samples"]:
            snap["entities"][self.SECOND] = snap["entities"].pop(TUNA)
        return rollout

    def _pickle(self, rollouts, name="tuna.pkl"):
        path = self.success / "fetch" / "tidy_house" / "pick" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as stream:
            pickle.dump({
                "_schema_version": MIN_ROLLOUT_SCHEMA,
                "subtask_type": "pick",
                "entity_key": TUNA,
                "provenance": {"task_group": "tidy_house"},
                "interaction_rollouts": rollouts,
            }, stream)
        return path

    def _mine(self, policy=MEMBERSHIP_FULL_EVIDENCE, expect=()):
        argv = [
            "--success-states-dir", str(self.success),
            "--task-group", "tidy_house",
            "--membership-policy", policy,
            "--out-dir", str(self.raw),
            "--affordance-json", str(self.affordances),
        ]
        if expect:
            argv += ["--expect-targets", *expect]
        return build_subtask_whitelists.main(argv)

    def _prune(self):
        return prune_whitelists.main([
            "--raw-dir", str(self.raw), "--out-dir", str(self.runtime),
            "--task-group", "tidy_house", "--subtask", "pick",
        ])

    def _files(self, directory):
        return sorted(p.name for p in directory.glob("*.json"))

    def test_the_whole_pipeline_runs(self):
        self._pickle([self._rollout(), self._second_target()])
        self.assertEqual(self._mine(), 0)
        self.assertEqual(
            self._files(self.raw),
            ["pick_003_cracker_box.json", "pick_007_tuna_fish_can.json"])
        self.assertEqual(self._prune(), 0)
        self.assertEqual(
            self._files(self.runtime),
            ["pick_003_cracker_box.json", "pick_007_tuna_fish_can.json",
             "pick_all.json"])

    def test_the_runtime_assets_validate(self):
        self._pickle([self._rollout(), self._second_target()])
        self._mine()
        self._prune()
        self.assertEqual(_unmeasurable_members(self.runtime), [])

    def test_the_furniture_is_in_the_raw_files_and_not_the_runtime_ones(self):
        self._pickle([self._rollout(), self._second_target()])
        self._mine()
        self._prune()
        name = "pick_007_tuna_fish_can.json"
        raw = json.loads((self.raw / name).read_text())["members"]
        runtime = json.loads((self.runtime / name).read_text())["members"]
        self.assertIn(SOFA, raw)
        self.assertIn(DOOR, raw)
        self.assertNotIn(SOFA, runtime)
        self.assertNotIn(DOOR, runtime)

    def test_the_declared_rest_site_survives_into_the_runtime_asset(self):
        """It carries no family and never needed one; the gate must not read
        it as an omission."""
        self._pickle([self._rollout()])
        self._mine()
        self._prune()
        members = json.loads(
            (self.runtime / "pick_007_tuna_fish_can.json").read_text())
        self.assertIn(SITE_EE_REST, members["members"])

    def test_a_refused_mine_writes_nothing_at_all(self):
        """The failure this replaces: four of nine targets on disk, looking
        exactly like a finished mine. One target that cannot be written now
        takes the whole run with it, so what is there is always all of it."""
        rollout = self._rollout()
        rollout["extents"][COUNTER] = {"extent_status": "no-collision-shapes"}
        self._pickle([rollout, self._second_target()])
        with self.assertRaises(ValueError):
            self._mine(MEMBERSHIP_TARGET_SUPPORTERS)
        self.assertEqual(self._files(self.raw), [])

    def test_the_raw_union_keeps_the_mark(self):
        """``pick_all.json`` is where the runtime reads its bins and its
        vocabulary, so a union that dropped the mark would be the one file
        able to load a raw tree as if it were pruned."""
        self._pickle([self._rollout()])
        self._mine()
        build_union_whitelist.main([
            "--whitelist-dir", str(self.raw), "--subtask", "pick"])
        union = json.loads((self.raw / "pick_all.json").read_text())
        self.assertEqual(sorted(union["_unresolved_members"]),
                         sorted([DOOR, SOFA]))
        with self.assertRaises(ValueError):
            load_whitelist(str(self.raw / "pick_all.json"))

    def test_the_pruned_union_is_clean_and_loads(self):
        self._pickle([self._rollout()])
        self._mine()
        self._prune()
        union = self.runtime / "pick_all.json"
        self.assertNotIn("_unresolved_members",
                         json.loads(union.read_text()))
        self.assertTrue(load_whitelist(str(union)).families)

    def test_the_expected_targets_pass_when_they_are_all_there(self):
        self._pickle([self._rollout(), self._second_target()])
        self.assertEqual(
            self._mine(expect=["007_tuna_fish_can", "003_cracker_box"]), 0)

    def test_a_missing_target_stops_the_mine_before_it_writes(self):
        """The other shape of the same failure: not a run that stopped
        halfway, but one that finished over the wrong set. Both leave a
        directory that looks complete."""
        self._pickle([self._rollout()])
        self.assertEqual(
            self._mine(expect=["007_tuna_fish_can", "003_cracker_box"]), 2)
        self.assertEqual(self._files(self.raw), [])

    def test_a_target_nobody_asked_for_stops_it_too(self):
        self._pickle([self._rollout(), self._second_target()])
        self.assertEqual(self._mine(expect=["007_tuna_fish_can"]), 2)
        self.assertEqual(self._files(self.raw), [])

    def test_canonical_keys_are_accepted_as_well_as_bare_names(self):
        self._pickle([self._rollout()])
        self.assertEqual(self._mine(expect=[TUNA]), 0)

    def test_a_refused_prune_writes_nothing_at_all(self):
        rollout = self._rollout()
        rollout["extents"][COUNTER] = {"extent_status": "no-collision-shapes"}
        self._pickle([rollout, self._second_target()])
        self.assertEqual(self._mine(), 0)
        self.assertEqual(len(self._files(self.raw)), 2)
        self.assertEqual(self._prune(), 1)
        self.assertEqual(self._files(self.runtime), [])


if __name__ == "__main__":
    unittest.main()
