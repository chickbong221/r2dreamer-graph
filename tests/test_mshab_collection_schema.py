"""What the MS-HAB collector records, and which scenes it records it from.

Two separable things, both settled before any sim-hours are spent:

* **Scene pinning.** A task plan file holds plans from many build
  configurations and the collector cycles them across the vector environments.
  Experiment B trains on one scene, so the evidence has to come from one --
  and provenance recorded afterwards can only report that it did not.
* **The evidence itself.** Collision extents, the end-effector rest
  calibration and the per-rollout build configuration are newly *recorded*.
  No re-mining can recover them from a pickle that never held them, so every
  reader that could accept an older one has to stop.

``collect_contact_data`` imports gymnasium and sapien, so its module-level
helpers are exec'd from source the way ``test_maniskill_env_branch`` does, and
the wrapper's own internals are checked against that source.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np

from scenegraph.core.entity_identity import (
    _articulation,
    entity_kind,
    entity_name,
    stable_entity_key,
)
from scenegraph.tools.collect_robot_success_states import (
    REQUIRED_ROLLOUT_SCHEMA,
    assert_pinned_build_config,
    build_config_names,
    select_build_config_plans,
)

WRAPPER = Path("scenegraph/adapters/collect_contact_data.py")
SOURCE = WRAPPER.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _load(*names):
    """Exec selected top-level definitions without importing the module."""
    wanted, out = set(names), {
        "stable_entity_key": stable_entity_key, "entity_name": entity_name,
        "entity_kind": entity_kind, "_articulation": _articulation,
        "Optional": Optional, "Dict": Dict, "Any": Any, "np": np,
    }
    body = [
        item for item in TREE.body
        if (getattr(item, "name", None) in wanted
            or (isinstance(item, ast.Assign)
                and any(getattr(t, "id", "") in wanted for t in item.targets)))
    ]
    exec(compile(ast.Module(body=body, type_ignores=[]), "<sel>", "exec"), out)
    return SimpleNamespace(**out)


mod = _load("_record", "_plan_identity", "_live_plan", "_to_np",
            "_SCHEMA_VERSION")


def _plan(build_config, init_config="init-0"):
    return SimpleNamespace(build_config_name=build_config,
                           init_config_name=init_config)


PLANS = [
    _plan("apt-0", "init-0"), _plan("apt-0", "init-1"),
    _plan("apt-1", "init-0"), _plan("apt-2", "init-0"),
]


class BuildConfigSelectionTest(unittest.TestCase):
    """One collection, one scene -- when a scene is asked for."""

    def test_it_enumerates_the_configurations_a_plan_file_offers(self):
        self.assertEqual(build_config_names(PLANS), ["apt-0", "apt-1", "apt-2"])

    def test_pinning_keeps_only_that_configuration(self):
        kept = select_build_config_plans(PLANS, "apt-0", "plan.json")
        self.assertEqual(len(kept), 2)
        self.assertEqual({p.build_config_name for p in kept}, {"apt-0"})

    def test_pinning_keeps_the_init_variety_within_the_scene(self):
        """One scene, many spawns: the diversity that is still wanted."""
        kept = select_build_config_plans(PLANS, "apt-0", "plan.json")
        self.assertEqual({p.init_config_name for p in kept},
                         {"init-0", "init-1"})

    def test_an_unmatched_configuration_stops_the_run(self):
        with self.assertRaises(SystemExit) as ctx:
            select_build_config_plans(PLANS, "apt-9", "plan.json")
        message = str(ctx.exception)
        self.assertIn("apt-9", message)
        self.assertIn("apt-0", message)

    def test_no_pin_keeps_every_plan(self):
        """Unpinned collection stays legal: a whitelist meant to cover a whole
        split wants exactly that."""
        self.assertEqual(len(select_build_config_plans(PLANS, "", "p.json")),
                         len(PLANS))

    def test_no_pin_over_several_configurations_says_so(self):
        import io
        import contextlib
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            select_build_config_plans(PLANS, "", "p.json")
        printed = buffer.getvalue()
        self.assertIn("NO --build-config", printed)
        self.assertIn("apt-1", printed)

    def test_a_single_configuration_file_needs_no_warning(self):
        import io
        import contextlib
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            select_build_config_plans([_plan("apt-0")], "", "p.json")
        self.assertNotIn("NO --build-config", buffer.getvalue())


class PostResetAssertionTest(unittest.TestCase):
    """Filtering selects plans; this checks what was actually built."""

    def _venv(self, *configs):
        return SimpleNamespace(unwrapped=SimpleNamespace(
            bc_to_task_plans={c: [] for c in configs}))

    def test_the_requested_configuration_passes(self):
        assert_pinned_build_config(self._venv("apt-0"), "apt-0")

    def test_a_mixed_environment_stops_the_run(self):
        with self.assertRaises(SystemExit) as ctx:
            assert_pinned_build_config(self._venv("apt-0", "apt-1"), "apt-0")
        self.assertIn("apt-1", str(ctx.exception))

    def test_a_different_configuration_stops_the_run(self):
        with self.assertRaises(SystemExit):
            assert_pinned_build_config(self._venv("apt-1"), "apt-0")

    def test_an_unpinned_run_is_not_checked(self):
        assert_pinned_build_config(self._venv("apt-0", "apt-1"), "")


class WrapperGuaranteeTest(unittest.TestCase):
    """The wrapper refuses a mixed environment rather than recording that it
    was mixed. Source-level: constructing it needs a simulator."""

    def _ctor(self):
        for node in ast.walk(TREE):
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                return ast.get_source_segment(SOURCE, node)
        raise AssertionError("no wrapper __init__")

    def test_it_takes_a_required_build_config(self):
        self.assertIn("require_build_config", self._ctor())

    def test_it_raises_when_the_environment_holds_another_configuration(self):
        body = self._ctor()
        self.assertIn("self.build_configs != [wanted]", body)
        self.assertIn("raise ValueError", body)

    def test_it_reads_the_configurations_from_the_environment_itself(self):
        """``bc_to_task_plans`` is what was built, not what was requested."""
        self.assertIn("base.bc_to_task_plans", self._ctor())

    def test_it_takes_the_per_environment_plan_list(self):
        self.assertIn("env_plans", self._ctor())


class EntityIdentityTest(unittest.TestCase):
    """Canonicalisation is not reversible, so the raw names travel too."""

    def test_the_canonical_key_and_the_raw_name_are_both_kept(self):
        ent = SimpleNamespace(name="env-0_scs-[2,3]_frl_apartment_table_01-0")
        record = mod._record(ent, key="actor:frl_apartment_table_01")
        self.assertEqual(record["key"], "actor:frl_apartment_table_01")
        self.assertEqual(record["name"],
                         "env-0_scs-[2,3]_frl_apartment_table_01-0")

    def test_a_link_carries_its_raw_articulation_name(self):
        """The suffix audit asks whether one logical counter is numbered
        differently across build configurations. Only the raw name can say."""
        art = SimpleNamespace(name="env-0_kitchen_counter-1")
        link = SimpleNamespace(name="env-0_drawer3", articulation=art)
        record = mod._record(link, key="link:kitchen_counter-1/drawer3")
        self.assertEqual(record["raw_articulation"], "env-0_kitchen_counter-1")

    def test_an_actor_has_no_articulation_field(self):
        record = mod._record(SimpleNamespace(name="apple-0"),
                             key="actor:013_apple")
        self.assertNotIn("raw_articulation", record)

    def test_an_unkeyable_entity_records_nothing(self):
        """A record with no key would be a member nothing can look up."""
        self.assertIsNone(mod._record(None))


class PlanIdentityTest(unittest.TestCase):

    def test_it_carries_the_scene_and_the_arrangement(self):
        identity = mod._plan_identity(_plan("apt-3", "init-7"), 5)
        self.assertEqual(identity["build_config_name"], "apt-3")
        self.assertEqual(identity["init_config_name"], "init-7")
        self.assertEqual(identity["task_plan_index"], 5)

    def test_a_plan_missing_its_names_yields_empty_strings_not_none(self):
        identity = mod._plan_identity(SimpleNamespace(), 0)
        self.assertEqual(identity["build_config_name"], "")
        self.assertEqual(identity["init_config_name"], "")


class LivePlanTest(unittest.TestCase):
    """Which plan a sub-scene is running *now*.

    MS-HAB resamples ``task_plan_idxs`` on every reset and resolves the active
    plan through ``build_config_idx_to_task_plans``. The list handed to
    ``gym.make`` describes what each vector slot started with, so recording
    that would attribute every episode after the first to the wrong
    arrangement -- and do it silently, since both are real plans.
    """

    def _base(self, build_idxs, plan_idxs):
        return SimpleNamespace(
            build_config_idxs=list(build_idxs),
            task_plan_idxs=list(plan_idxs),
            build_config_idx_to_task_plans={
                0: [_plan("apt-0", "init-0"), _plan("apt-0", "init-1")],
                1: [_plan("apt-1", "init-0")],
            },
        )

    def test_it_resolves_the_plan_the_indices_point_at(self):
        plan, bci, tpi = mod._live_plan(self._base([0, 0], [0, 1]), 1)
        self.assertEqual(plan.init_config_name, "init-1")
        self.assertEqual((bci, tpi), (0, 1))

    def test_each_sub_scene_resolves_independently(self):
        base = self._base([0, 1], [1, 0])
        self.assertEqual(mod._live_plan(base, 0)[0].build_config_name, "apt-0")
        self.assertEqual(mod._live_plan(base, 1)[0].build_config_name, "apt-1")

    def test_a_resampled_index_changes_the_answer(self):
        """The regression itself: the same slot, two episodes, two plans."""
        first = mod._live_plan(self._base([0, 0], [0, 0]), 0)
        base = self._base([0, 0], [0, 0])
        base.task_plan_idxs[0] = 1          # MS-HAB resamples on reset
        second = mod._live_plan(base, 0)
        self.assertEqual(first[0].init_config_name, "init-0")
        self.assertEqual(second[0].init_config_name, "init-1")
        self.assertNotEqual(first[2], second[2])

    def test_an_environment_without_the_indices_reports_nothing(self):
        """None, so the caller can say 'unavailable' instead of substituting
        a value it already knows is stale."""
        self.assertEqual(mod._live_plan(SimpleNamespace(), 0),
                         (None, None, None))

    def test_an_out_of_range_index_reports_nothing(self):
        self.assertEqual(mod._live_plan(self._base([1], [7]), 0),
                         (None, None, None))


class RecordedEvidenceTest(unittest.TestCase):
    """What the new schema must actually write, checked against the source."""

    def _method(self, name):
        for node in ast.walk(TREE):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return ast.get_source_segment(SOURCE, node)
        raise AssertionError(f"{name} is not defined in {WRAPPER}")

    def test_a_rollout_carries_provenance_extents_and_rest_samples(self):
        body = self._method("commit_success")
        for field in ("provenance", "extents", "ee_rest_samples"):
            with self.subTest(field=field):
                self.assertIn(f'"{field}"', body)

    def test_provenance_names_the_scene_the_environment_and_the_split(self):
        body = self._method("_rollout_provenance")
        for field in ("env_idx", "split", "env_build_configs",
                      "requested_build_config", "target_resolution"):
            with self.subTest(field=field):
                self.assertIn(f'"{field}"', body)

    def test_provenance_resolves_the_plan_live(self):
        """Not from ``env_plan_identity``, which is the construction-time
        assignment and stops describing the slot after the first autoreset."""
        body = self._method("_rollout_provenance")
        self.assertIn("_live_plan(self._base_env, env_idx)", body)
        self.assertIn('"plan_source"', body)

    def test_a_stale_assignment_never_poses_as_the_live_plan(self):
        body = self._method("_rollout_provenance")
        self.assertIn('"assigned_at_construction"', body)
        self.assertIn('"unavailable"', body)

    def test_provenance_reports_whether_resolution_fell_back(self):
        """Whether the protected-target path can make the fallback fatal."""
        self.assertIn('"merged"', self._method("_rollout_provenance"))
        self.assertIn("_episode_merged_fallback",
                      self._method("_observe_step"))

    def test_extents_keep_the_reason_a_read_failed(self):
        """"No extent" and "small" have to stay distinguishable: assuming
        small is how a metre-wide counter gets measured from its own origin."""
        body = self._method("_observe_extents")
        self.assertIn("extent_status", body)
        self.assertIn("collision_half_extents_status", body)

    def test_extents_are_read_once_per_entity_not_once_per_frame(self):
        self.assertIn("if key in store:", self._method("_observe_extents"))

    def test_the_rest_samples_use_the_dedicated_scale(self):
        body = self._method("_observe_ee_rest")
        self.assertIn("EE_SITE_SCOPE", body)
        self.assertNotIn("EE_OBJECT_SCOPE", body)

    def test_the_rest_samples_come_from_the_shared_helper(self):
        """Same reader as the runtime provider, so the mined scale and the
        labelled distance measure the same point."""
        self.assertIn("ee_rest_geometry(self, env_idx)",
                      self._method("_observe_ee_rest"))

    def test_the_rest_sampling_is_restricted_to_pick(self):
        """``pick_cfg.ee_rest_thresh`` is that subtask's own geometry."""
        self.assertIn('self.subtask_type != "pick"',
                      self._method("_observe_ee_rest"))

    def test_the_exact_predicate_and_tolerance_travel_with_the_distance(self):
        body = self._method("_observe_ee_rest")
        for field in ("euclidean_distance", "tolerance", "reached"):
            with self.subTest(field=field):
                self.assertIn(f'"{field}"', body)

    def test_geometry_and_rest_buffers_are_cleared_on_a_boundary(self):
        """The scene may be rebuilt and the robot is re-placed, so a read held
        over measures the previous episode."""
        body = self._method("_reset_buffers")
        for buf in ("_episode_extents", "_episode_ee_rest",
                    "_episode_merged_fallback"):
            with self.subTest(buffer=buf):
                self.assertIn(buf, body)


class SchemaVersionTest(unittest.TestCase):
    """Three readers describe the same evidence. A floor left behind lets a
    stale pickle satisfy a recollection it cannot actually satisfy."""

    def test_the_collector_writes_the_new_version(self):
        self.assertEqual(mod._SCHEMA_VERSION, 9)

    def test_skip_done_requires_what_the_collector_writes(self):
        self.assertEqual(REQUIRED_ROLLOUT_SCHEMA, mod._SCHEMA_VERSION)

    def test_the_miner_requires_what_the_collector_writes(self):
        from scenegraph.tools.build_subtask_whitelists import (
            MIN_ROLLOUT_SCHEMA,
        )
        self.assertEqual(MIN_ROLLOUT_SCHEMA, mod._SCHEMA_VERSION)

    def test_an_older_pickle_is_not_treated_as_already_collected(self):
        import pickle
        import tempfile
        from scenegraph.tools.collect_robot_success_states import _is_complete
        payload = {
            "_schema_version": 8,
            "robot_qpos": [[0.0]] * 30,
            "tcp_pose_wrt_base": [[0.0]] * 30,
            "interaction_rollouts": [{}] * 30,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "024_bowl.pkl"
            with open(path, "wb") as handle:
                pickle.dump(payload, handle)
            self.assertFalse(_is_complete(path, 30))
            payload["_schema_version"] = 9
            with open(path, "wb") as handle:
                pickle.dump(payload, handle)
            self.assertTrue(_is_complete(path, 30))


if __name__ == "__main__":
    unittest.main()
