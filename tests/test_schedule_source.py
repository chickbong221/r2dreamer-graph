"""Where a run finds its schedule and its mined assets.

Two naming schemes share one compiler. An ordinary ManiSkill task is named
once, by its gym id, which is simultaneously the schedule file, the affordance
file and the whitelist directory. MS-HAB is named by a task group and a
subtask: its gym id (``PickSubtaskTrain-v0``) appears in no asset path, its
assets live under the group (``set_table``), and one group holds several
subtasks side by side -- so its union whitelist is ``pick_all.json``, never
``task_all.json``.

Making a single id stand for all three is why the MS-HAB lookup resolved none
of them. These pin both layouts, and pin that the ordinary one did not move.
"""

import os
import unittest

from scenegraph.core.schedule import (
    ScheduleError,
    ScheduleSource,
    compile_from_source,
    load_assets_from_source,
    maniskill_schedule_source,
    mshab_schedule_source,
)

CONFIGS = os.path.join("scenegraph", "configs")
SCHEDULES = os.path.join(CONFIGS, "schedules")
MANISKILL_TASKS = ("PickCube-v1", "PlaceSphere-v1", "PegInsertionSide-v1",
                   "PullCubeTool-v1")


class ManiskillLayoutTest(unittest.TestCase):
    """The pre-existing layout, spelled out so a refactor cannot drift it."""

    def _source(self, env_id="PickCube-v1"):
        return maniskill_schedule_source(env_id, CONFIGS, SCHEDULES)

    def test_the_gym_id_names_the_schedule(self):
        self.assertEqual(self._source().schedule_path,
                         os.path.join(SCHEDULES, "PickCube-v1.json"))

    def test_the_gym_id_names_the_affordance_file(self):
        self.assertEqual(
            self._source().affordance_path,
            os.path.join(CONFIGS, "affordances", "PickCube-v1.json"))

    def test_the_union_whitelist_is_still_task_all(self):
        """``task`` is the subtask slot an ordinary task files under and
        ``all`` the target slot; both are constants because it has one of
        each."""
        self.assertEqual(
            self._source().union_whitelist_path,
            os.path.join(CONFIGS, "subtask_whitelists", "PickCube-v1",
                         "task_all.json"))

    def test_every_shipped_task_resolves_to_files_that_exist(self):
        for env_id in MANISKILL_TASKS:
            with self.subTest(task=env_id):
                source = self._source(env_id)
                for path in (source.schedule_path, source.affordance_path,
                             source.union_whitelist_path):
                    self.assertTrue(os.path.isfile(path), path)

    def test_every_shipped_task_still_compiles_through_the_source(self):
        from scenegraph.adapters.graph_vocab import build_entity_vocab
        for env_id in MANISKILL_TASKS:
            with self.subTest(task=env_id):
                source = self._source(env_id)
                compile_from_source(
                    source, build_entity_vocab(source.whitelist_dir))

    def test_the_legacy_entry_point_compiles_the_same_schedule(self):
        """``compile_from_files`` is the call every probe and tool makes."""
        from scenegraph.adapters.graph_vocab import build_entity_vocab
        from scenegraph.core.schedule import compile_from_files
        for env_id in MANISKILL_TASKS:
            with self.subTest(task=env_id):
                source = self._source(env_id)
                legacy = compile_from_files(
                    env_id, SCHEDULES, CONFIGS,
                    build_entity_vocab(source.whitelist_dir))
                structured = compile_from_source(
                    source, build_entity_vocab(source.whitelist_dir))
                self.assertEqual(legacy, structured)

    def test_the_legacy_asset_loader_reads_the_same_files(self):
        from scenegraph.core.schedule import load_assets
        for env_id in MANISKILL_TASKS:
            with self.subTest(task=env_id):
                self.assertEqual(load_assets(env_id, CONFIGS),
                                 load_assets_from_source(self._source(env_id)))


class MshabLayoutTest(unittest.TestCase):
    """Group for the assets, subtask for the schedule and the union file."""

    def _source(self, group="set_table", subtask="pick"):
        return mshab_schedule_source(group, subtask, CONFIGS, SCHEDULES)

    def test_the_affordance_file_is_named_for_the_task_group(self):
        self.assertEqual(
            self._source().affordance_path,
            os.path.join(CONFIGS, "affordances", "set_table.json"))

    def test_the_union_whitelist_is_named_for_the_subtask(self):
        self.assertEqual(
            self._source().union_whitelist_path,
            os.path.join(CONFIGS, "subtask_whitelists", "set_table",
                         "pick_all.json"))

    def test_the_schedule_lives_under_the_group(self):
        self.assertEqual(
            self._source().schedule_path,
            os.path.join(SCHEDULES, "set_table", "pick.json"))

    def test_the_gym_id_appears_in_no_path(self):
        source = self._source()
        for path in (source.schedule_path, source.affordance_path,
                     source.union_whitelist_path, source.whitelist_dir):
            self.assertNotIn("SubtaskTrain", path)

    def test_two_subtasks_of_one_group_share_a_whitelist_directory(self):
        pick, place = self._source(), self._source(subtask="place")
        self.assertEqual(pick.whitelist_dir, place.whitelist_dir)
        self.assertEqual(pick.affordance_path, place.affordance_path)
        self.assertNotEqual(pick.union_whitelist_path,
                            place.union_whitelist_path)
        self.assertNotEqual(pick.schedule_path, place.schedule_path)

    def test_the_shipped_group_assets_resolve_to_files_that_exist(self):
        for group in ("set_table", "tidy_house"):
            with self.subTest(group=group):
                source = self._source(group)
                self.assertTrue(os.path.isfile(source.affordance_path))
                self.assertTrue(os.path.isfile(source.union_whitelist_path))

    def test_a_missing_half_of_the_pair_is_refused(self):
        for group, subtask in (("", "pick"), ("set_table", "")):
            with self.subTest(group=group, subtask=subtask):
                with self.assertRaises(ScheduleError):
                    mshab_schedule_source(group, subtask, CONFIGS, SCHEDULES)


class WhitelistDirectoryTest(unittest.TestCase):
    """The compiler must read the directory the graph adapter resolved.

    Roles compile to entity-vocabulary ids and the packer writes rows from the
    same vocabulary. Built from two different directories they would disagree
    silently: every phase would simply never resolve.
    """

    BUILDERS = (
        lambda d: maniskill_schedule_source(
            "PickCube-v1", CONFIGS, SCHEDULES, d),
        lambda d: mshab_schedule_source(
            "set_table", "pick", CONFIGS, SCHEDULES, d),
    )

    def test_an_explicit_directory_is_used_verbatim(self):
        wanted = os.path.join("elsewhere", "set_table")
        for i, build in enumerate(self.BUILDERS):
            with self.subTest(builder=i):
                self.assertEqual(build(wanted).whitelist_dir, wanted)

    def test_the_union_file_always_sits_inside_that_directory(self):
        for i, build in enumerate(self.BUILDERS):
            with self.subTest(builder=i):
                source = build(os.path.join("elsewhere", "grp"))
                self.assertEqual(os.path.dirname(source.union_whitelist_path),
                                 source.whitelist_dir)

    def test_the_affordance_name_matches_what_the_runtime_loader_picks(self):
        """The adapter already resolves an MS-HAB affordance file by task
        group. The compiler must land on the same file, or the run would
        label with one asset's components and score against another's."""
        from scenegraph.configs.loader import load_config
        for group in ("set_table", "tidy_house"):
            with self.subTest(group=group):
                runtime = load_config(None, task_group=group,
                                      require_assets=True)
                source = mshab_schedule_source(
                    group, "pick",
                    os.path.dirname(os.path.dirname(
                        runtime["whitelists"]["dir_abs"])),
                    SCHEDULES, runtime["whitelists"]["dir_abs"])
                self.assertEqual(
                    os.path.normpath(source.affordance_path),
                    os.path.normpath(
                        runtime["affordances"]["asset_path_abs"]))

    def test_the_union_name_matches_what_the_runtime_binder_asks_for(self):
        """GraphBuilder resolves its global bins through the same rule."""
        from scenegraph.core.graph_builder import (
            TASK_LEVEL_SUBTASK, TASK_LEVEL_TARGET,
        )
        from scenegraph.core.whitelist import whitelist_path
        source = maniskill_schedule_source("PickCube-v1", CONFIGS, SCHEDULES)
        self.assertEqual(
            source.union_whitelist_path,
            whitelist_path(source.whitelist_dir, TASK_LEVEL_SUBTASK,
                           TASK_LEVEL_TARGET))


class LoudFailureTest(unittest.TestCase):
    """A lookup that resolved nothing has to say which paths it tried."""

    def _absent(self, **kw):
        base = dict(
            label="grp/pick",
            schedule_path=os.path.join("no", "such", "pick.json"),
            affordance_path=os.path.join("no", "such", "grp.json"),
            union_whitelist_path=os.path.join("no", "such", "pick_all.json"),
            whitelist_dir=os.path.join("no", "such"),
        )
        base.update(kw)
        return ScheduleSource(**base)

    def test_a_missing_schedule_names_the_path_it_wanted(self):
        with self.assertRaises(ScheduleError) as ctx:
            compile_from_source(self._absent(), None)
        message = str(ctx.exception)
        self.assertIn(os.path.join("no", "such", "pick.json"), message)
        self.assertIn("grp/pick", message)

    def test_a_missing_affordance_file_names_the_path(self):
        source = self._absent(
            schedule_path=os.path.join(SCHEDULES, "PickCube-v1.json"))
        with self.assertRaises(ScheduleError) as ctx:
            load_assets_from_source(source)
        self.assertIn(os.path.join("no", "such", "grp.json"),
                      str(ctx.exception))

    def test_a_missing_union_whitelist_names_the_path(self):
        source = self._absent(
            affordance_path=os.path.join(
                CONFIGS, "affordances", "PickCube-v1.json"))
        with self.assertRaises(ScheduleError) as ctx:
            load_assets_from_source(source)
        self.assertIn(os.path.join("no", "such", "pick_all.json"),
                      str(ctx.exception))

    def test_the_absent_mshab_schedule_is_reported_not_guessed(self):
        """No MS-HAB schedule is shipped yet. Asking for one must say so with
        the path rather than falling back to some other task's file."""
        from scenegraph.adapters.graph_vocab import build_entity_vocab
        source = mshab_schedule_source("set_table", "pick", CONFIGS, SCHEDULES)
        self.assertFalse(os.path.exists(source.schedule_path))
        with self.assertRaises(ScheduleError) as ctx:
            compile_from_source(
                source, build_entity_vocab(source.whitelist_dir))
        self.assertIn(source.schedule_path, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
