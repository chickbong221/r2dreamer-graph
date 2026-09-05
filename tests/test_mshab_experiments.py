"""Which objects and which scenes each experiment actually trains on.

Both selections used to be implicit and neither survives inspection:

* ``num_build_configs`` takes a sorted *prefix*, so "the first scene" moves
  when the build list does and is not an identity a result can be checked
  against. An experiment pinned to a scene has to name it.
* Concatenating five objects' plan files samples them in proportion to how
  many plans each happens to have -- 5,115 to 5,823 across tidy_house's pick
  targets -- so one object would get 14% more episodes for no reason anyone
  chose.

``envs/maniskill`` imports torch, so the selectors are exec'd from source the
way ``test_maniskill_env_branch`` does.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

SOURCE = Path("envs/maniskill.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)

FIVE = ["002_master_chef_can", "003_cracker_box", "004_sugar_box",
        "007_tuna_fish_can", "024_bowl"]
HELD_OUT = ["005_tomato_soup_can", "008_pudding_box", "009_gelatin_box",
            "010_potted_meat_can"]
SCENE = "v3_sc0_staging_00.scene_instance.json"


def _load(*names):
    wanted, out = set(names), {}
    body = [item for item in TREE.body
            if getattr(item, "name", None) in wanted]
    exec(compile(ast.Module(body=body, type_ignores=[]), "<sel>", "exec"), out)
    return SimpleNamespace(**out)


mod = _load("select_named_build_configs", "balance_objects",
            "_select_build_configs")


def _plan(config, init="i0"):
    return SimpleNamespace(build_config_name=config, init_config_name=init)


class NamedSceneTest(unittest.TestCase):
    """A scene named outright, not the first of a sorted list."""

    PLANS = [_plan(SCENE), _plan(SCENE, "i1"),
             _plan("v3_sc0_staging_01.scene_instance.json"),
             _plan("v3_sc2_staging_20.scene_instance.json")]

    def test_only_the_named_configuration_survives(self):
        kept = mod.select_named_build_configs(self.PLANS, [SCENE])
        self.assertEqual({p.build_config_name for p in kept}, {SCENE})
        self.assertEqual(len(kept), 2)

    def test_the_spawn_variety_inside_it_is_kept(self):
        """One scene, many arrangements: that is the variation the experiment
        wants."""
        kept = mod.select_named_build_configs(self.PLANS, [SCENE])
        self.assertEqual({p.init_config_name for p in kept}, {"i0", "i1"})

    def test_several_names_are_allowed(self):
        kept = mod.select_named_build_configs(
            self.PLANS, [SCENE, "v3_sc2_staging_20.scene_instance.json"])
        self.assertEqual(len(kept), 3)

    def test_a_name_that_does_not_exist_raises(self):
        """Silently training in a different apartment is the failure this
        removes."""
        with self.assertRaises(ValueError) as ctx:
            mod.select_named_build_configs(self.PLANS, ["v3_sc9_staging_99"])
        self.assertIn("v3_sc9_staging_99", str(ctx.exception))

    def test_an_empty_list_selects_everything(self):
        """So an unpinned run keeps its previous behaviour."""
        self.assertEqual(len(mod.select_named_build_configs(self.PLANS, [])),
                         len(self.PLANS))

    def test_it_is_not_the_sorted_prefix(self):
        """The two disagree whenever the wanted scene is not first, which is
        exactly when the distinction matters."""
        prefix = mod._select_build_configs(self.PLANS, 1)
        named = mod.select_named_build_configs(
            self.PLANS, ["v3_sc2_staging_20.scene_instance.json"])
        self.assertNotEqual({p.build_config_name for p in prefix},
                            {p.build_config_name for p in named})


class ObjectBalanceTest(unittest.TestCase):
    """Equal episodes per object, deterministically."""

    def _uneven(self):
        return {"a": [_plan(SCENE, f"a{i}") for i in range(10)],
                "b": [_plan(SCENE, f"b{i}") for i in range(4)],
                "c": [_plan(SCENE, f"c{i}") for i in range(7)]}

    def test_every_object_contributes_the_same_count(self):
        out = mod.balance_objects(self._uneven())
        counts = {}
        for plan in out:
            counts[plan.init_config_name[0]] = \
                counts.get(plan.init_config_name[0], 0) + 1
        self.assertEqual(set(counts.values()), {4})

    def test_the_total_is_the_smallest_times_the_object_count(self):
        self.assertEqual(len(mod.balance_objects(self._uneven())), 12)

    def test_it_is_deterministic(self):
        first = [p.init_config_name for p in mod.balance_objects(self._uneven())]
        second = [p.init_config_name for p in mod.balance_objects(self._uneven())]
        self.assertEqual(first, second)

    def test_already_equal_counts_are_untouched(self):
        even = {"a": [_plan(SCENE)] * 3, "b": [_plan(SCENE)] * 3}
        self.assertEqual(len(mod.balance_objects(even)), 6)

    def test_no_objects_yields_nothing(self):
        self.assertEqual(mod.balance_objects({}), [])


class ExperimentConfigTest(unittest.TestCase):
    """What the shipped configs actually say."""

    def _config(self, name):
        with open(f"configs/env/mshab_pick_{name}.yaml") as handle:
            return yaml.safe_load(handle)

    def test_a_names_the_five_training_objects(self):
        self.assertEqual(self._config("a")["mshab_objects"], FIVE)

    def test_a_holds_out_the_other_four(self):
        """They exist for transfer and must not be trained on."""
        named = set(self._config("a")["mshab_objects"])
        self.assertEqual(named & set(HELD_OUT), set())

    def test_a_trains_and_evaluates_in_one_named_scene(self):
        config = self._config("a")
        self.assertEqual(config["train_build_config_ids"], [SCENE])
        self.assertEqual(config["eval_build_config_ids"], [SCENE])

    def test_a_evaluates_five_episodes_per_object(self):
        config = self._config("a")
        self.assertEqual(config["eval_episode_num"],
                         5 * len(config["mshab_objects"]))

    def test_b_trains_one_object_in_the_same_scene(self):
        config = self._config("b")
        self.assertEqual(config["mshab_obj"], "004_sugar_box")
        self.assertEqual(config["mshab_objects"], [])
        self.assertEqual(config["train_build_config_ids"], [SCENE])

    def test_b_evaluates_over_every_configuration(self):
        """One environment per scene: reconfiguration_freq is 0, so a
        sub-scene keeps its configuration and the distinct-scene count is
        bounded by the environment count rather than the episode count."""
        config = self._config("b")
        self.assertEqual(config["eval_num_build_configs"], 0)
        self.assertEqual(config["eval_episode_num"], 63)
        self.assertTrue(config["eval_even_build_configs"])

    def test_a_does_not_ask_for_an_even_spread(self):
        """It evaluates one scene, so divisibility would be a constraint with
        nothing to satisfy."""
        self.assertFalse(self._config("a")["eval_even_build_configs"])

    def test_both_attach_the_task_schedule(self):
        for name in ("a", "b"):
            with self.subTest(experiment=name):
                self.assertEqual(self._config(name)["progress_mode"],
                                 "task_schedule")

    def test_pick_profiles_disable_all_object_pairs_by_default(self):
        for name in ("a", "b", "c"):
            with self.subTest(experiment=name):
                self.assertTrue(self._config(name)["graph"]["disable_object_object_relations"])

    def test_c_is_b_evaluation_with_approved_lighting(self):
        config = self._config("c")
        self.assertEqual(config["defaults"][0], "mshab_pick_b")
        self.assertEqual(config["train_build_config_ids"], [SCENE])
        base = yaml.safe_load(Path("configs/env/mshab.yaml").read_text())
        self.assertEqual(base["eval_lighting"]["envs_per_condition"], 10)
        self.assertEqual(base["eval_lighting"]["conditions"],
                         {"dim": 0.4, "nominal": 1.0, "bright": 2.0})
        self.assertTrue(self._config("b")["eval_lighting"]["enabled"])

    def test_the_base_config_changes_nothing_by_default(self):
        """An ordinary MS-HAB run must behave as it did."""
        with open("configs/env/mshab.yaml") as handle:
            base = yaml.safe_load(handle)
        self.assertEqual(base["mshab_objects"], [])
        self.assertEqual(base["train_build_config_ids"], [])
        self.assertFalse(base["eval_even_build_configs"])
        self.assertFalse(base["graph"]["disable_object_object_relations"])


class LauncherTest(unittest.TestCase):
    """Launchers use approved settings and still validate supplied assets."""

    def _script(self, name):
        return Path(f"runs/mshab/experiment_{name}.sh").read_text(
            encoding="utf-8")

    def test_they_do_not_reuse_the_old_all_object_launcher(self):
        for name in ("a", "b"):
            with self.subTest(experiment=name):
                self.assertIn(f"env=mshab_pick_{name}", self._script(name))

    def test_they_refuse_an_asset_that_does_not_validate(self):
        """Including a key-migrated one: the launcher used to grep for that
        single field, and a run has more ways to be unrunnable than one. The
        validator checks every gate the graph builder applies at
        construction, migration among them."""
        for name in ("a", "b"):
            with self.subTest(experiment=name):
                self.assertIn("validate_task_assets", self._script(name))

    def test_the_validator_checks_migration_and_the_required_bins(self):
        source = Path("tests/probes/validate_task_assets.py").read_text(
            encoding="utf-8")
        self.assertIn("migrated_pre_anchor", source)
        self.assertIn("required_bin_keys", source)

    def test_approved_capacity_and_asset_sized_vocabulary(self):
        for name in ("a", "b"):
            with self.subTest(experiment=name):
                script = self._script(name)
                self.assertIn('"${ENTITY_VOCAB:-}"', script)
                self.assertIn("model.graph.n_max=8", script)
                self.assertIn("model.graph.e_max=168", script)
                self.assertIn('"$@"', script)
                self.assertIn('logdir="$REPO_ROOT/logdir/', script)
        probe = Path("runs/mshab/validate.sh").read_text(encoding="utf-8")
        self.assertIn("--n-max 8 --e-max 168", probe)

    def test_the_approved_checkpoint_metric_is_success_once(self):
        for name in ("a", "b"):
            with self.subTest(experiment=name):
                script = self._script(name)
                self.assertIn("checkpoint.metric=eval/success_once", script)
                self.assertIn("checkpoint.tiebreak=''", script)
                self.assertNotIn("CKPT_METRIC", script)
                self.assertNotIn("CKPT_TIEBREAK", script)

    def test_neither_launcher_pins_the_old_entity_vocabulary(self):
        for name in ("a", "b"):
            script = self._script(name)
            with self.subTest(experiment=name):
                self.assertNotIn("entity_vocab=14", script)


class CheckpointConfigBlockTest(unittest.TestCase):

    def _config(self):
        with open("configs/configs.yaml") as handle:
            return yaml.safe_load(handle)["checkpoint"]

    def test_it_is_off_by_default(self):
        self.assertFalse(self._config()["enabled"])

    def test_the_metric_is_unset(self):
        self.assertEqual(self._config()["metric"], "")

    def test_the_start_step_is_the_agreed_one(self):
        self.assertEqual(float(self._config()["start_step"]), 8e6)

    def test_one_path_and_no_milestone_settings(self):
        config = self._config()
        self.assertEqual(config["path"], "checkpoint_best.pt")
        for absent in ("save_latest", "save_final", "save_periodically",
                       "save_on_interrupt"):
            with self.subTest(key=absent):
                self.assertNotIn(absent, config)


if __name__ == "__main__":
    unittest.main()
