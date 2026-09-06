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
import re
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

    def test_b_evaluates_one_environment_per_selected_scene(self):
        """One environment per scene: reconfiguration_freq is 0, so a
        sub-scene keeps its configuration and the distinct-scene count is
        bounded by the environment count rather than the episode count.

        The two counts have to match, or the scene panel cannot allocate one
        episode per scene and refuses to build."""
        config = self._config("b")
        self.assertEqual(config["eval_num_build_configs"], 20)
        self.assertEqual(config["eval_episode_num"], 20)
        self.assertEqual(config["eval_episode_num"],
                         config["eval_num_build_configs"])
        self.assertTrue(config["eval_even_build_configs"])

    def test_b_evaluates_beyond_the_scene_it_trains_in(self):
        """A held-out split is the whole point; one scene would measure fit."""
        config = self._config("b")
        self.assertEqual(len(config["train_build_config_ids"]), 1)
        self.assertGreater(config["eval_num_build_configs"],
                           len(config["train_build_config_ids"]))

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

    EXPERIMENTS = ("a", "b")
    ARMS = ("beta005", "baseline")

    def _script(self, experiment, arm):
        return Path(f"runs/mshab/slurm_{experiment}_{arm}.sh").read_text(
            encoding="utf-8")

    def _scripts(self, arm=None):
        arms = self.ARMS if arm is None else (arm,)
        return [(experiment, one, self._script(experiment, one))
                for experiment in self.EXPERIMENTS for one in arms]

    def test_every_arm_names_its_own_experiment_profile(self):
        for experiment, arm, script in self._scripts():
            with self.subTest(experiment=experiment, arm=arm):
                self.assertIn(f"env=mshab_pick_{experiment}", script)

    def test_the_graph_arm_refuses_an_asset_that_does_not_validate(self):
        """Including a key-migrated one: the launcher used to grep for that
        single field, and a run has more ways to be unrunnable than one. The
        validator checks every gate the graph builder applies at
        construction, migration among them."""
        for experiment, arm, script in self._scripts("beta005"):
            with self.subTest(experiment=experiment, arm=arm):
                self.assertIn("validate_task_assets", script)

    def test_the_validator_checks_migration_and_the_required_bins(self):
        source = Path("tests/probes/validate_task_assets.py").read_text(
            encoding="utf-8")
        self.assertIn("migrated_pre_anchor", source)
        self.assertIn("required_bin_keys", source)

    def test_approved_capacity_and_asset_sized_vocabulary(self):
        for experiment, arm, script in self._scripts("beta005"):
            with self.subTest(experiment=experiment, arm=arm):
                self.assertIn("model.graph.entity_vocab=19", script)
                self.assertIn("model.graph.n_max=8", script)
                self.assertIn("model.graph.e_max=168", script)
                self.assertIn("model.progress.beta=0.05", script)
        probe = Path("runs/mshab/validate.sh").read_text(encoding="utf-8")
        self.assertIn("--n-max 8 --e-max 168", probe)

    def test_the_baseline_arm_carries_no_graph_or_progress_override(self):
        """size50M inherits both switches off; overriding them would say the
        control was configured rather than structurally matched."""
        for experiment, arm, script in self._scripts("baseline"):
            with self.subTest(experiment=experiment, arm=arm):
                self.assertIn("model=size50M \\", script)
                self.assertIn("env.obs_mode=rgb", script)
                for absent in ("model.graph.", "model.progress.",
                               "env.graph.whitelist_dir"):
                    self.assertNotIn(absent, script)

    @staticmethod
    def _active(script):
        """The one uncommented `python train.py` block, continuations included.

        Read the active command, never the whole file: every launcher also
        carries a commented variant, and asserting against both at once is how
        a disabled command gets mistaken for the one that runs.
        """
        block, started = [], False
        for line in script.splitlines():
            started = started or line.startswith("python train.py")
            if not started:
                continue
            block.append(line)
            if not line.rstrip().endswith("\\"):
                break
        return "\n".join(block)

    def test_each_launcher_has_exactly_one_active_training_command(self):
        for experiment, arm, script in self._scripts():
            with self.subTest(experiment=experiment, arm=arm):
                active = [line for line in script.splitlines()
                          if line.startswith("python train.py")]
                self.assertEqual(len(active), 1)

    def test_every_arm_writes_its_own_named_run(self):
        seen = set()
        for experiment, arm, script in self._scripts():
            with self.subTest(experiment=experiment, arm=arm):
                active = self._active(script)
                self.assertIn("logdir=$HOME/logdir/r2dreamer-graph/$TIMESTAMP/",
                              active)
                self.assertIn(f"wandb.group=mshab_tidy_house_pick_"
                              f"{experiment.upper()}", active)
                name = re.search(r"wandb\.name=(\S+)", active).group(1)
                self.assertIn(arm, name)
                seen.add(name)
        self.assertEqual(len(seen), 4)

    def test_the_approved_checkpoint_metric_is_success_once(self):
        for experiment, arm, script in self._scripts():
            with self.subTest(experiment=experiment, arm=arm):
                self.assertIn("checkpoint.metric=eval/success_once", script)
                self.assertIn("checkpoint.tiebreak=''", script)
                self.assertNotIn("CKPT_METRIC", script)
                self.assertNotIn("CKPT_TIEBREAK", script)

    def test_the_selected_model_is_saved_outside_the_log_tree(self):
        """Clearing a logdir must not take the checkpoint every later number
        is read from, and four arms must not overwrite each other's."""
        destinations = set()
        for experiment, arm, script in self._scripts():
            with self.subTest(experiment=experiment, arm=arm):
                self.assertIn("CKPT_DIR=$MS_ASSET_DIR/mshab_transfer_checkpoint",
                              script)
                # Fails before the budget when the volume is not mounted.
                self.assertIn('mkdir -p $HOME/output "$CKPT_DIR"', script)
                path = re.search(r"checkpoint\.path=(\S+)",
                                 self._active(script)).group(1)
                self.assertTrue(path.startswith("$CKPT_DIR/"), path)
                # Timestamped, so a rerun cannot silently replace the best a
                # previous run of the same arm earned.
                self.assertIn("${TIMESTAMP}", path)
                destinations.add(path)
        self.assertEqual(len(destinations), 4)

    def test_no_launcher_pins_the_old_entity_vocabulary(self):
        for experiment, arm, script in self._scripts():
            with self.subTest(experiment=experiment, arm=arm):
                self.assertNotIn("entity_vocab=14", script)

    def test_a_transfers_for_five_million_and_b_does_not(self):
        """A is the training-plus-transfer experiment; B is generalization.

        The 5M budget is an explicit override in both A launchers, so the
        shared `finetune.steps` default keeps applying to unrelated runs.
        """
        default = yaml.safe_load(
            Path("configs/configs.yaml").read_text(encoding="utf-8"))["finetune"]
        self.assertEqual(default["steps"], 3_000_000)
        self.assertFalse(default["enabled"])
        for arm in self.ARMS:
            with self.subTest(arm=arm):
                a = self._active(self._script("a", arm))
                self.assertIn("finetune.enabled=true", a)
                self.assertIn("finetune.steps=5000000", a)
                b = self._active(self._script("b", arm))
                self.assertIn("finetune.enabled=false", b)
                self.assertNotIn("finetune.steps", b)

    def test_both_a_arms_transfer_on_the_same_budget(self):
        """A matched comparison needs the two arms to spend the same steps."""
        budgets = {re.findall(r"finetune\.steps=(\d+)",
                              self._active(self._script("a", arm)))[0]
                   for arm in self.ARMS}
        self.assertEqual(budgets, {"5000000"})

    def test_validation_checks_the_launchers_without_running_them(self):
        probe = Path("runs/mshab/validate.sh").read_text(encoding="utf-8")
        self.assertIn("check_launchers", probe)
        self.assertIn("bash -n", probe)
        self.assertIn("^python train\\.py", probe)
        for experiment in self.EXPERIMENTS:
            for arm in self.ARMS:
                self.assertIn(f"runs/mshab/slurm_{experiment}_{arm}.sh", probe)
        # Reading only: validation never executes a launcher.
        for line in probe.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or "slurm_" not in stripped:
                continue
            self.assertNotRegex(stripped, r"^(bash|sh|sbatch|source|\.)\s+\S*slurm_")


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
