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
way ``test_maniskill_env_branch`` does. ``envs/scene_manifest`` does not, so
it is imported by path.
"""

import ast
import importlib.util
import json
import re
import sys
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
MANIFEST = "configs/scenes/mshab_pick_b.json"


def _scene_manifest():
    spec = importlib.util.spec_from_file_location(
        "_scene_manifest", "envs/scene_manifest.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


manifest = _scene_manifest()


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

    def test_b_trains_one_object_from_a_frozen_scene_manifest(self):
        """The scenes are named in one file, not restated in the config.

        Both lists stay empty here on purpose: the manifest fills them in
        before anything is built, and two sources for one decision is how the
        two drift apart."""
        config = self._config("b")
        self.assertEqual(config["mshab_obj"], "004_sugar_box")
        self.assertEqual(config["mshab_objects"], [])
        self.assertEqual(config["scene_manifest"], MANIFEST)
        self.assertEqual(config["train_build_config_ids"], [])
        self.assertEqual(config["eval_build_config_ids"], [])

    def test_b_allocates_twenty_five_training_environments_per_scene(self):
        """125 over 5, and the even-spread flag is what makes it exact.

        Without it MS-HAB assigns whatever it assigns; with it the run refuses
        to start unless the count divides."""
        config = self._config("b")
        split = manifest.load_manifest(MANIFEST)
        self.assertEqual(config["env_num"], 125)
        self.assertEqual(config["env_num"] % len(split.train), 0)
        self.assertEqual(config["env_num"] // len(split.train), 25)
        self.assertTrue(config["train_even_build_configs"])

    def test_b_composes_its_eighty_two_case_panel(self):
        """42 unseen + 2 x 5 training + 30 lighting, and the primary count is
        the first two."""
        config = self._config("b")
        split = manifest.load_manifest(MANIFEST)
        repeats = config["eval_scene_episodes"]
        primary = (repeats["held_out"] * len(split.held_out)
                   + repeats["training"] * len(split.train))
        self.assertEqual(repeats, {"training": 2, "held_out": 1})
        self.assertEqual(primary, 52)
        self.assertEqual(config["eval_episode_num"], primary)
        base = yaml.safe_load(Path("configs/env/mshab.yaml").read_text())
        lighting = len(base["eval_lighting"]["conditions"]) * \
            base["eval_lighting"]["envs_per_condition"]
        self.assertEqual(lighting, 30)
        self.assertEqual(primary + lighting, 82)

    def test_b_does_not_ask_for_an_even_evaluation_spread(self):
        """The panel pins every scene itself, and the two halves are
        deliberately not weighted equally."""
        self.assertFalse(self._config("b")["eval_even_build_configs"])

    def test_b_pins_the_lighting_comparison_to_one_training_scene(self):
        """Training in five scenes must not multiply C by five."""
        config = self._config("b")
        self.assertEqual(config["eval_lighting"]["scene"], SCENE)
        self.assertIn(SCENE, manifest.load_manifest(MANIFEST).train)

    def test_b_evaluates_beyond_the_scenes_it_trains_in(self):
        """A held-out split is the whole point; training scenes alone would
        measure fit."""
        split = manifest.load_manifest(MANIFEST)
        self.assertGreater(len(split.held_out), len(split.train))
        self.assertEqual(split.evaluation, split.train + split.held_out)

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
        # It restates nothing about the scenes: a second copy of the split is
        # a second thing to keep in step with the first.
        for absent in ("train_build_config_ids", "eval_build_config_ids",
                       "scene_manifest", "eval_lighting"):
            self.assertNotIn(absent, config)
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
        # The keys B needs exist here as no-ops, so an ordinary MS-HAB run
        # picks up none of B's behaviour by inheriting them.
        self.assertEqual(base["scene_manifest"], "")
        self.assertFalse(base["train_even_build_configs"])
        self.assertEqual(base["eval_scene_episodes"],
                         {"training": 0, "held_out": 0})
        self.assertEqual(base["eval_lighting"]["scene"], "")


class SceneManifestTest(unittest.TestCase):
    """The frozen five/42 split, and that nothing can silently overlap."""

    def setUp(self):
        self.raw = json.loads(Path(MANIFEST).read_text(encoding="utf-8"))
        self.split = manifest.load_manifest(MANIFEST)

    def test_it_freezes_five_training_and_forty_two_unseen_scenes(self):
        self.assertEqual(manifest.counts(self.split),
                         {"train": 5, "held_out": 42, "evaluation": 47})

    def test_the_two_halves_cannot_overlap(self):
        """An unseen-scene score measured on a trained scene is not a
        generalisation number, so the loader refuses rather than warns."""
        self.assertEqual(set(self.split.train) & set(self.split.held_out), set())
        with self.assertRaises(ValueError) as caught:
            manifest.SceneSplit(train=[SCENE], held_out=[SCENE],
                                lighting_scene=SCENE).validate()
        self.assertIn("trains and holds out", str(caught.exception))

    def test_the_lighting_scene_is_the_original_training_scene(self):
        self.assertEqual(self.split.lighting_scene, SCENE)
        self.assertIn(SCENE, self.split.train)
        with self.assertRaises(ValueError):
            manifest.SceneSplit(train=[SCENE], held_out=["other.scene_instance.json"],
                                lighting_scene="other.scene_instance.json").validate()

    def test_every_name_is_a_build_configuration(self):
        for name in self.split.evaluation:
            self.assertTrue(name.endswith(".scene_instance.json"), name)

    def test_training_scenes_are_arrangements_of_one_apartment(self):
        """Held-out scenes come from apartments the policy never saw, which
        is what makes the number a scene-generalisation result."""
        groups = {manifest.scene_group(n) for n in self.split.train}
        self.assertEqual(len(groups), 1)
        self.assertFalse(
            groups & {manifest.scene_group(n) for n in self.split.held_out})

    def test_the_split_rule_reproduces_the_shipped_manifest(self):
        """The tool that writes the file and the file agree, so regenerating
        it on the training machine is a no-op unless the dataset moved."""
        available = sorted(set(self.split.evaluation))
        recomputed = manifest.split_scenes(available, SCENE, 5, 42)
        self.assertEqual(recomputed.train, self.split.train)
        self.assertEqual(recomputed.held_out, self.split.held_out)

    def test_the_manifest_records_what_it_was_frozen_from(self):
        for key in ("task", "subtask", "object", "split", "available"):
            self.assertIn(key, self.raw)
        self.assertEqual(self.raw["task"], "tidy_house")
        self.assertEqual(self.raw["object"], "004_sugar_box")

    def test_applying_it_fills_both_lists_and_refuses_a_second_source(self):
        config = SimpleNamespace(
            scene_manifest=MANIFEST, train_build_config_ids=[],
            eval_build_config_ids=[],
            eval_lighting=SimpleNamespace(enabled=True, scene=""))
        manifest.apply_scene_manifest(config)
        self.assertEqual(config.train_build_config_ids, self.split.train)
        self.assertEqual(config.eval_build_config_ids, self.split.evaluation)
        self.assertEqual(config.eval_lighting.scene, SCENE)
        clash = SimpleNamespace(
            scene_manifest=MANIFEST, train_build_config_ids=["elsewhere.json"],
            eval_build_config_ids=[], eval_lighting=None)
        with self.assertRaises(ValueError):
            manifest.apply_scene_manifest(clash)

    def test_no_manifest_changes_nothing(self):
        config = SimpleNamespace(scene_manifest="",
                                 train_build_config_ids=[SCENE],
                                 eval_build_config_ids=[])
        self.assertIsNone(manifest.apply_scene_manifest(config))
        self.assertEqual(config.train_build_config_ids, [SCENE])

    def test_the_launchers_check_it_against_the_installed_plans(self):
        """A manifest that has drifted from the dataset has to stop the run
        before the budget, not at the first evaluation."""
        for name in ("slurm_b_beta005", "slurm_b_baseline",
                     "slurm_beta005", "slurm_baseline"):
            script = Path(f"runs/mshab/{name}.sh").read_text(encoding="utf-8")
            with self.subTest(script=name):
                self.assertIn("freeze_scene_split --check", script)
        probe = Path("runs/mshab/validate.sh").read_text(encoding="utf-8")
        self.assertIn("check_scene_manifest", probe)
        self.assertIn("freeze_scene_split --check", probe)


class LauncherTest(unittest.TestCase):
    """Launchers use approved settings and still validate supplied assets."""

    # The graph launcher keeps its filename and reports a different beta:
    # renaming six files mid-experiment is how a submitted job points at a
    # script that no longer exists.
    EXPERIMENTS = ("a", "b")
    ARMS = ("beta005", "baseline")
    LABELS = {"beta005": "beta01", "baseline": "baseline"}
    METRICS = {"a": "eval/success_once",
               "b": "eval_scene/training/success_once"}

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
                self.assertIn("model.progress.beta=0.1", script)
        probe = Path("runs/mshab/validate.sh").read_text(encoding="utf-8")
        self.assertIn("--n-max 8 --e-max 168", probe)

    def test_the_baseline_arm_carries_no_graph_or_progress_override(self):
        """size100M inherits both switches off; overriding them would say the
        control was configured rather than structurally matched."""
        for experiment, arm, script in self._scripts("baseline"):
            with self.subTest(experiment=experiment, arm=arm):
                self.assertIn("model=size100M \\", script)
                self.assertIn("env.obs_mode=rgb", script)
                for absent in ("model.graph.", "model.progress.",
                               "env.graph.whitelist_dir"):
                    self.assertNotIn(absent, script)

    MERGED = {"beta005": "runs/mshab/slurm_beta005.sh",
              "baseline": "runs/mshab/slurm_baseline.sh"}

    @staticmethod
    def _active_blocks(script):
        """Every uncommented `python train.py` block, continuations included.

        Read the active commands, never the whole file: the single-arm
        launchers also carry a commented variant, and asserting against both
        at once is how a disabled command gets mistaken for the one that runs.
        """
        blocks, current = [], None
        for line in script.splitlines():
            if line.startswith("python train.py"):
                current = [line]
            elif current is not None:
                current.append(line)
            else:
                continue
            if not current[-1].rstrip().endswith("\\"):
                blocks.append("\n".join(current))
                current = None
        return blocks

    def _active(self, script):
        return self._active_blocks(script)[0]

    def _merged(self, arm):
        return Path(self.MERGED[arm]).read_text(encoding="utf-8")

    def test_each_merged_launcher_runs_b_then_a(self):
        """B first: it is the generalization result, it has no transfer stage
        queued behind it, and a node that dies overnight should already have
        spent its hours on the load-bearing run."""
        for arm in self.ARMS:
            with self.subTest(arm=arm):
                merged = self._merged(arm)
                blocks = self._active_blocks(merged)
                self.assertEqual(len(blocks), 2)
                self.assertIn("env=mshab_pick_b", blocks[0])
                self.assertIn("env=mshab_pick_a", blocks[1])
                self.assertIn("CKPT_DIR=$MS_ASSET_DIR/mshab_transfer_checkpoint",
                              merged)
                self.assertIn('mkdir -p $HOME/output "$CKPT_DIR"', merged)

    def test_the_merged_launchers_repeat_the_single_arm_commands_verbatim(self):
        """Six files, one set of commands. A copy that drifts from its source
        is a silently different experiment, which is the whole risk of keeping
        both shapes."""
        for arm in self.ARMS:
            blocks = self._active_blocks(self._merged(arm))
            for experiment, block in zip(("b", "a"), blocks):
                with self.subTest(experiment=experiment, arm=arm):
                    self.assertEqual(
                        block, self._active(self._script(experiment, arm)))

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
                self.assertIn(self.LABELS[arm], name)
                seen.add(name)
        self.assertEqual(len(seen), 4)

    def test_each_experiment_selects_on_its_own_metric(self):
        """B selects on the ten training-scene cases, not on the number it
        reports: eval/success_once pools B's unseen scenes in, and selecting
        on that would pick whichever checkpoint got luckiest on the test set.
        A trains and evaluates in one scene, so it has no such half."""
        for experiment, arm, script in self._scripts():
            with self.subTest(experiment=experiment, arm=arm):
                active = self._active(script)
                self.assertIn(f"checkpoint.metric={self.METRICS[experiment]}",
                              active)
                self.assertIn("checkpoint.tiebreak=''", script)
                self.assertNotIn("CKPT_METRIC", script)
                self.assertNotIn("CKPT_TIEBREAK", script)
        for arm in self.ARMS:
            blocks = self._active_blocks(self._merged(arm))
            for experiment, block in zip(("b", "a"), blocks):
                with self.subTest(experiment=experiment, arm=arm, merged=True):
                    self.assertIn(
                        f"checkpoint.metric={self.METRICS[experiment]}", block)

    def test_every_arm_spends_the_agreed_eight_million_steps(self):
        """Stated in the launcher rather than inherited: the env default is
        10M, and an arm that quietly ran two million steps longer would not
        be comparable to the one beside it."""
        for experiment, arm, script in self._scripts():
            with self.subTest(experiment=experiment, arm=arm):
                self.assertIn("env.steps=8000000", self._active(script))

    def test_checkpoint_eligibility_starts_two_million_steps_early(self):
        """6M against an 8M budget keeps the agreed two-million-step selection
        window; leaving it at 8M would reduce selection to the final
        evaluation."""
        default = yaml.safe_load(
            Path("configs/configs.yaml").read_text(encoding="utf-8"))
        self.assertEqual(float(default["checkpoint"]["start_step"]), 6e6)
        for experiment, arm, script in self._scripts():
            with self.subTest(experiment=experiment, arm=arm):
                active = self._active(script)
                self.assertIn("checkpoint.start_step=6000000", active)
                budget = int(re.search(r"env\.steps=(\d+)", active).group(1))
                self.assertEqual(budget - 6_000_000, 2_000_000)

    def test_every_arm_uses_the_matched_hundred_million_models(self):
        """One capacity across both arms, or the comparison measures size."""
        for experiment, arm, script in self._scripts():
            with self.subTest(experiment=experiment, arm=arm):
                active = self._active(script)
                self.assertIn(
                    "model=size100M_graph_simple" if arm == "beta005"
                    else "model=size100M", active)
                self.assertNotIn("size50M", active)

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
        for arm in self.ARMS:
            self.assertIn(f"runs/mshab/slurm_{arm}.sh:2", probe)
            for experiment in self.EXPERIMENTS:
                self.assertIn(f"runs/mshab/slurm_{experiment}_{arm}.sh:1", probe)
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
        """6M, against the 8M budget both experiments now run: the agreed
        selection window is the last two million steps, and leaving
        eligibility at the budget would reduce selection to the final
        evaluation."""
        self.assertEqual(float(self._config()["start_step"]), 6e6)

    def test_one_path_and_no_milestone_settings(self):
        config = self._config()
        self.assertEqual(config["path"], "checkpoint_best.pt")
        for absent in ("save_latest", "save_final", "save_periodically",
                       "save_on_interrupt"):
            with self.subTest(key=absent):
                self.assertNotIn(absent, config)


if __name__ == "__main__":
    unittest.main()
