"""The ordinary-ManiSkill construction path, and that MS-HAB is unchanged.

`ManiSkillVecEnv.__init__` cannot run without a simulator, so these read the
source of the branch rather than executing it: which kwargs each arm passes to
`gym.make`, which wrappers each applies, and which config keys each requires.
The parts that are pure functions -- the MS-HAB predicate, the graph config
flattening, the schedule source -- are exercised directly.
"""

import ast
import re
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

SOURCE = Path("envs/maniskill.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _node(name):
    for item in TREE.body:
        if getattr(item, "name", None) == name:
            return item
        if isinstance(item, ast.ClassDef):
            for sub in item.body:
                if getattr(sub, "name", None) == name:
                    return sub
    raise AssertionError(f"{name} is not defined in envs/maniskill.py")


def _source(name) -> str:
    return ast.get_source_segment(SOURCE, _node(name))


def _ctor_source() -> str:
    return _source("__init__")


def _load(*names):
    """Exec selected top-level definitions without importing the package.

    ``envs/__init__`` pulls torch, and the whole point of these tests is that
    the ordinary-ManiSkill branch does not depend on the heavy stack.
    """
    wanted, out = set(names), {"Path": Path, "__file__": "envs/maniskill.py"}
    body = [
        item for item in TREE.body
        if (getattr(item, "name", None) in wanted
            or (isinstance(item, ast.Assign)
                and any(getattr(t, "id", "") in wanted for t in item.targets)))
    ]
    exec(compile(ast.Module(body=body, type_ignores=[]), "<sel>", "exec"), out)
    return SimpleNamespace(**out)


mod = _load("is_mshab_task", "mshab_subtask", "task_schedule_source",
            "graph_observation_config",
            "_repo_path", "_GRAPH_CONFIG_KEYS", "_GRAPH_CONFIG_CASTS",
            "rendered_cameras", "_camera_keys", "camera_obs_key")


class TaskIdentityTest(unittest.TestCase):
    def test_mshab_subtask_envs_are_recognised(self):
        self.assertTrue(mod.is_mshab_task("PickSubtaskTrain-v0"))
        self.assertTrue(mod.is_mshab_task("PlaceSubtaskTrain-v0"))

    def test_ordinary_tasks_are_not(self):
        for task in ("PickCube-v1", "StackCube-v1", "PegInsertionSide-v1",
                     "PlugCharger-v1", "PlaceSphere-v1", "PullCubeTool-v1"):
            with self.subTest(task=task):
                self.assertFalse(mod.is_mshab_task(task))

    def test_the_gym_id_drops_the_config_prefix(self):
        """Assets and schedules are stored under the gym id; the config can
        only spell it with a maniskill_ prefix."""
        source = _ctor_source()
        self.assertIn('task = str(config.task).split("_", 1)[1]', source)
        self.assertEqual("maniskill_PickCube-v1".split("_", 1)[1], "PickCube-v1")


class BranchTest(unittest.TestCase):
    """Everything plan-shaped stays behind the MS-HAB guard."""

    MSHAB_ONLY = (
        "task_plans", "scene_builder_cls", "spawn_data_fp",
        "plan_data_from_file", "FetchActionWrapper", "mshab_obj",
        "require_build_configs_repeated_equally_across_envs",
        "balance_objects", "build_panel", "case_intensities", "eval_panel",
    )

    def setUp(self):
        self.source = _ctor_source()

    def _guarded_regions(self):
        """File line spans of every ``if self._is_mshab:`` body in the ctor.

        Taken from the module tree rather than a re-parsed snippet, so the line
        numbers are the file's and no dedenting is involved.

        One span per body, first statement to last, rather than one per
        statement. A body is contiguous, so nothing between its ends is
        outside the branch -- while per-statement spans leave the gaps between
        them uncovered, and a comment sitting in one reads as unguarded.
        """
        spans = []
        for node in ast.walk(_node("__init__")):
            if isinstance(node, ast.If) and "_is_mshab" in ast.dump(node.test):
                if not node.body:
                    continue
                spans.append((
                    node.body[0].lineno,
                    max(getattr(s, "end_lineno", s.lineno) for s in node.body)))
        return spans

    def test_every_mshab_only_symbol_sits_behind_the_guard(self):
        ctor = _node("__init__")
        lines = SOURCE.splitlines()
        spans = self._guarded_regions()
        for symbol in self.MSHAB_ONLY:
            hits = [
                i for i in range(ctor.lineno, ctor.end_lineno + 1)
                if symbol in lines[i - 1]
            ]
            self.assertTrue(hits, f"{symbol} vanished from the constructor")
            for line_no in hits:
                self.assertTrue(
                    any(lo <= line_no <= hi for lo, hi in spans),
                    f"{symbol} on line {line_no} is not behind if self._is_mshab",
                )

    def test_the_shared_make_kwargs_carry_no_plan(self):
        """An ordinary task gets id/obs/render/sensors/backend/reward/steps and
        nothing that presumes a ReplicaCAD scene."""
        head = self.source.split("if self._is_mshab:")[0]
        self.assertIn("id=task", head)
        self.assertIn("reward_mode=", head)
        self.assertIn('make_kwargs["max_episode_steps"] = horizon', head)
        for symbol in self.MSHAB_ONLY:
            self.assertNotIn(symbol, head)

    def test_mshab_imports_are_not_required_by_the_ordinary_path(self):
        """An ordinary run should not need the MS-HAB package installed."""
        head = self.source.split("if self._is_mshab:")[0]
        self.assertNotIn("import mshab", head)
        self.assertNotIn("ASSET_DIR", head)


class InstructionTest(unittest.TestCase):
    def test_the_reader_is_optional(self):
        source = _ctor_source()
        self.assertIn("if instruction_path", source)
        self.assertIn("else None", source)

    def test_every_use_is_guarded(self):
        """A None reader must not reach the observation space or a transition:
        the key is absent for ordinary ManiSkill, not present and constant."""
        for method in ("_build_observation_space", "_transition", "step"):
            body = _source(method)
            for line_no, line in enumerate(body.splitlines(), 1):
                if "self._instruction" not in line or "is not None" in line:
                    continue
                if "self._instruction_obs" in line and "None" in line:
                    continue
                context = "\n".join(body.splitlines()[max(0, line_no - 4):line_no])
                self.assertIn(
                    "is not None", context,
                    f"{method} line {line_no} reads the instruction unguarded:"
                    f"\n{line}")


class ConfigTest(unittest.TestCase):
    def _env(self, name):
        with open(f"configs/env/{name}.yaml") as handle:
            return yaml.safe_load(handle)

    def test_ordinary_maniskill_declares_no_instruction_table(self):
        self.assertNotIn("instruction_table", self._env("maniskill"))

    def test_mshab_still_declares_one(self):
        self.assertIn("instruction_table", self._env("mshab"))

    def _base(self):
        with open("configs/model/_base_.yaml") as handle:
            return yaml.safe_load(handle)

    def test_ordinary_maniskill_runs_the_task_schedule(self):
        self.assertEqual(self._env("maniskill")["progress_mode"], "task_schedule")

    def test_mshab_keeps_the_end_effector_target(self):
        self.assertEqual(self._env("mshab")["progress_mode"], "ee_target")

    def test_the_model_reads_the_mode_from_the_env(self):
        """The value has to reach `config.model`, which is all Dreamer sees.

        This is the wiring that was wrong: an env-level `progress:` block reads
        as configuration and is in fact inert, so a targetless task trained its
        progress head against the end-effector ladder without a word.
        """
        self.assertEqual(self._base()["progress"]["mode"],
                         "${oc.select:env.progress_mode,ee_target}")

    def test_no_env_declares_a_progress_block(self):
        """Nothing reads `env.progress`. Declaring one is silently inert."""
        for path in sorted(Path("configs/env").glob("*.yaml")):
            with self.subTest(env=path.name), open(path) as handle:
                self.assertNotIn("progress", yaml.safe_load(handle) or {})

    def test_no_model_preset_pins_the_mode(self):
        """A literal `mode:` in a preset shadows the interpolation, and picks
        the shaping target by model size."""
        for path in sorted(Path("configs/model").glob("*.yaml")):
            if path.name == "_base_.yaml":
                continue
            with self.subTest(model=path.name), open(path) as handle:
                progress = (yaml.safe_load(handle) or {}).get("progress", {})
                self.assertNotIn("mode", progress)

    def test_the_schedule_directory_is_declared(self):
        with open("configs/model/_base_.yaml") as handle:
            base = yaml.safe_load(handle)["progress"]
        self.assertTrue(base["schedule_dir"])

    def test_ordinary_maniskill_keeps_object_object_spatial(self):
        """The schedules' carry phases read object-to-object distance; MS-HAB
        deliberately does not emit it."""
        self.assertTrue(self._env("maniskill")["graph"]["object_object_spatial"])
        self.assertFalse(self._env("mshab")["graph"]["object_object_spatial"])


class TaskDefaultsTest(unittest.TestCase):
    """Ordinary ManiSkill takes the task's own horizon and robot.

    Both are registration facts, and overriding either quietly makes a
    different task than the one a published number was measured on. MS-HAB is
    the exception: its 100/200 and its Fetch come from mshab's configs, not
    from the registration, so it still states them.
    """

    def _env(self, name):
        with open(f"configs/env/{name}.yaml") as handle:
            return yaml.safe_load(handle)

    def test_ordinary_maniskill_defers_on_both_horizons(self):
        env = self._env("maniskill")
        self.assertEqual(int(env["time_limit"]), 0)
        self.assertEqual(int(env["eval_time_limit"]), 0)

    def test_mshab_still_states_its_own(self):
        env = self._env("mshab")
        self.assertEqual(int(env["time_limit"]), 100)
        self.assertEqual(int(env["eval_time_limit"]), 200)

    def test_ordinary_maniskill_defers_on_the_robot(self):
        self.assertEqual(str(self._env("maniskill")["robot_uids"] or ""), "")

    def test_ordinary_maniskill_names_no_cameras(self):
        """Which cameras exist follows from the task and its robot -- a bare
        panda renders one, panda_wristcam two -- so the list is discovered."""
        self.assertEqual(list(self._env("maniskill")["cameras"] or []), [])

    def test_its_cnn_keys_match_whatever_turns_up(self):
        """A pattern pinned to one camera would drop the second on a task that
        renders two."""
        env = self._env("maniskill")
        for side in ("encoder", "decoder"):
            with self.subTest(side=side):
                pattern = re.compile(env[side]["cnn_keys"])
                self.assertTrue(pattern.search("image_base"))
                self.assertTrue(pattern.search("image_hand"))
                self.assertFalse(pattern.search("state"))

    def test_mshab_still_names_its_two(self):
        """A ReplicaCAD scene carries sensors the graph has no use for."""
        self.assertEqual(list(self._env("mshab")["cameras"]),
                         ["fetch_head", "fetch_hand"])

    def test_no_env_declares_a_camera_count(self):
        """It is only knowable once the env exists, so a config value could
        only ever be a guess that drifts."""
        for path in sorted(Path("configs/env").glob("*.yaml")):
            with self.subTest(env=path.name), open(path) as handle:
                self.assertNotIn("n_cams", yaml.safe_load(handle) or {})

    def test_zero_omits_the_kwarg_rather_than_passing_zero(self):
        """max_episode_steps=0 would be a zero-length episode, not a default."""
        source = _ctor_source()
        self.assertIn("if horizon > 0:", source)
        self.assertIn('make_kwargs["max_episode_steps"] = horizon', source)

    def test_an_empty_uid_omits_the_kwarg(self):
        source = _ctor_source()
        self.assertIn("if robot_uids:", source)
        self.assertIn('make_kwargs["robot_uids"] = robot_uids', source)


class CameraDiscoveryTest(unittest.TestCase):
    """The camera set comes from the built env, not from config."""

    def _env(self, sensors=None, registry=None):
        base = SimpleNamespace()
        if sensors is not None:
            base._init_raw_obs = {"sensor_data": {n: {} for n in sensors}}
        if registry is not None:
            base._sensors = {n: object() for n in registry}
        return SimpleNamespace(unwrapped=base)

    def test_it_reads_what_the_task_rendered(self):
        self.assertEqual(
            mod.rendered_cameras(self._env(["base_camera"])), ["base_camera"])

    def test_order_is_the_task_order_not_alphabetical(self):
        """Camera order is the bbox row order in the packed graph, so it has to
        be the task's own and stable, not sorted."""
        self.assertEqual(
            mod.rendered_cameras(self._env(["base_camera", "hand_camera"])),
            ["base_camera", "hand_camera"])

    def test_the_sensor_registry_is_the_fallback(self):
        env = self._env(sensors=None, registry=["fetch_head"])
        self.assertEqual(mod.rendered_cameras(env), ["fetch_head"])

    def test_a_task_that_renders_nothing_raises(self):
        """Silently building a graph with no pixels would produce an empty
        scene every frame and never say why."""
        with self.assertRaises(RuntimeError):
            mod.rendered_cameras(self._env([]))

    def test_two_cameras_sharing_a_key_still_raise(self):
        """The two naming conventions can still meet on one key: MS-HAB's
        `fetch_hand` and ManiSkill's `hand_camera` both shorten to image_hand.
        Discovery must not lose that check."""
        self.assertEqual(mod.camera_obs_key("fetch_hand"),
                         mod.camera_obs_key("hand_camera"))
        with self.assertRaises(ValueError):
            mod._camera_keys(["fetch_hand", "hand_camera"])

    def test_the_discovered_names_survive_the_key_mapping(self):
        keys = mod._camera_keys(["base_camera", "hand_camera"])
        self.assertEqual(keys, {"image_base": "base_camera",
                                "image_hand": "hand_camera"})

    def test_discovery_happens_after_the_env_exists(self):
        """It cannot happen before: the cameras are a property of the robot the
        task registers, which gym.make decides."""
        source = _ctor_source()
        self.assertLess(source.index("_make_with_supported_reward("),
                        source.index("rendered_cameras(env)"))

    def test_a_named_list_is_left_alone(self):
        source = _ctor_source()
        self.assertIn("if not self._camera_names:", source)


class ModelCameraCountTest(unittest.TestCase):
    """Dreamer sizes its per-camera layers from the packed graph."""

    SOURCE = Path("dreamer.py").read_text(encoding="utf-8")

    def test_it_reads_the_packed_bbox(self):
        start = self.SOURCE.index("def _sync_camera_count")
        body = self.SOURCE[start:self.SOURCE.index("class Dreamer", start)]
        self.assertIn("graph_node_bbox", body)
        self.assertIn("graph_config.n_cams = observed", body)

    def test_it_runs_before_the_graph_modules_are_built(self):
        """GraphEncoder reads config.n_cams in its constructor, so a later
        override would size the layers off the stale value."""
        self.assertLess(self.SOURCE.index("_sync_camera_count(config.graph"),
                        self.SOURCE.index("GraphEncoder(config.graph)"))

    def test_it_is_skipped_without_a_graph(self):
        guard = "if self.graph_enabled:" + chr(10) + (" " * 12)
        self.assertIn(guard + "_sync_camera_count", self.SOURCE)


class VideoPredTest(unittest.TestCase):
    """Open-loop prediction has to read the cameras the decoder was built for.

    `video_pred` used to hardcode the observation key `image`, which no
    multi-camera env produces -- turning trainer.video_pred_log on raised a
    KeyError at the first metrics log instead of showing model drift.
    """

    SOURCE = Path("dreamer.py").read_text(encoding="utf-8")

    def _body(self):
        start = self.SOURCE.index("def _pixel_keys")
        return self.SOURCE[start:self.SOURCE.index("def _trace_stage_start")]

    def test_the_single_camera_key_is_gone(self):
        body = self._body()
        self.assertNotIn('["image"]', body)
        self.assertNotIn('data["image"]', body)

    def test_the_keys_come_from_the_decoder(self):
        self.assertIn("self.decoder, \"cnn_shapes\"", self._body())

    def test_truth_and_reconstruction_share_one_order(self):
        """Tiled side by side, so a column of the strip is one camera in all
        three rows -- truth, model, error."""
        body = self._body()
        self.assertEqual(body.count("self._tile_cameras"), 3)
        self.assertEqual(body.count("keys = self._pixel_keys()"), 1)

    def test_a_pixelless_run_says_so(self):
        self.assertIn("needs a pixel decoder", self._body())


class BaselineArmTest(unittest.TestCase):
    """A graph-free run must survive env=maniskill's progress_mode.

    The env config names task_schedule because the graph arms want it, but the
    baseline turns the graph off -- there is no whitelist to resolve roles
    against, and nothing consuming a potential either.
    """

    SOURCE = Path("dreamer.py").read_text(encoding="utf-8")

    def test_disabled_progress_skips_the_compile(self):
        self.assertIn(
            'if not self.progress_enabled or self.progress_mode != "task_schedule":',
            self.SOURCE)

    def test_the_graphless_env_still_reports_nothing(self):
        """build_graph_obs returns None when disabled, so the source lookup has
        to cope rather than assume a builder."""
        envs = SimpleNamespace(_task_id="PickCube-v1", _graph=None)
        self.assertIsNone(mod.task_schedule_source(envs))

    def test_each_baseline_matches_its_graph_arm(self):
        """A control that also changed the layer sizes would not be measuring
        the graph."""
        def load(name):
            with open(f"configs/model/{name}.yaml") as handle:
                return yaml.safe_load(handle)
        for size in ("size50M", "size100M"):
            arm, base = load(f"{size}_graph_simple"), load(size)
            for key in ("deter", "hidden", "units", "depth", "act", "norm",
                        "discrete"):
                with self.subTest(size=size, key=key):
                    self.assertEqual(arm[key], base[key])

    def test_no_baseline_carries_a_graph_or_progress_block(self):
        for size in ("size50M", "size100M"):
            with self.subTest(size=size), \
                    open(f"configs/model/{size}.yaml") as handle:
                preset = yaml.safe_load(handle)
                self.assertNotIn("progress", preset)
                self.assertNotIn("graph", preset)


class RepLossDefaultTest(unittest.TestCase):
    """A plain preset is a pure DreamerV3 run, with nothing to switch off."""

    def _base(self):
        with open("configs/model/_base_.yaml") as handle:
            return yaml.safe_load(handle)

    def test_the_default_is_stock_dreamer(self):
        self.assertEqual(self._base()["rep_loss"], "dreamer")

    def test_the_plain_preset_overrides_nothing(self):
        """size50M inherits the whole switch set, so `model=size50M` alone is
        the baseline -- no graph, no progress, stock representation loss."""
        with open("configs/model/size50M.yaml") as handle:
            preset = yaml.safe_load(handle)
        for key in ("rep_loss", "graph", "graph_simple", "graph_only_latent",
                    "progress"):
            with self.subTest(key=key):
                self.assertNotIn(key, preset)

    def test_everything_is_off_in_the_base(self):
        base = self._base()
        self.assertFalse(base["graph"]["enabled"])
        self.assertFalse(base["progress"]["enabled"])

    def test_the_retired_switches_are_gone(self):
        """graph.enabled and progress.enabled are the only two left."""
        base = self._base()
        for key in ("graph_simple", "graph_only_latent"):
            self.assertNotIn(key, base)
        for key in ("state_mode", "slot_dim", "slot_births", "app_dim",
                    "uid_vocab"):
            self.assertNotIn(key, base["graph"])
        self.assertNotIn("source", base["progress"])

    def test_the_graph_presets_still_pin_dreamer(self):
        """Dreamer raises on graph.enabled with any other rep_loss, so the
        presets must not rely on the default drifting back."""
        self.assertEqual(self._base()["rep_loss"], "dreamer")

    def test_only_four_presets_remain(self):
        from pathlib import Path as _P
        names = sorted(p.stem for p in _P("configs/model").glob("*.yaml"))
        self.assertIn("size50M", names)
        self.assertIn("size50M_graph_simple", names)
        self.assertIn("size100M", names)
        self.assertIn("size100M_graph_simple", names)
        for gone in ("size50M_graph", "size50M_graph_slots",
                     "size100M_graph", "size100M_graph_slots"):
            self.assertNotIn(gone, names)


class ArmSummaryTest(unittest.TestCase):
    """Every run states which arm it is, once, at construction."""

    SOURCE = Path("dreamer.py").read_text(encoding="utf-8")

    def _body(self):
        start = self.SOURCE.index("def arm_summary")
        return self.SOURCE[start:self.SOURCE.index("    @staticmethod", start)]

    def test_it_is_printed(self):
        self.assertIn('print(f"[arm] {self.arm_summary()}", flush=True)',
                      self.SOURCE)

    def test_it_names_both_graph_states(self):
        body = self._body()
        for name in ("off", "on"):
            with self.subTest(shape=name):
                self.assertIn(f'"{name}"', body)

    def test_it_reports_progress_and_beta(self):
        body = self._body()
        self.assertIn("self.progress_mode", body)
        self.assertIn("self.progress_beta", body)


class SweepScriptTest(unittest.TestCase):
    """Every command in every arm carries the same knobs."""

    PATHS = sorted(Path("runs/maniskill").glob("slurm_*.sh"))

    @staticmethod
    def _commands(path):
        """One dict of overrides per `python train.py` block."""
        out = []
        for block in path.read_text(encoding="utf-8").split("\n\n"):
            if not block.lstrip().startswith("python train.py"):
                continue
            pairs = {}
            for token in re.findall(r"([\w.]+)=(\S+)", block):
                pairs[token[0]] = token[1].rstrip(" '" + chr(92))
            out.append(pairs)
        return out

    def test_no_arm_is_empty(self):
        """How many arms and tasks are live is an experiment decision and
        changes between sweeps. Emptiness is not: an arm whose commands were
        all commented out would make every per-command check below pass by
        having nothing to check."""
        self.assertGreaterEqual(len(self.PATHS), 3)
        for path in self.PATHS:
            with self.subTest(script=path.name):
                self.assertTrue(self._commands(path))

    def test_no_command_restates_a_shared_default(self):
        """Anything the same in all runs belongs in config."""
        for path in self.PATHS:
            text = path.read_text(encoding="utf-8")
            for flag in ("trainer.video_pred_log", "env.env_num", "batch_size",
                         "batch_length", "seed=", "buffer.max_size",
                         "trainer.steps", "device=", "wandb.project",
                         "model.graph.n_max", "model.graph.e_max"):
                with self.subTest(script=path.name, flag=flag):
                    self.assertNotIn(flag, text)

    def test_the_shared_defaults_are_declared(self):
        import yaml
        with open("configs/configs.yaml") as f:
            root = yaml.safe_load(f)
        with open("configs/env/maniskill.yaml") as f:
            env = yaml.safe_load(f)
        self.assertEqual(root["batch_size"], 32)
        self.assertEqual(root["batch_length"], 64)
        self.assertIs(root["trainer"]["video_pred_log"], True)
        self.assertEqual(env["env_num"], 200)
        self.assertEqual(float(env["steps"]), 4e6)

    def test_sparse_runs_are_strict_about_the_reward(self):
        """A sparse run that silently fell back would not be the sparse arm."""
        for path in self.PATHS:
            for cmd in self._commands(path):
                name = cmd["wandb.name"]
                with self.subTest(run=name):
                    if "-sparse-" in name:
                        self.assertEqual(cmd.get("env.reward_mode"), "sparse")
                        self.assertEqual(cmd.get("env.reward_fallback"), "[]")
                    else:
                        self.assertNotIn("env.reward_mode", cmd)

    def test_beta_is_constant_within_a_reward_mode(self):
        """One arm, one beta per reward mode.

        Which betas an arm runs is an experiment decision and changes between
        sweeps, so nothing here pins the values. Mixing two betas across runs
        that share a reward mode is not a decision -- it makes the arm
        uninterpretable, because a difference cannot be attributed.
        """
        for path in self.PATHS:
            seen = {}
            for cmd in self._commands(path):
                beta = cmd.get("model.progress.beta")
                if beta is None:
                    continue
                kind = "sparse" if "-sparse-" in cmd["wandb.name"] else "native"
                seen.setdefault(kind, set()).add(beta)
            with self.subTest(script=path.name):
                mixed = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
                self.assertFalse(
                    mixed, f"{path.name} mixes betas within a reward mode: {mixed}")

    def test_the_baseline_is_a_plain_preset(self):
        """No graph overrides: the preset carries no graph block, so nothing
        needs switching off. Three flags to read instead of none was the whole
        problem."""
        commands = self._commands(Path("runs/maniskill/slurm_baseline.sh"))
        self.assertTrue(commands)
        for cmd in commands:
            with self.subTest(run=cmd["wandb.name"]):
                self.assertEqual(cmd["model"], "size50M")
                self.assertEqual(cmd["env.obs_mode"], "rgb")
                for absent in ("model.graph.enabled", "model.progress.beta",
                               "model.rep_loss"):
                    self.assertNotIn(absent, cmd)

    def test_every_arm_groups_by_task(self):
        """The baseline runs a different preset now, so ${run_name} would put
        it in a different wandb group than the arms it is the control for."""
        for path in self.PATHS:
            with self.subTest(script=path.name):
                commands = self._commands(path)
                self.assertTrue(commands)
                for cmd in commands:
                    group, task = cmd.get("wandb.group"), cmd.get("env.task")
                    with self.subTest(run=cmd["wandb.name"]):
                        self.assertTrue(str(group).startswith("maniskill_"))
                        # The group is the task, so every run of a task lands
                        # beside the others whatever else the arm varies.
                        # How many runs an arm holds is not pinned: that is an
                        # experiment decision and changes between sweeps.
                        if task is not None:
                            self.assertEqual(group, task)

    def test_each_arm_has_its_own_names(self):
        """Two arms sharing a run name overwrite each other in wandb."""
        names = [c["wandb.name"] for p in self.PATHS for c in self._commands(p)]
        self.assertTrue(names)
        self.assertEqual(len(set(names)), len(names))

    def test_the_graph_arms_keep_segmentation(self):
        """Nodes are seeded from it; rgb alone would build an empty graph. The
        env config supplies it, so the graph arms say nothing and only the
        baseline overrides it away."""
        import yaml
        with open("configs/env/maniskill.yaml") as f:
            self.assertEqual(yaml.safe_load(f)["obs_mode"], "rgb+segmentation")
        for name in ("slurm_beta0.sh", "slurm_beta005.sh"):
            with self.subTest(script=name):
                text = (Path("runs/maniskill") / name).read_text(encoding="utf-8")
                self.assertNotIn("env.obs_mode", text)

    def test_no_command_pins_a_horizon_or_a_robot(self):
        """Both now come from the task's own registration."""
        for path in self.PATHS:
            with self.subTest(script=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("time_limit=", text)
                self.assertNotIn("robot_uids=", text)
                self.assertNotIn("cameras=", text)


class EvalRenderTest(unittest.TestCase):
    """The eval video comes from the render camera, not the encoder's sensors.

    Two separate cameras: `sensor_configs` sizes what the policy reads and must
    stay at the encoder's resolution, `human_render_camera_configs` sizes what
    a person watches and is free to be much larger. Confusing them either
    starves the video or silently retrains the encoder on bigger images.
    """

    def setUp(self):
        self.ctor = _ctor_source()
        self.trainer = Path("trainer.py").read_text(encoding="utf-8")

    def test_both_env_configs_declare_a_render_size(self):
        for name in ("maniskill", "mshab"):
            with self.subTest(env=name), open(f"configs/env/{name}.yaml") as f:
                size = yaml.safe_load(f)["eval_render_size"]
                self.assertEqual(len(size), 2)
                self.assertTrue(all(int(v) > 0 for v in size))

    def test_the_render_camera_is_eval_only(self):
        """A training env paying to render a 512x512 view nothing reads would
        be pure cost."""
        head, _, tail = self.ctor.partition("if self._eval:")
        self.assertNotIn("human_render_camera_configs", head)
        self.assertIn("human_render_camera_configs", tail)

    def test_the_sensors_keep_the_encoder_resolution(self):
        self.assertIn("sensor_configs=dict(width=size[1], height=size[0])",
                      self.ctor)

    def test_the_render_frame_is_preferred_with_a_fallback(self):
        """Suites with no render camera -- dmc, atari -- must still get the
        observation strip."""
        body = self.trainer
        self.assertIn("frame = _render_frame(envs, 0, panel_fn)", body)
        self.assertIn("frame = _observation_frame(trans, panel_fn)", body)
        self.assertLess(body.index("_render_frame(envs, 0, panel_fn)"),
                        body.index("_observation_frame(trans, panel_fn)"))

    def test_the_graph_panel_joins_the_render_frame(self):
        """The whole point: one video, graph beside the high-resolution view."""
        start = self.trainer.index("def _render_frame")
        end = self.trainer.index("def _observation_frame")
        self.assertIn("_with_panel", self.trainer[start:end])


class GraphConfigTest(unittest.TestCase):
    KEYS = mod._GRAPH_CONFIG_KEYS

    def _config(self, **overrides):
        values = {key: overrides.get(key, _default(key)) for key in self.KEYS}
        return SimpleNamespace(**values)

    def test_the_task_group_override_names_the_asset_tree(self):
        out = mod.graph_observation_config(
            self._config(mshab_task="maniskill_PickCube-v1"),
            ["base_camera"], task_group="PickCube-v1")
        self.assertEqual(out["mshab_task"], "PickCube-v1")

    def test_no_override_leaves_the_configured_group(self):
        """MS-HAB names a task group that is not a gym id, so it must pass
        through untouched."""
        out = mod.graph_observation_config(
            self._config(mshab_task="set_table", disable_object_object_relations=True),
            ["fetch_head"])
        self.assertEqual(out["mshab_task"], "set_table")
        self.assertIs(out["disable_object_object_relations"], True)


class ScheduleSourceTest(unittest.TestCase):
    def test_it_reports_the_gym_id_and_resolved_directory(self):
        envs = SimpleNamespace(
            _task_id="PickCube-v1",
            _graph=SimpleNamespace(whitelist_dir="/a/subtask_whitelists/PickCube-v1"))
        source = mod.task_schedule_source(envs)
        self.assertEqual(source.label, "PickCube-v1")
        self.assertEqual(source.whitelist_dir,
                         "/a/subtask_whitelists/PickCube-v1")

    def test_it_unwraps_nested_envs(self):
        inner = SimpleNamespace(
            _task_id="StackCube-v1",
            _graph=SimpleNamespace(whitelist_dir="/a/b/StackCube-v1"))
        self.assertEqual(
            mod.task_schedule_source(SimpleNamespace(env=inner)).label,
            "StackCube-v1")

    def test_an_env_with_no_mshab_marks_takes_the_ordinary_arm(self):
        """The attributes are read with getattr defaults, so an env stack that
        predates them -- or any wrapper -- still resolves by gym id."""
        envs = SimpleNamespace(
            _task_id="PickCube-v1",
            _graph=SimpleNamespace(whitelist_dir="/a/subtask_whitelists/PickCube-v1"))
        self.assertFalse(hasattr(envs, "_is_mshab"))
        self.assertTrue(
            mod.task_schedule_source(envs).union_whitelist_path.endswith(
                "task_all.json"))

    def test_a_graphless_env_reports_nothing(self):
        envs = SimpleNamespace(_task_id="PickCube-v1", _graph=None)
        self.assertIsNone(mod.task_schedule_source(envs))


def _default(key):
    cast = mod._GRAPH_CONFIG_CASTS.get(key, str)
    return {bool: False, int: 0, float: 0.0}.get(cast, "")


class ScheduleSourceLayoutTest(unittest.TestCase):
    """Which asset identity each arm hands the compiler.

    An ordinary task is named once by its gym id. MS-HAB is named by a task
    group and a subtask, and its gym id names no asset at all -- asking for
    ``PickSubtaskTrain-v0`` found no schedule, no affordance file and no
    whitelist, which is why the whole task_schedule path was unreachable for
    it.
    """

    CONFIGS = str(Path("scenegraph") / "configs")
    SCHEDULES = str(Path("scenegraph") / "configs" / "schedules")

    def _maniskill(self, whitelist_dir=None):
        directory = whitelist_dir or str(
            Path(self.CONFIGS) / "subtask_whitelists" / "PickCube-v1")
        return SimpleNamespace(
            _task_id="PickCube-v1", _is_mshab=False,
            _mshab_task_group="", _mshab_subtask="",
            _graph=SimpleNamespace(whitelist_dir=directory))

    def _mshab(self, whitelist_dir=None):
        directory = whitelist_dir or str(
            Path(self.CONFIGS) / "subtask_whitelists" / "set_table")
        return SimpleNamespace(
            _task_id="PickSubtaskTrain-v0", _is_mshab=True,
            _mshab_task_group="set_table", _mshab_subtask="pick",
            _graph=SimpleNamespace(whitelist_dir=directory))

    def test_the_subtask_comes_off_the_gym_id(self):
        self.assertEqual(mod.mshab_subtask("PickSubtaskTrain-v0"), "pick")
        self.assertEqual(mod.mshab_subtask("PlaceSubtaskTrain-v0"), "place")

    def test_an_ordinary_task_has_no_subtask(self):
        for task in ("PickCube-v1", "PegInsertionSide-v1", "PullCubeTool-v1"):
            with self.subTest(task=task):
                self.assertEqual(mod.mshab_subtask(task), "")

    def test_the_ordinary_arm_resolves_by_gym_id(self):
        source = mod.task_schedule_source(self._maniskill(), self.SCHEDULES)
        self.assertEqual(source.label, "PickCube-v1")
        self.assertTrue(source.schedule_path.endswith("PickCube-v1.json"))
        self.assertTrue(source.affordance_path.endswith("PickCube-v1.json"))
        self.assertTrue(source.union_whitelist_path.endswith("task_all.json"))

    def test_the_mshab_arm_resolves_by_group_and_subtask(self):
        source = mod.task_schedule_source(self._mshab(), self.SCHEDULES)
        self.assertEqual(source.label, "set_table/pick")
        self.assertTrue(source.affordance_path.endswith("set_table.json"))
        self.assertTrue(source.union_whitelist_path.endswith("pick_all.json"))
        self.assertTrue(source.schedule_path.endswith(
            str(Path("set_table") / "pick.json")))

    def test_the_mshab_gym_id_reaches_no_path(self):
        source = mod.task_schedule_source(self._mshab(), self.SCHEDULES)
        for path in (source.schedule_path, source.affordance_path,
                     source.union_whitelist_path, source.whitelist_dir):
            self.assertNotIn("SubtaskTrain", path)

    def test_both_arms_carry_the_graph_adapter_directory_verbatim(self):
        """The compiler and the packer have to read one vocabulary."""
        directory = str(Path("custom") / "root" / "grp")
        for build in (self._maniskill, self._mshab):
            with self.subTest(arm=build.__name__):
                source = mod.task_schedule_source(
                    build(directory), self.SCHEDULES)
                self.assertEqual(source.whitelist_dir, directory)

    def test_the_configs_root_is_derived_from_that_directory(self):
        """Two levels up from <configs>/subtask_whitelists/<tree>, so the
        affordance file is looked for beside the whitelists the run bound."""
        directory = str(Path("custom") / "subtask_whitelists" / "set_table")
        source = mod.task_schedule_source(self._mshab(directory),
                                          self.SCHEDULES)
        self.assertEqual(source.affordance_path,
                         str(Path("custom") / "affordances" / "set_table.json"))

    def test_the_lookup_still_unwraps_nested_envs(self):
        for attr in ("env", "_env"):
            with self.subTest(attr=attr):
                outer = SimpleNamespace(**{attr: self._mshab()})
                source = mod.task_schedule_source(outer, self.SCHEDULES)
                self.assertEqual(source.label, "set_table/pick")


class _Frames:
    """Tensor stand-in that records the shape reaching the CPU transfer."""

    def __init__(self, shape, log):
        self.shape, self.log, self.dtype = tuple(shape), log, "uint8"

    @property
    def ndim(self):
        return len(self.shape)

    def __getitem__(self, key):
        if key is None:
            return _Frames((1,) + self.shape, self.log)
        return _Frames(self.shape[1:], self.log)

    def detach(self):
        return self

    def cpu(self):
        self.log.append(self.shape)
        return self


class RenderSelectionTest(unittest.TestCase):
    """A 93-env eval batch of 512x512 frames must not cross the bus per step."""

    def _namespace(self):
        fake_torch = SimpleNamespace(
            is_tensor=lambda value: isinstance(value, _Frames),
            as_tensor=lambda value: value,
            uint8="uint8")
        namespace = {"torch": fake_torch,
                     "np": SimpleNamespace(asarray=lambda value: value)}
        body = [_node("render"), _node("render_one")]
        exec(compile(ast.Module(body=body, type_ignores=[]), "<render>", "exec"),
             namespace)
        return namespace

    def _actor(self, log, frames=(93, 512, 512, 3)):
        namespace = self._namespace()
        actor = SimpleNamespace(
            _env=SimpleNamespace(render=lambda: _Frames(frames, log)))
        for name in ("render", "render_one"):
            setattr(actor, name, types.MethodType(namespace[name], actor))
        return actor

    def test_render_one_selects_the_row_before_the_cpu_transfer(self):
        log = []
        frame = self._actor(log).render_one(7)
        self.assertEqual(frame.shape, (512, 512, 3))
        self.assertEqual(log, [(512, 512, 3)])

    def test_the_whole_batch_is_still_available_without_an_index(self):
        log = []
        self.assertEqual(self._actor(log).render().shape, (93, 512, 512, 3))
        self.assertEqual(log, [(93, 512, 512, 3)])

    def test_a_single_frame_is_batched_before_the_row_is_taken(self):
        log = []
        self.assertEqual(self._actor(log, (512, 512, 3)).render_one(0).shape,
                         (512, 512, 3))

    def test_a_renderer_that_returns_nothing_stays_none(self):
        namespace, actor = self._namespace(), SimpleNamespace()
        actor._env = SimpleNamespace(render=lambda: None)
        for name in ("render", "render_one"):
            setattr(actor, name, types.MethodType(namespace[name], actor))
        self.assertIsNone(actor.render())
        self.assertIsNone(actor.render_one(3))


if __name__ == "__main__":
    unittest.main()
