"""The ordinary-ManiSkill construction path, and that MS-HAB is unchanged.

`ManiSkillVecEnv.__init__` cannot run without a simulator, so these read the
source of the branch rather than executing it: which kwargs each arm passes to
`gym.make`, which wrappers each applies, and which config keys each requires.
The parts that are pure functions -- the MS-HAB predicate, the graph config
flattening, the schedule source -- are exercised directly.
"""

import ast
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


mod = _load("is_mshab_task", "task_schedule_source", "graph_observation_config",
            "_repo_path", "_GRAPH_CONFIG_KEYS", "_GRAPH_CONFIG_CASTS")


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
    )

    def setUp(self):
        self.source = _ctor_source()

    def _guarded_regions(self):
        """File line spans of every ``if self._is_mshab:`` body in the ctor.

        Taken from the module tree rather than a re-parsed snippet, so the line
        numbers are the file's and no dedenting is involved.
        """
        spans = []
        for node in ast.walk(_node("__init__")):
            if isinstance(node, ast.If) and "_is_mshab" in ast.dump(node.test):
                for stmt in node.body:
                    spans.append((stmt.lineno,
                                  getattr(stmt, "end_lineno", stmt.lineno)))
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
        self.assertIn("max_episode_steps=", head)
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
            self._config(mshab_task="set_table"), ["fetch_head"])
        self.assertEqual(out["mshab_task"], "set_table")


class ScheduleSourceTest(unittest.TestCase):
    def test_it_reports_the_gym_id_and_resolved_directory(self):
        envs = SimpleNamespace(
            _task_id="PickCube-v1",
            _graph=SimpleNamespace(whitelist_dir="/a/subtask_whitelists/PickCube-v1"))
        self.assertEqual(mod.task_schedule_source(envs),
                         ("PickCube-v1", "/a/subtask_whitelists/PickCube-v1"))

    def test_it_unwraps_nested_envs(self):
        inner = SimpleNamespace(
            _task_id="StackCube-v1",
            _graph=SimpleNamespace(whitelist_dir="/a/b/StackCube-v1"))
        self.assertEqual(mod.task_schedule_source(SimpleNamespace(env=inner))[0],
                         "StackCube-v1")

    def test_a_graphless_env_reports_nothing(self):
        envs = SimpleNamespace(_task_id="PickCube-v1", _graph=None)
        self.assertIsNone(mod.task_schedule_source(envs))


def _default(key):
    cast = mod._GRAPH_CONFIG_CASTS.get(key, str)
    return {bool: False, int: 0, float: 0.0}.get(cast, "")


if __name__ == "__main__":
    unittest.main()
