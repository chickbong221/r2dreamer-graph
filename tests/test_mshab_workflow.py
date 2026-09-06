"""One contract suite for fixed B+C evaluation and transfer orchestration."""

import ast
from collections import Counter
import contextlib
import importlib.util
from pathlib import Path
import sys
import types
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np
import yaml

spec = importlib.util.spec_from_file_location("_eval_contract", "envs/evaluation.py")
evaluation = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = evaluation
spec.loader.exec_module(evaluation)


def config(mode="scenes", count=63, lighting=True):
    return NS(eval_panel=mode, eval_episode_num=count, train_build_config_ids=["s00"],
              eval_lighting=NS(enabled=lighting, envs_per_condition=10,
                               conditions={"dim": .4, "nominal": 1., "bright": 2.}))


def plans(scenes=63, obj="sugar"):
    return {obj: [NS(build_config_name=f"s{i:02d}", init_config_name=f"init{j:02d}")
                  for i in range(scenes) for j in range(12)]}


class PanelTest(unittest.TestCase):
    def test_one_parallel_panel_has_63_primary_and_30_matched_light_cases(self):
        panel = evaluation.build_panel(plans(), config())
        self.assertEqual(len(panel), 93)
        self.assertEqual(len({c.scene for c in panel if c.group == "scene"}), 63)
        lights = [c for c in panel if c.group == "light"]
        self.assertEqual(Counter(c.condition for c in lights), {"dim": 10, "nominal": 10, "bright": 10})
        self.assertEqual({c.scene for c in lights}, {"s00"})
        for k in range(10):
            self.assertEqual(len({c.plan_index for c in lights if c.repetition == k}), 1)
        self.assertEqual(panel, evaluation.build_panel(plans(), config()))

    def test_the_shipped_b_counts_compose_into_one_fifty_env_panel(self):
        """B's two counts come from mshab_pick_b.yaml; read them, don't restate."""
        shipped = yaml.safe_load(
            Path("configs/env/mshab_pick_b.yaml").read_text(encoding="utf-8"))
        scenes = int(shipped["eval_num_build_configs"])
        panel = evaluation.build_panel(
            plans(scenes), config(count=int(shipped["eval_episode_num"])))
        self.assertEqual(len(panel), scenes + 30)
        self.assertEqual(len({c.scene for c in panel if c.group == "scene"}),
                         scenes)
        self.assertEqual(Counter(c.condition for c in panel if c.group == "light"),
                         {"dim": 10, "nominal": 10, "bright": 10})
        # One episode per scene, and the training scene is among them.
        self.assertEqual(Counter(c.scene for c in panel if c.group == "scene"),
                         {f"s{i:02d}": 1 for i in range(scenes)})
        self.assertIn("s00", {c.scene for c in panel if c.group == "light"})

    def test_a_is_fixed_and_balanced_without_lighting(self):
        source = {}
        for i in range(5):
            source.update(plans(1, f"object{i}"))
        panel = evaluation.build_panel(source, config("objects", 25, False))
        self.assertEqual(Counter(c.object for c in panel), {f"object{i}": 5 for i in range(5)})
        self.assertEqual(len({c.plan_index for c in panel}), 25)

    def test_primary_success_never_includes_lighting_and_heldout_is_separate(self):
        panel = evaluation.build_panel(plans(), config())
        success = np.zeros(93)
        success[0] = 1  # only training scene succeeds
        success[63:] = 1  # C cannot improve B's checkpoint score
        result = evaluation.panel_metrics(panel, {
            "success_once": success, "graph_cache_entries": np.ones(93)}, ["s00"])
        self.assertAlmostEqual(result["eval/success_once"], 1 / 63)
        self.assertEqual(result["eval_scene/training/success_once"], 1)
        self.assertEqual(result["eval_scene/held_out/success_once"], 0)
        self.assertEqual(result["eval_light/dim/episodes"], 10)
        self.assertFalse(any("/per_scene/" in key for key in result))
        self.assertIn("eval/graph_cache_entries", result)
        self.assertFalse(any(key.endswith("graph_cache_entries") and not key.startswith("eval/")
                             for key in result))

    def test_no_scene_is_reported_on_its_own_and_selection_is_unaffected(self):
        panel = evaluation.build_panel(plans(), config())
        success = np.zeros(93)
        success[:32] = 1
        values = {"success_once": success, "success_at_end": success,
                  "fail_once": np.zeros(93), "score": np.arange(93.0),
                  "length": np.ones(93), "graph_cache_entries": np.ones(93)}
        result = evaluation.panel_metrics(panel, values, ["s00"])
        self.assertEqual(
            {key.rsplit("/", 1)[0] for key in result},
            {"eval", "eval_scene/all", "eval_scene/training", "eval_scene/held_out",
             "eval_light/dim", "eval_light/nominal", "eval_light/bright"})
        for scene in {c.scene for c in panel}:
            self.assertFalse([k for k in result if scene.rstrip(".json") in k])
        # Sub-groups carry outcomes only; the full gauge set stays on eval/.
        self.assertFalse([k for k in result
                          if k.endswith("graph_cache_entries") and not k.startswith("eval/")])
        # The checkpoint metric is the 63-case aggregate, and lighting rows
        # cannot move it whatever they score.
        self.assertAlmostEqual(result["eval/success_once"], 32 / 63)
        for filler in (0.0, 1.0):
            shifted = dict(values)
            shifted["success_once"] = np.concatenate([success[:63], np.full(30, filler)])
            self.assertEqual(
                evaluation.panel_metrics(panel, shifted, ["s00"])["eval/success_once"],
                result["eval/success_once"])

    def test_milestones_track_every_reported_aggregate(self):
        tracker = evaluation.SuccessMilestones()
        crossed = tracker.update(
            {"eval/success_once": .9, "eval_scene/held_out/success_once": .6}, 400)
        self.assertEqual(crossed, {
            "eval/steps_to_50": 400, "eval/steps_to_70": 400, "eval/steps_to_80": 400,
            "eval_scene/held_out/steps_to_50": 400})

    def test_invalid_counts_conditions_and_missing_scene_are_refused(self):
        bad = config(count=62)
        with self.assertRaises(ValueError):
            evaluation.build_panel(plans(), bad)
        bad = config()
        bad.eval_lighting.conditions = {"nominal": 1.0}
        with self.assertRaises(ValueError):
            evaluation.build_panel(plans(), bad)
        bad = config()
        bad.train_build_config_ids = ["missing"]
        with self.assertRaises(ValueError):
            evaluation.build_panel(plans(), bad)

    def test_thresholds_keep_first_crossing_and_never_invent_unreached_steps(self):
        tracker = evaluation.SuccessMilestones()
        self.assertEqual(tracker.update({"eval/success_once": .4}, 100), {})
        tracker.update({"eval/success_once": .7}, 200)
        result = tracker.update({"eval/success_once": .5}, 300)
        self.assertEqual(result, {"eval/steps_to_50": 200, "eval/steps_to_70": 200})


AMBIENT = np.array([.3] * 3)
LIGHT = np.array([2., 1.6, 1.])
POSITIONS = ([-1.1, 2.775, 2.3], [2.4, -1.6, 2.3])


def scene_class():
    """Stand-in for the ManiSkill scene class ``construction_lighting`` patches."""

    class ManiSkillScene:
        def __init__(self, count):
            self.sub_scenes = [NS(render_system=NS(ambient_light=None), entities=[])
                               for _ in range(count)]
            self.parallel_in_single_scene = False

        def build(self):
            """ReplicaCADSceneBuilder.build's call shape, nothing more."""
            self.set_ambient_light(list(AMBIENT))
            for position in POSITIONS:
                self.add_point_light(position, color=LIGHT.copy())

        def clear(self):
            for sub in self.sub_scenes:
                sub.entities = []

        def add_point_light(self, position, color, shadow=False, scene_idxs=None):
            chosen = range(len(self.sub_scenes)) if scene_idxs is None else scene_idxs
            for index in chosen:
                light = type("RenderPointLightComponent", (), {})()
                light.color = np.asarray(color, float)
                light.position = tuple(position)
                light.shadow = shadow
                self.sub_scenes[index].entities.append(
                    NS(name="point_light", components=[light]))

        def add_directional_light(self, direction, color, shadow=False, scene_idxs=None):
            self.add_point_light(direction, color=color, shadow=shadow,
                                 scene_idxs=scene_idxs)

        def set_ambient_light(self, color):
            for sub in self.sub_scenes:
                sub.render_system.ambient_light = np.asarray(color, float)

    return ManiSkillScene


@contextlib.contextmanager
def maniskill_scene_module(cls):
    modules = {}
    for name in ("mani_skill", "mani_skill.envs", "mani_skill.envs.scene"):
        module = types.ModuleType(name)
        module.__path__ = []
        modules[name] = module
    modules["mani_skill.envs.scene"].ManiSkillScene = cls
    with patch.dict(sys.modules, modules):
        yield


class LightingTest(unittest.TestCase):
    def setUp(self):
        self.cls = scene_class()
        self.saved = {name: getattr(self.cls, name)
                      for name in ("add_point_light", "add_directional_light",
                                   "set_ambient_light")}

    def assert_restored(self):
        for name, function in self.saved.items():
            self.assertIs(getattr(self.cls, name), function)

    def built(self, panel, intensities=None):
        scene = self.cls(len(panel))
        with maniskill_scene_module(self.cls):
            if intensities is None:
                scene.build()
                return scene, {}
            with evaluation.construction_lighting(intensities) as created:
                scene.build()
        return scene, created

    def assert_scaled(self, scene, panel):
        for sub, case in zip(scene.sub_scenes, panel):
            np.testing.assert_allclose(sub.render_system.ambient_light,
                                       AMBIENT * case.intensity)
            self.assertEqual(len(sub.entities), len(POSITIONS))
            for entity, position in zip(sub.entities, POSITIONS):
                light = entity.components[0]
                np.testing.assert_allclose(light.color, LIGHT * case.intensity)
                # Only the colour moves.
                self.assertEqual(light.position, tuple(position))
                self.assertFalse(light.shadow)

    def test_construction_scales_each_sub_scene_and_restores_the_class(self):
        panel = evaluation.build_panel(plans(1), config(count=1))
        scene, created = self.built(panel, evaluation.case_intensities(panel))
        self.assert_scaled(scene, panel)
        self.assertEqual(created, {"point": 2 * len(panel), "directional": 0,
                                   "ambient": len(panel)})
        self.assert_restored()

    def test_a_rebuild_rescales_the_original_rather_than_compounding(self):
        panel = evaluation.build_panel(plans(1), config(count=1))
        intensities = evaluation.case_intensities(panel)
        scene, _ = self.built(panel, intensities)
        for _ in range(2):
            scene.clear()
            with maniskill_scene_module(self.cls), \
                 evaluation.construction_lighting(intensities):
                scene.build()
            self.assert_scaled(scene, panel)

    def test_selected_scene_and_positional_colour_preserve_other_arguments(self):
        scene = self.cls(3)
        with maniskill_scene_module(self.cls), \
             evaluation.construction_lighting([1.0, 0.4, 2.0]):
            scene.add_point_light(POSITIONS[0], LIGHT, True, scene_idxs=[1])
        self.assertEqual([len(s.entities) for s in scene.sub_scenes], [0, 1, 0])
        light = scene.sub_scenes[1].entities[0].components[0]
        np.testing.assert_allclose(light.color, LIGHT * 0.4)
        self.assertEqual(light.position, tuple(POSITIONS[0]))
        self.assertTrue(light.shadow)
        np.testing.assert_array_equal(LIGHT, [2.0, 1.6, 1.0])
        self.assert_restored()

    def test_a_failed_construction_still_restores_the_class(self):
        panel = evaluation.build_panel(plans(1), config(count=1))
        with maniskill_scene_module(self.cls):
            with self.assertRaises(RuntimeError):
                with evaluation.construction_lighting(
                        evaluation.case_intensities(panel)):
                    raise RuntimeError("scene build blew up")
        self.assert_restored()
        # A later environment -- training, or another task -- is unaffected.
        after = self.cls(3)
        after.build()
        for sub in after.sub_scenes:
            np.testing.assert_allclose(sub.render_system.ambient_light, AMBIENT)
            np.testing.assert_allclose(sub.entities[0].components[0].color, LIGHT)

    def test_a_scene_reaching_past_the_panel_is_refused(self):
        with maniskill_scene_module(self.cls):
            with evaluation.construction_lighting([1.0, 0.4]):
                with self.assertRaises(RuntimeError):
                    self.cls(3).build()
        self.assert_restored()

    def test_an_all_nominal_panel_opens_no_window(self):
        """A's object panel and the training env are never patched at all."""
        objects = {}
        for i in range(5):
            objects.update(plans(1, f"object{i}"))
        panel = evaluation.build_panel(objects, config("objects", 25, False))
        self.assertEqual(evaluation.case_intensities(panel), [])
        for bad in ([], [1.0, 0.0], [1.0, float("nan")]):
            with self.subTest(intensities=bad), maniskill_scene_module(self.cls):
                with self.assertRaises(ValueError):
                    with evaluation.construction_lighting(bad):
                        pass
        self.assert_restored()

    def test_verification_reads_the_built_scene_not_the_intent(self):
        panel = evaluation.build_panel(plans(1), config(count=1))
        scene, _ = self.built(panel, evaluation.case_intensities(panel))
        result = evaluation.verify_construction_lighting(scene, panel)
        self.assertLess(result["eval_light/construction_intensity_max_error"], 1e-9)
        # An unscaled build is exactly the failure this has to catch.
        plain, _ = self.built(panel)
        with self.assertRaises(RuntimeError):
            evaluation.verify_construction_lighting(plain, panel)
        scene.parallel_in_single_scene = True
        with self.assertRaises(RuntimeError):
            evaluation.verify_construction_lighting(scene, panel)
        # Nominal-only panels have nothing to verify.
        self.assertEqual(evaluation.verify_construction_lighting(
            scene, [c for c in panel if c.group != "light"]), {})

    def test_unmatched_state_and_unchanged_rgb_do_not_pass(self):
        import torch
        panel = evaluation.build_panel(plans(1), config(count=1))
        poses = torch.zeros(len(panel), 7)
        poses[:, 3] = 1
        base = NS(agent=NS(robot=NS(pose=NS(raw_pose=poses), qpos=poses, qvel=poses)),
                  subtask_objs=[NS(pose=NS(raw_pose=poses))])
        self.assertEqual(evaluation.check_lighting_reset(base, panel)["eval_light/reset_max_state_difference"], 0)
        poses[1, 0] = .01
        with self.assertRaises(RuntimeError):
            evaluation.check_lighting_reset(base, panel)
        image = torch.zeros(len(panel), 2, 2, 3)
        with self.assertRaises(RuntimeError):
            evaluation.lighting_pixel_metrics({"image_head": image}, panel)
        for i, case in enumerate(panel):
            image[i] = 100 * case.intensity
        values = evaluation.lighting_pixel_metrics({"image_head": image}, panel)
        self.assertEqual(values["eval_light/dim/reset_rgb_mae"], 60)
        self.assertEqual(values["eval_light/bright/reset_rgb_mae"], 100)
        self.assertEqual(values["eval_light/dim/reset_changed_pixel_fraction"], 1)
        self.assertEqual(values["eval_light/nominal/reset_mean_brightness"], 100)
        self.assertEqual(values["eval_light/dim/reset_mean_brightness"], 40)

    def test_a_sub_scene_sized_difference_is_not_accepted_as_illumination(self):
        """The measured cross-sub-scene artifact: ~0.008 MAE over ~2% of pixels."""
        import torch
        panel = evaluation.build_panel(plans(1), config(count=1))
        image = torch.full((len(panel), 40, 40, 3), 100.0)
        for i, case in enumerate(panel):
            if case.condition != "nominal":
                image[i, 0, :2] = 130  # a few pixels move a lot, the scene does not
        with self.assertRaises(RuntimeError) as caught:
            evaluation.lighting_pixel_metrics({"image_head": image}, panel)
        self.assertIn("not illumination", str(caught.exception))
        # Enough pixels, but each barely moving, is refused too.
        image = torch.full((len(panel), 40, 40, 3), 100.0)
        for i, case in enumerate(panel):
            if case.condition != "nominal":
                image[i] += 0.5
        with self.assertRaises(RuntimeError):
            evaluation.lighting_pixel_metrics({"image_head": image}, panel)
        # The response actually measured at build time clears both thresholds.
        for i, case in enumerate(panel):
            image[i] = 100.0 * case.intensity
        values = evaluation.lighting_pixel_metrics({"image_head": image}, panel)
        self.assertGreaterEqual(values["eval_light/dim/reset_rgb_mae"],
                                evaluation.LIGHTING_MIN_RGB_MAE)
        self.assertGreaterEqual(
            values["eval_light/bright/reset_changed_pixel_fraction"],
            evaluation.LIGHTING_MIN_CHANGED_PIXEL_FRACTION)


class ResetIntegrationTest(unittest.TestCase):
    def test_video_rows_use_scene_identity_not_row_zero(self):
        panel = evaluation.build_panel(plans(), config())
        panel[0], panel[2] = panel[2], panel[0]
        rows = evaluation.evaluation_video_rows(panel, ["s00"])
        self.assertEqual(panel[rows["eval/video"]].scene, "s00")
        self.assertEqual(panel[rows["eval/video"]].intensity, 1.0)
        self.assertNotEqual(panel[rows["eval/unseen_scene"]].scene, "s00")
        self.assertEqual(panel[rows["eval/dim_light"]].intensity, 0.4)
        self.assertEqual(panel[rows["eval/dim_light"]].scene, "s00")
        self.assertEqual(evaluation.evaluation_video_rows([], []), {"eval/video": 0})
        same_scene = evaluation.build_panel(plans(1), config(count=1, lighting=False))
        self.assertEqual(evaluation.evaluation_video_rows(same_scene, ["s00"]),
                         {"eval/video": 0})

    def test_actual_adapter_reset_passes_fixed_plans_spawns_and_scenes(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        tree = ast.parse(Path("envs/maniskill.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ManiSkillVecEnv")
        reset = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_reset_simulator")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[reset], type_ignores=[]), "<adapter>", "exec"), namespace)
        rows = [NS(subtasks=[NS(composite_subtask_uids=["uid"])]) for _ in range(12)]
        base = NS(scene_builder=NS(build_config_names_to_idxs={"s00": 7}),
                  build_config_idx_to_task_plans={7: rows},
                  spawn_data={"uid": {"robot_qpos": np.zeros((20, 5))}}, build_config_idxs=[7])
        calls = []
        env = NS(unwrapped=base, reset=lambda **kwargs: calls.append(kwargs) or ({}, {}))
        base.scene = None
        actor = NS(_env=env, _eval_seed=42, _device="cpu", _light_intensities=[],
                   eval_cases=evaluation.build_panel(plans(1), config(count=1, lighting=False)))
        with patch.dict(sys.modules, {"envs.evaluation": evaluation}):
            namespace["_reset_simulator"](actor, initial=True)
            namespace["_reset_simulator"](actor, initial=False)
        self.assertEqual(calls[0]["options"]["build_config_idxs"], [7])
        self.assertTrue(calls[0]["options"]["reconfigure"])
        self.assertEqual(calls[1]["options"]["spawn_selection_idxs"], [0])
        self.assertEqual(calls[1]["options"]["task_plan_idxs"].tolist(), [0])
        self.assertEqual(calls[0]["seed"], calls[1]["seed"])
        # One reset per call now: the extra refresh reset is gone.
        self.assertEqual(len(calls), 2)

    def test_only_a_reconfiguring_reset_opens_the_lighting_window(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        tree = ast.parse(Path("envs/maniskill.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "ManiSkillVecEnv")
        reset = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                     and n.name == "_reset_simulator")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[reset], type_ignores=[]), "<adapter>", "exec"),
             namespace)
        panel = evaluation.build_panel(plans(1), config(count=1))
        rows = [NS(subtasks=[NS(composite_subtask_uids=["uid"])]) for _ in range(12)]
        base = NS(scene_builder=NS(build_config_names_to_idxs={"s00": 7}),
                  build_config_idx_to_task_plans={7: rows}, scene=None,
                  spawn_data={"uid": {"robot_qpos": np.zeros((20, 5))}},
                  build_config_idxs=[7] * len(panel))
        scene_type = scene_class()
        base.scene = scene_type(len(panel))
        poses = torch.zeros(len(panel), 7)
        poses[:, 3] = 1
        base.agent = NS(robot=NS(pose=NS(raw_pose=poses), qpos=poses, qvel=poses))
        base.subtask_objs = [NS(pose=NS(raw_pose=poses))]
        windows, calls = [], []

        def simulator_reset(**kwargs):
            calls.append(kwargs)
            if kwargs["options"].get("reconfigure"):
                base.scene.clear()
                base.scene.build()
            levels = [float(s.render_system.ambient_light[0]) / 0.3 * 100
                      for s in base.scene.sub_scenes]
            image = torch.tensor(levels)[:, None, None, None].expand(-1, 2, 2, 3)
            return {"image_head": image}, {}

        env = NS(unwrapped=base, reset=simulator_reset)
        actor = NS(_env=env, _eval_seed=42, _device="cpu", eval_cases=panel,
                   _light_intensities=evaluation.case_intensities(panel))

        @contextlib.contextmanager
        def spy(intensities):
            windows.append(list(intensities))
            with evaluation.construction_lighting(intensities) as created:
                yield created

        stub = types.SimpleNamespace(
            construction_lighting=spy, check_lighting_reset=evaluation.check_lighting_reset,
            lighting_pixel_metrics=evaluation.lighting_pixel_metrics,
            verify_construction_lighting=evaluation.verify_construction_lighting)
        with maniskill_scene_module(scene_type), \
             patch.dict(sys.modules, {"envs.evaluation": stub}):
            namespace["_reset_simulator"](actor, initial=True)
            self.assertEqual(len(windows), 1)
            self.assertEqual(windows[0], evaluation.case_intensities(panel))
            initial_metrics = dict(actor.eval_reset_metrics)
            light_ids = [id(s.entities[0].components[0]) for s in base.scene.sub_scenes]
            # An ordinary reset rescales nothing and rebuilds nothing.
            for _ in range(2):
                namespace["_reset_simulator"](actor, initial=False)
                self.assertEqual(actor.eval_reset_metrics, initial_metrics)
                self.assertEqual(
                    [id(s.entities[0].components[0]) for s in base.scene.sub_scenes],
                    light_ids)
            self.assertEqual(len(windows), 1)
            self.assertEqual(len(calls), 3)
            # An explicit later reconfigure applies original intensities again.
            namespace["_reset_simulator"](actor, initial=True)
            self.assertEqual(len(windows), 2)
            self.assertEqual(actor.eval_reset_metrics, initial_metrics)


class TransferConfigTest(unittest.TestCase):
    def test_chosen_transfer_is_held_out_and_budget_is_3m(self):
        cfg = yaml.safe_load(Path("configs/configs.yaml").read_text())
        ft = cfg["finetune"]
        self.assertEqual(ft["objects"], ["008_pudding_box"])
        self.assertEqual(ft["steps"], 3_000_000)
        for experiment in ("a", "b"):
            main = yaml.safe_load(Path(f"configs/env/mshab_pick_{experiment}.yaml").read_text())
            objects = main.get("mshab_objects") or [main.get("mshab_obj")]
            self.assertFalse(set(objects).intersection(ft["objects"]))
        self.assertFalse(ft["enabled"])

    def test_real_transfer_handoff_loads_best_weights_with_fresh_training_state(self):
        import copy
        import tempfile
        import torch
        import tools
        from checkpointing import CheckpointConfig, Checkpointer, load_checkpoint, CheckpointError

        tree = ast.parse(Path("train.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_finetune")

        class Agent(torch.nn.Module):
            def __init__(self, *args):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(-1.))
                self.optimizer = torch.optim.SGD(self.parameters(), lr=.1)

            def attach_task_schedule(self, env):
                pass

        logger = NS(scalar=lambda *args: None, write=lambda *args: None,
                    log_hydra_config=lambda *args, **kwargs: None)
        ft = NS(objects=["008_pudding_box"], steps=3_000_000, eval_episode_num=10,
                initialization="pretrained", checkpoint_path="")
        cfg = NS(finetune=ft, seed=0, device="cpu", model=NS(graph=NS(enabled=True)),
                 env=NS(train_build_config_ids=["s00"], eval_lighting=NS(enabled=True)),
                 trainer=NS(), checkpoint=NS(enabled=True), buffer=NS())
        fake_conf = NS(to_container=lambda value, resolve: copy.deepcopy(value), create=lambda value: value)
        seen, closed = [], []
        env = NS(close=lambda: closed.append(1))

        def make_envs(config):
            self.assertEqual(config.mshab_objects, ["008_pudding_box"])
            self.assertFalse(config.eval_lighting.enabled)
            self.assertEqual(config.eval_panel, "objects")
            self.assertEqual(config.eval_build_config_ids, ["s00"])
            return env, env, {}, {}

        class Trainer:
            def __init__(self, config, buffer, *args, checkpointer):
                self.config = config
                self.buffer = buffer
                self.checkpointer = checkpointer

            def begin(self, agent):
                seen.append((float(agent.weight), self.config.steps, self.buffer,
                             len(agent.optimizer.state), self.checkpointer))

        identity = {"graph_schema": "graph"}
        namespace = dict(torch=torch, Dreamer=Agent, make_envs=make_envs, tools=tools,
                         checkpoint_identity=lambda *args: identity,
                         load_checkpoint=load_checkpoint, CheckpointError=CheckpointError,
                         OnlineTrainer=Trainer, Buffer=lambda config: object())
        exec(compile(ast.Module(body=[function], type_ignores=[]), "<transfer>", "exec"), namespace)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"omegaconf": NS(OmegaConf=fake_conf)}):
            keeper = Checkpointer(CheckpointConfig(enabled=True, metric="eval/success_once"), tmp, identity)
            keeper.maybe_save(9_000_000, {"eval/success_once": .8},
                             lambda: {"model": {"weight": torch.tensor(3.)}})
            original = Path(keeper.path).read_bytes()
            namespace["run_finetune"](cfg, logger, Path(tmp), 10_000_000, keeper.path)
            cfg.finetune.initialization = "scratch"
            namespace["run_finetune"](cfg, logger, Path(tmp), 10_000_000, None)
            self.assertEqual(seen[0][0], 3.)
            self.assertEqual(seen[1][0], -1.)
            self.assertIsNot(seen[0][2], seen[1][2])
            for _, steps, _, optimizer_entries, checkpoint in seen:
                self.assertEqual(steps, 3_000_000)
                self.assertEqual(optimizer_entries, 0)
                self.assertIsNone(checkpoint)
            self.assertEqual(Path(keeper.path).read_bytes(), original)
            self.assertEqual(len(closed), 4)


class ActorDistTest(unittest.TestCase):
    """The transfer stage builds a second agent from the same config.

    ``dreamer`` imports torch, so the resolver is exec'd from source the way
    the adapter tests do.
    """

    class Node(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    def setUp(self):
        tree = ast.parse(Path("dreamer.py").read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "resolve_actor_dist")
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<dreamer>", "exec"),
             namespace)
        self.resolve = namespace["resolve_actor_dist"]

    def config(self):
        node = self.Node
        return node(actor=node(dist=node(
            cont=node(name="bounded_normal"), disc=node(name="onehot"),
            multi_disc=node(name="multi_onehot"))))

    def test_a_second_agent_reuses_the_resolution_instead_of_failing(self):
        config, box = self.config(), NS(shape=(8,))
        self.assertFalse(self.resolve(config, box))
        self.assertEqual(config.actor.dist.name, "bounded_normal")
        # The second call is the transfer stage; it used to raise here.
        self.assertFalse(self.resolve(config, box))
        self.assertEqual(config.actor.dist.name, "bounded_normal")

    def test_each_action_space_picks_its_own_branch(self):
        for space, name, discrete in (
                (NS(shape=(8,)), "bounded_normal", False),
                (NS(discrete=True, n=5), "onehot", True),
                (NS(multi_discrete=True, discrete=True, n=5), "multi_onehot", True)):
            with self.subTest(dist=name):
                config = self.config()
                self.assertEqual(self.resolve(config, space), discrete)
                self.assertEqual(config.actor.dist.name, name)

    def test_a_config_resolved_for_another_action_space_is_refused(self):
        """Reuse must not quietly hand a continuous agent a discrete head."""
        config = self.config()
        self.resolve(config, NS(shape=(8,)))
        with self.assertRaises(ValueError):
            self.resolve(config, NS(discrete=True, n=5))
        config = self.config()
        self.resolve(config, NS(discrete=True, n=5))
        with self.assertRaises(ValueError):
            self.resolve(config, NS(shape=(8,)))


class EvaluationLoopTest(unittest.TestCase):
    def test_only_scheduled_episodes_render_without_changing_rollout(self):
        import torch
        import trainer as trainer_module

        class Batch(dict):
            def to(self, *args, **kwargs):
                return self

            def detach(self):
                return self

        rollouts = []
        for period in (100, 0):
            with self.subTest(video_every=period):
                builder = NS(record_graph_env_indices=set(), last_graph_by_env={}, last_masks_by_env={})
                tick, capture, replayed, videos = 0, [], [], []
                def env_step(action, reset):
                    nonlocal tick
                    tick = 0 if reset[0] else tick + 1
                    capture.append(bool(builder.record_graph_env_indices))
                    return Batch(reward=torch.full((1, 1), float(tick)), is_first=reset[:, None]), torch.tensor([tick == 2])
                cfg = NS(steps=6, pretrain=0, eval_every=0, eval_episode_num=0,
                         video_pred_log=False, video_fps=15, params_hist_log=False,
                         batch_length=64, batch_size=1, train_ratio=1, update_log_every=10,
                         video_every=period, action_repeat=1)
                replay = NS(count=lambda: 0, add_transition=lambda trans: replayed.append(float(trans["reward"].item())))
                logger = NS(scalar=lambda *a: None, write=lambda *a: None,
                            video=lambda *a, **kw: videos.append(a[0]))
                env = NS(env_num=1, step=env_step)
                agent = NS(device="cpu", get_initial_state=lambda n: {"prev_action": torch.zeros(n, 1)},
                           act=lambda trans, state, eval: (torch.zeros(1, 1), state))
                runner = trainer_module.OnlineTrainer(cfg, replay, logger, None, env, None)
                with patch.object(trainer_module, "_graph_builder", return_value=builder), \
                     patch.object(trainer_module, "_observation_frame", return_value=torch.zeros(2, 2, 3)) as render:
                    runner.begin(agent)
                    self.assertEqual(render.call_count, 3 if period else 0)
                self.assertEqual(capture, [bool(period)] * 3 + [False] * 6)
                self.assertEqual(videos, ["train_video"] if period else [])
                rollouts.append(replayed)
        self.assertEqual(rollouts[0], rollouts[1])

    def test_final_panel_eval_runs_only_after_normal_budget_completion(self):
        import torch
        import trainer as trainer_module

        class Batch(dict):
            def to(self, *args, **kwargs):
                return self

            def detach(self):
                return self

        def step(action, reset):
            return Batch(reward=torch.zeros(1, 1), is_first=reset[:, None]), torch.zeros(1, dtype=torch.bool)

        cfg = NS(steps=3, pretrain=0, eval_every=100, eval_episode_num=1,
                 video_pred_log=False, video_fps=15, params_hist_log=False,
                 batch_length=64, batch_size=1, train_ratio=1, update_log_every=10,
                 video_every=100, action_repeat=1)
        replay = NS(count=lambda: 0, add_transition=lambda trans: None)
        logger = NS(scalar=lambda *args: None, write=lambda *args: None)
        env = NS(env_num=1, step=step, eval_cases=[object()])
        agent = NS(device="cpu", get_initial_state=lambda n: {"prev_action": torch.zeros(n, 1)},
                   act=lambda trans, state, eval: (torch.zeros(1, 1), state))
        runner = trainer_module.OnlineTrainer(cfg, replay, logger, None, env, env)
        evaluations = []
        def evaluate(agent, at):
            evaluations.append(at)
            runner._last_eval_step = at
            return {}
        runner.eval = evaluate
        with patch.object(trainer_module, "_graph_builder", return_value=None), \
             patch.object(trainer_module, "_observation_frame", return_value=None):
            self.assertEqual(runner.begin(agent), 3)
            self.assertEqual(evaluations, [0, 3])
            evaluations.clear()
            env.step = lambda *args: (_ for _ in ()).throw(KeyboardInterrupt())
            runner._should_eval = lambda step: True
            with self.assertRaises(KeyboardInterrupt):
                runner.begin(agent)
            self.assertEqual(evaluations, [0])

    def test_real_eval_loop_uses_entire_panel_and_restores_training_rng(self):
        import torch
        import tools
        import trainer as trainer_module

        class Batch(dict):
            def to(self, *args, **kwargs):
                return self

        class Env:
            env_num = 93
            training_scenes = ["s00"]
            eval_cases = evaluation.build_panel(plans(), config())
            ticks = 0

            def step(self, action, reset):
                self.ticks += 1
                ended = torch.full((93,), self.ticks == 3)
                success = torch.zeros(93, 1)
                success[63:] = 1
                # Evaluation deliberately consumes RNG; decorator must restore it.
                torch.rand(3)
                return Batch(reward=torch.ones(93, 1), log_success_once=success), ended

        class Agent:
            device = "cpu"
            batch_sizes = []
            training = True

            def eval(self):
                self.training = False

            def train(self):
                self.training = True

            def get_initial_state(self, n):
                return {"prev_action": torch.zeros(n, 2)}

            def act(self, trans, state, eval=False):
                self.batch_sizes.append(len(trans["reward"]))
                return torch.zeros(93, 2), state

        logged = {}
        logger = NS(scalar=lambda key, value: logged.update({key: float(value)}), write=lambda step: None)
        runner = trainer_module.OnlineTrainer.__new__(trainer_module.OnlineTrainer)
        runner.eval_envs, runner.logger = Env(), logger
        runner.video_pred_log, runner.batch_length, runner.video_fps = False, 64, 15
        runner.eval_video_log, runner.profile_timing = False, True
        agent = Agent()
        torch.manual_seed(12)
        before = torch.get_rng_state().clone()
        with patch.dict(sys.modules, {"envs.evaluation": evaluation}), \
             patch.object(trainer_module, "_graph_builder", return_value=None), \
             patch.object(trainer_module, "_render_frame", return_value=None) as render, \
             patch.object(trainer_module, "_observation_frame", return_value=None) as fallback:
            result = runner.eval(agent, 50000)
        render.assert_not_called()
        fallback.assert_not_called()
        self.assertGreaterEqual(result["eval_timing/env_step_ms"], 0)
        self.assertEqual(agent.batch_sizes, [93, 93, 93])
        self.assertEqual(result["eval/success_once"], 0)
        self.assertEqual(result["eval_light/bright/success_once"], 1)
        self.assertEqual(result["eval/episodes"], 63)
        self.assertTrue(agent.training)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(runner._last_eval_step, 50000)

        videos, selected_rows = {}, []
        runner.eval_video_log = True
        runner.eval_envs.ticks = 0
        runner.logger.video = lambda key, value, **kw: videos.update({key: value})

        def render_selected(indices):
            selected_rows.append(indices)
            return torch.stack([torch.full((2, 2, 3), i, dtype=torch.uint8)
                                for i in indices])

        runner.eval_envs.render_selected = render_selected
        with patch.dict(sys.modules, {"envs.evaluation": evaluation}), \
             patch.object(trainer_module, "_graph_builder", return_value=None):
            runner.eval(agent, 51000)
        rows = evaluation.evaluation_video_rows(Env.eval_cases, Env.training_scenes)
        self.assertEqual(selected_rows, [list(rows.values())] * 3)
        self.assertEqual(set(videos), set(rows))
        for key, index in rows.items():
            self.assertEqual(videos[key].shape, (1, 3, 2, 2, 3))
            self.assertTrue((videos[key] == index).all())

        parent = NS(scalar=lambda k, v: logged.update({k: v}), write=lambda step: logged.update(global_step=step))
        stage = tools.StageLogger(parent, "finetune", 10_000_000)
        stage.scalar("eval/success_once", .7)
        stage.write(20000)
        self.assertEqual(logged["finetune/eval/success_once"], .7)
        self.assertEqual(logged["finetune/step"], 20000)
        self.assertEqual(logged["global_step"], 10_020_000)
        self.assertIn("finetune/eval/success_once", tools.wandb_scalars(list(logged.items())))


if __name__ == "__main__":
    unittest.main()
