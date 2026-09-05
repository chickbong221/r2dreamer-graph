"""One contract suite for fixed B+C evaluation and transfer orchestration."""

import ast
from collections import Counter
import importlib.util
from pathlib import Path
import sys
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
        result = evaluation.panel_metrics(panel, {"success_once": success}, ["s00"])
        self.assertAlmostEqual(result["eval/success_once"], 1 / 63)
        self.assertEqual(result["eval_scene/training/success_once"], 1)
        self.assertEqual(result["eval_scene/held_out/success_once"], 0)
        self.assertEqual(result["eval_light/dim/episodes"], 10)

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


class LightingTest(unittest.TestCase):
    def scene(self, count):
        light_type = type("RenderPointLightComponent", (), {})
        scenes = []
        for _ in range(count):
            light = light_type()
            light.color = np.array([2., 1.6, 1.])
            scenes.append(NS(render_system=NS(ambient_light=np.array([.3] * 3)),
                             entities=[NS(components=[light])]))
        return NS(sub_scenes=scenes, parallel_in_single_scene=False)

    def test_only_intensity_changes_and_reset_does_not_compound_it(self):
        panel = evaluation.build_panel(plans(1), config(count=1))
        scene = self.scene(len(panel))
        controller = evaluation.LightingController()
        for _ in range(2):
            controller.apply(scene, panel)
            for sub, case in zip(scene.sub_scenes, panel):
                np.testing.assert_allclose(sub.render_system.ambient_light, [.3 * case.intensity] * 3)
                np.testing.assert_allclose(sub.entities[0].components[0].color,
                                           np.array([2., 1.6, 1.]) * case.intensity)
        scene.parallel_in_single_scene = True
        with self.assertRaises(RuntimeError):
            controller.apply(scene, panel)

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


class ResetIntegrationTest(unittest.TestCase):
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
        controller = NS(apply=lambda *args: None)
        base.scene = None
        actor = NS(_env=env, _eval_seed=42, _device="cpu", _lighting_controller=controller,
                   eval_cases=evaluation.build_panel(plans(1), config(count=1, lighting=False)))
        with patch.dict(sys.modules, {"envs.evaluation": evaluation}):
            namespace["_reset_simulator"](actor, initial=True)
            namespace["_reset_simulator"](actor, initial=False)
        self.assertEqual(calls[0]["options"]["build_config_idxs"], [7])
        self.assertTrue(calls[0]["options"]["reconfigure"])
        self.assertEqual(calls[1]["options"]["spawn_selection_idxs"], [0])
        self.assertEqual(calls[1]["options"]["task_plan_idxs"].tolist(), [0])
        self.assertEqual(calls[0]["seed"], calls[1]["seed"])


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


class EvaluationLoopTest(unittest.TestCase):
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
        agent = Agent()
        torch.manual_seed(12)
        before = torch.get_rng_state().clone()
        with patch.dict(sys.modules, {"envs.evaluation": evaluation}), \
             patch.object(trainer_module, "_graph_builder", return_value=None), \
             patch.object(trainer_module, "_render_frame", return_value=None), \
             patch.object(trainer_module, "_observation_frame", return_value=None):
            result = runner.eval(agent, 50000)
        self.assertEqual(agent.batch_sizes, [93, 93, 93])
        self.assertEqual(result["eval/success_once"], 0)
        self.assertEqual(result["eval_light/bright/success_once"], 1)
        self.assertEqual(result["eval/episodes"], 63)
        self.assertTrue(agent.training)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(runner._last_eval_step, 50000)

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
