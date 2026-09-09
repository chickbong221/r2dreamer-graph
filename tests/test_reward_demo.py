"""Reward logging over a scripted motion-planning episode.

No simulator. The env is a script of ``(reward, success)`` pairs returned in
ManiSkill's batched shape, and the solver is a function that resets and steps
it -- what can go wrong here is arithmetic (a return that double-counts the
first step, a discount applied off by one), episode boundaries (a trace that
carries the previous attempt's steps, or counts steps the task already
considers over), and file layout (a failed attempt written as a successful
one's table).
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scenegraph.tools import demo_motionplanning_reward as demo
from scenegraph.tools.demo_motionplanning_reward import (
    RewardTrace, draw_return_figure, figure_title, printed_steps,
    read_csv_trace, scalar,
)


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #
class _Tensor:
    """What ManiSkill returns even at batch size one."""

    def __init__(self, value):
        self._value = np.asarray([value])

    def cpu(self):
        return self._value


class _GymStub:
    """Just enough ``gymnasium`` for ``capture_wrapper`` to subclass."""

    class Wrapper:
        def __init__(self, env):
            self.env = env

        def __getattr__(self, name):
            return getattr(self.env, name)


class _ScriptEnv:
    """Replays a per-seed script of ``(reward, success)`` pairs.

    ``terminated`` follows success the way a task with a terminal goal does;
    ``truncated`` fires at ``horizon``, and the env keeps stepping past it,
    which is exactly what a scripted plan longer than the horizon does.
    """

    def __init__(self, scripts, horizon=None, fail_at=None):
        self.scripts = scripts
        self.horizon = horizon
        self.fail_at = fail_at
        self.seed = None
        self.step_index = 0
        self.closed = False

    @property
    def script(self):
        return self.scripts[self.seed]

    def reset(self, seed=None, **kwargs):
        self.seed = 0 if seed is None else int(seed)
        self.step_index = 0
        return {}, {}

    def step(self, action):
        del action
        reward, success = self.script[self.step_index]
        self.step_index += 1
        if self.fail_at is not None and self.step_index >= self.fail_at:
            raise RuntimeError("planning failed")
        truncated = bool(self.horizon and self.step_index >= self.horizon)
        return ({}, _Tensor(reward), _Tensor(success), _Tensor(truncated),
                {"success": _Tensor(success)})

    def close(self):
        self.closed = True

    def solve(self, env, seed=None, debug=False, vis=False):
        """The scripted solution's contract: reset, then step the plan out."""
        del debug, vis
        env.reset(seed=seed)
        for _ in range(len(self.script)):
            env.step(np.zeros(2))


def _trace(rewards, successes=(), discount=0.5):
    trace = RewardTrace(discount=discount)
    successes = set(successes)
    for index, reward in enumerate(rewards, start=1):
        success = index in successes
        trace.observe({}, reward, False, False, {"success": success})
    return trace


# --------------------------------------------------------------------------- #
# The arithmetic
# --------------------------------------------------------------------------- #
class TestRewardTrace(unittest.TestCase):
    def test_scalar_unwraps_every_shape_a_task_returns(self):
        self.assertAlmostEqual(scalar(_Tensor(0.25)), 0.25)
        self.assertAlmostEqual(scalar(0.5), 0.5)
        self.assertAlmostEqual(scalar(np.float32(0.75)), 0.75)
        self.assertAlmostEqual(scalar(np.array([1.5])), 1.5)

    def test_return_is_the_running_sum_and_discount_starts_at_one(self):
        trace = _trace([0.1, 0.2, 0.3, 0.4], successes=[3, 4])
        self.assertEqual([s.step for s in trace.steps], [1, 2, 3, 4])
        self.assertAlmostEqual(trace.steps[-1].ret, 1.0)
        # The first reward is undiscounted: gamma^0, not gamma^1.
        self.assertAlmostEqual(trace.steps[0].discounted, 0.1)
        self.assertAlmostEqual(trace.steps[-1].discounted, 0.325)

    def test_summary_splits_the_return_at_the_horizon(self):
        summary = _trace([0.1, 0.2, 0.3, 0.4], successes=[3, 4]).summary(2)
        self.assertAlmostEqual(summary["return"], 1.0)
        self.assertAlmostEqual(summary["return_within_horizon"], 0.3)
        self.assertEqual(summary["steps_past_horizon"], 2)
        # Success happened, but two steps after the episode would have ended.
        self.assertTrue(summary["success"])
        self.assertEqual(summary["first_success_step"], 3)
        self.assertFalse(summary["success_within_horizon"])
        self.assertAlmostEqual(summary["return_to_first_success"], 0.6)
        self.assertTrue(summary["success_at_last_step"])

    def test_summary_without_a_horizon_reports_one_return(self):
        summary = _trace([1.0, 2.0], successes=[2]).summary(None)
        self.assertEqual(summary["horizon"], None)
        self.assertEqual(summary["steps_past_horizon"], 0)
        self.assertAlmostEqual(summary["return_within_horizon"],
                               summary["return"])
        self.assertTrue(summary["success_within_horizon"])

    def test_a_failed_episode_reports_no_success_step(self):
        summary = _trace([0.0, 0.0, 0.1]).summary(10)
        self.assertFalse(summary["success"])
        self.assertIsNone(summary["first_success_step"])
        self.assertIsNone(summary["return_to_first_success"])
        self.assertAlmostEqual(summary["reward_max"], 0.1)

    def test_reset_drops_the_previous_episode(self):
        trace = _trace([1.0, 1.0], successes=[2])
        trace.reset()
        trace.observe({}, 0.5, False, False, {})
        self.assertEqual(len(trace.steps), 1)
        self.assertAlmostEqual(trace.steps[0].ret, 0.5)
        self.assertAlmostEqual(trace.steps[0].discounted, 0.5)

    def test_printed_steps_keep_what_a_stride_would_miss(self):
        trace = _trace([0.0] * 25, successes=[7])
        steps = [s.step for s in printed_steps(trace, every=10, horizon=20)]
        self.assertEqual(steps, [1, 7, 10, 20, 25])

    def test_csv_carries_one_row_per_step(self):
        trace = _trace([0.1, 0.2], successes=[2])
        with tempfile.TemporaryDirectory() as tmp:
            path = trace.write_csv(Path(tmp) / "nested" / "trace.csv")
            lines = path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(
            lines[0].split(","),
            ["step", "reward", "return", "discounted_return", "success",
             "terminated", "truncated"],
        )
        self.assertEqual(lines[2].split(",")[:3], ["2", "0.2", "0.3"])


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #
class TestFigure(unittest.TestCase):
    """A real matplotlib render each time: what breaks a figure is a value the
    drawing code cannot take, and only drawing it finds those."""

    def test_a_png_is_written_for_a_successful_episode(self):
        trace = _trace([0.1] * 12, successes=[9], discount=0.9)
        with tempfile.TemporaryDirectory() as tmp:
            path = draw_return_figure(
                trace, Path(tmp) / "nested" / "return.png", horizon=10,
                title=figure_title("PegInsertionSide-v1", 3, "normalized_dense"),
                dpi=110)
            self.assertTrue(path.exists())
            head = path.read_bytes()[:8]
        self.assertEqual(head, bytes.fromhex("89504e470d0a1a0a"))
        # ^ the PNG magic number, spelled without escapes

    def test_an_episode_with_no_success_and_no_overrun_still_draws(self):
        # Both vertical rules are absent here, which is the branch a figure
        # drawn only from successful episodes never exercises.
        trace = _trace([0.0, 0.1, 0.2])
        with tempfile.TemporaryDirectory() as tmp:
            path = draw_return_figure(trace, Path(tmp) / "flat.png",
                                      horizon=100, dpi=90)
        self.assertIsNotNone(path)

    def test_an_empty_trace_is_declined_rather_than_drawn(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "empty.png"
            self.assertIsNone(draw_return_figure(RewardTrace(), target))
            self.assertFalse(target.exists())

    def test_title_names_the_episode_in_ascii(self):
        title = figure_title("PegInsertionSide-v1", 7, "sparse")
        self.assertEqual(title, "PegInsertionSide-v1, seed 7, sparse")
        self.assertTrue(title.isascii())


class TestRedraw(unittest.TestCase):
    def test_csv_round_trips_every_column(self):
        trace = _trace([0.1, 0.2, 0.3], successes=[3], discount=0.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = trace.write_csv(Path(tmp) / "trace.csv")
            back = read_csv_trace(path, discount=0.5)
        self.assertEqual([s.row() for s in back.steps],
                         [s.row() for s in trace.steps])
        self.assertEqual(back.summary(2), trace.summary(2))

    def test_a_read_back_trace_keeps_accumulating(self):
        trace = _trace([1.0, 1.0], discount=1.0)
        with tempfile.TemporaryDirectory() as tmp:
            back = read_csv_trace(trace.write_csv(Path(tmp) / "t.csv"), 1.0)
        back.observe({}, 1.0, False, False, {})
        self.assertAlmostEqual(back.steps[-1].ret, 3.0)
        self.assertEqual(back.steps[-1].step, 3)

    def test_from_csv_redraws_without_touching_the_simulator(self):
        env = _ScriptEnv({0: [(0.2, False), (0.3, True)]})
        with tempfile.TemporaryDirectory() as tmp:
            args = demo.parse_args(["--env-id", "Fake-v1", "--out", tmp,
                                    "--no-plot"])
            with mock.patch.dict(sys.modules, {"gymnasium": _GymStub}),                     mock.patch("scenegraph.figures.rollout.get_solver",
                               return_value=env.solve),                     mock.patch.object(demo, "make_demo_env",
                                      return_value=(env, "normalized_dense")),                     mock.patch.object(demo, "episode_horizon",
                                      return_value=1):
                demo.run(args)
            written = Path(tmp) / "Fake-v1" / "seed0000_success.csv"
            self.assertFalse(written.with_suffix(".png").exists())

            # No env, no solver, no gymnasium: only the CSV and its sidecar.
            code = demo.main(["--from-csv", str(written), "--dpi", "90"])
            self.assertEqual(code, 0)
            self.assertTrue(written.with_suffix(".png").exists())

    def test_from_csv_takes_the_discount_from_the_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "seed0000_success.csv"
            _trace([1.0, 1.0], successes=[2], discount=0.5).write_csv(source)
            source.with_suffix(".json").write_text(json.dumps({
                "env_id": "PegInsertionSide-v1",
                "reward_mode": "sparse",
                "attempt": {"seed": 4},
                "summary": {"discount": 0.5, "horizon": 1},
            }), encoding="utf-8")
            target = Path(tmp) / "elsewhere" / "return.png"
            code = demo.main(["--from-csv", str(source), "--figure",
                              str(target), "--dpi", "90"])
            self.assertEqual(code, 0)
            self.assertTrue(target.exists())

    def test_a_missing_csv_is_an_error_not_an_empty_figure(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                demo.main(["--from-csv", str(Path(tmp) / "absent.csv")])


# --------------------------------------------------------------------------- #
# The wiring
# --------------------------------------------------------------------------- #
class TestRunnerWiring(unittest.TestCase):
    """The hook has to see the reward, which ``on_step`` does not carry."""

    def _runner(self, env, trace):
        from scenegraph.figures.rollout import MotionPlanRunner

        with mock.patch.dict(sys.modules, {"gymnasium": _GymStub}), \
                mock.patch("scenegraph.figures.rollout.get_solver",
                           return_value=env.solve):
            return MotionPlanRunner(
                env, "Fake-v1",
                on_reset=lambda obs: trace.reset(),
                on_transition=trace.observe,
            )

    def test_every_control_step_reaches_the_trace(self):
        env = _ScriptEnv({0: [(0.1, False), (0.2, False), (0.3, True)]})
        trace = RewardTrace(discount=1.0)
        attempt = self._runner(env, trace).attempt(0)
        self.assertTrue(attempt.success)
        self.assertEqual(attempt.steps, 3)
        self.assertEqual(len(trace.steps), 3)
        self.assertAlmostEqual(trace.steps[-1].ret, 0.6)
        self.assertEqual(trace.first_success_step(), 3)

    def test_the_next_attempt_starts_from_an_empty_trace(self):
        env = _ScriptEnv({
            0: [(1.0, False), (1.0, False)],
            1: [(0.5, True)],
        })
        trace = RewardTrace(discount=1.0)
        runner = self._runner(env, trace)
        runner.attempt(0)
        second = runner.attempt(1)
        self.assertTrue(second.success)
        self.assertEqual([s.reward for s in trace.steps], [0.5])

    def test_terminal_flags_are_recorded_per_step(self):
        env = _ScriptEnv({0: [(0.1, False)] * 4}, horizon=2)
        trace = RewardTrace(discount=1.0)
        self._runner(env, trace).attempt(0)
        self.assertEqual(trace.first_flag("truncated"), 2)
        # The plan kept going past the horizon; the summary says by how much.
        self.assertEqual(trace.summary(2)["steps_past_horizon"], 2)

    def test_a_solver_failure_keeps_the_steps_it_did_take(self):
        env = _ScriptEnv({0: [(0.1, False)] * 5}, fail_at=3)
        trace = RewardTrace(discount=1.0)
        attempt = self._runner(env, trace).attempt(0)
        self.assertFalse(attempt.success)
        self.assertIn("planning failed", attempt.error)
        # The step that raised never returned a transition, so the trace holds
        # the two that completed -- a partial episode, not a lost one.
        self.assertEqual(len(trace.steps), 2)


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
class TestRun(unittest.TestCase):
    def _run(self, env, extra=(), horizon=4):
        with tempfile.TemporaryDirectory() as tmp:
            args = demo.parse_args(
                ["--env-id", "Fake-v1", "--out", tmp, "--episodes", "1",
                 "--max-attempts", "3", "--print-every", "2", *extra]
            )
            with mock.patch.dict(sys.modules, {"gymnasium": _GymStub}), \
                    mock.patch("scenegraph.figures.rollout.get_solver",
                               return_value=env.solve), \
                    mock.patch.object(
                        demo, "make_demo_env",
                        return_value=(env, "normalized_dense")), \
                    mock.patch.object(demo, "episode_horizon",
                                      return_value=horizon):
                code = demo.run(args)
            out = Path(tmp) / "Fake-v1"
            files = sorted(p.name for p in out.glob("*")) if out.exists() else []
            index = out / "episodes.json"
            payload = (json.loads(index.read_text(encoding="utf-8"))
                       if index.exists() else None)
        return code, files, payload

    def test_only_the_successful_episode_is_written(self):
        env = _ScriptEnv({
            0: [(0.1, False), (0.1, False)],          # seed 0 never succeeds
            1: [(0.2, False), (0.3, True)],
        })
        code, files, payload = self._run(env)
        self.assertEqual(code, 0)
        # The figure is drawn by default: it is the point of the demo.
        self.assertEqual(files,
                         ["episodes.json", "seed0001_success.csv",
                          "seed0001_success.json", "seed0001_success.png"])
        self.assertEqual(len(payload), 1)
        summary = payload[0]["summary"]
        self.assertEqual(summary["first_success_step"], 2)
        self.assertAlmostEqual(summary["return"], 0.5)
        self.assertTrue(env.closed)

    def test_keep_failures_names_the_outcome_in_the_file(self):
        env = _ScriptEnv({
            0: [(0.1, False)],
            1: [(0.4, True)],
        })
        code, files, payload = self._run(env, extra=["--keep-failures"])
        self.assertEqual(code, 0)
        self.assertIn("seed0000_failed.csv", files)
        self.assertIn("seed0001_success.csv", files)
        self.assertEqual([entry["attempt"]["success"] for entry in payload],
                         [False, True])

    def test_no_success_is_a_nonzero_exit(self):
        env = _ScriptEnv({seed: [(0.1, False)] for seed in range(3)})
        code, files, payload = self._run(env)
        self.assertEqual(code, 1)
        self.assertEqual(files, [])
        self.assertIsNone(payload)

    def test_the_written_json_records_the_reward_mode_that_applied(self):
        env = _ScriptEnv({0: [(1.0, True)]})
        with tempfile.TemporaryDirectory() as tmp:
            args = demo.parse_args(["--env-id", "Fake-v1", "--out", tmp])
            with mock.patch.dict(sys.modules, {"gymnasium": _GymStub}), \
                    mock.patch("scenegraph.figures.rollout.get_solver",
                               return_value=env.solve), \
                    mock.patch.object(demo, "make_demo_env",
                                      return_value=(env, "sparse")), \
                    mock.patch.object(demo, "episode_horizon",
                                      return_value=100):
                demo.run(args)
            written = json.loads(
                (Path(tmp) / "Fake-v1" / "seed0000_success.json")
                .read_text(encoding="utf-8"))
        # Requested and effective are both kept: a task that has no dense
        # reward is a different learning problem, not a footnote.
        self.assertEqual(written["reward_mode"], "sparse")
        self.assertEqual(written["reward_mode_requested"], "normalized_dense")
        self.assertEqual(written["horizon"], 100)
        self.assertEqual(len(written["steps"]), 1)


if __name__ == "__main__":
    unittest.main()
