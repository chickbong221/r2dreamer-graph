"""Success-gated motion-planning rollouts, observed step by step.

ManiSkill ships a scripted solution per task and ``collect_maniskill_interactions``
already drives them to mine relation evidence. This does the same driving for a
different consumer: it hands every control step to a callback and reports whether
the episode succeeded, so a caller can throw the failures away.

Whether an episode succeeded is only known at its end, and a thousand-pixel
frame per step is far too much to hold until then. The callback is therefore
expected to write as it goes, and the caller deletes what the attempt did not
earn -- see :mod:`scenegraph.figures.writer`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

# Both are small and both already have exactly one definition in this repo:
# ``success_flag`` is what "did this episode succeed" means for the miner, and a
# figure that disagreed with it would illustrate a different set of episodes.
from ..tools.collect_maniskill_interactions import get_solver, success_flag

ResetHook = Callable[[dict], None]
StepHook = Callable[[dict, dict], None]


@dataclass
class Attempt:
    """What one seed produced."""

    seed: int
    success: bool
    steps: int
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "seed": int(self.seed),
            "success": bool(self.success),
            "steps": int(self.steps),
            "error": self.error,
        }


def capture_wrapper(env, *, on_reset: Optional[ResetHook] = None,
                    on_step: Optional[StepHook] = None):
    """Wrap ``env`` so every reset and control step reaches the hooks.

    The scripted solutions call ``planner.env.step``, and the planner is built
    from whatever ``solve`` was handed, so wrapping at this level is enough --
    the same trick ``collect_maniskill_interactions.make_env`` uses.
    """
    import gymnasium as gym

    class _Capture(gym.Wrapper):
        def reset(self, **kwargs):
            out = self.env.reset(**kwargs)
            if on_reset is not None:
                on_reset(out[0])
            return out

        def step(self, action):
            out = self.env.step(action)
            if on_step is not None:
                on_step(out[0], out[4])
            return out

    return _Capture(env)


class MotionPlanRunner:
    """Runs a task's scripted solution once per seed, reporting each outcome.

    Success is tracked as "succeeded at least once", matching the miner: a task
    that reaches its goal and is then nudged out of it by the release still
    demonstrated the behaviour the figure is of.
    """

    def __init__(self, env, env_id: str, *, on_reset: Optional[ResetHook] = None,
                 on_step: Optional[StepHook] = None):
        self._on_reset = on_reset
        self._on_step = on_step
        self.env = capture_wrapper(
            env, on_reset=self._handle_reset, on_step=self._handle_step
        )
        self.solve = get_solver(str(env_id))
        self._success = False
        self._steps = 0

    def _handle_reset(self, obs: dict) -> None:
        self._success = False
        self._steps = 0
        if self._on_reset is not None:
            self._on_reset(obs)

    def _handle_step(self, obs: dict, info: dict) -> None:
        self._steps += 1
        self._success = self._success or success_flag(info)
        if self._on_step is not None:
            self._on_step(obs, info)

    def attempt(self, seed: int) -> Attempt:
        """One episode from ``seed``. A solver failure is an outcome, not a
        crash: planning is stochastic and a caller asking for N successes has
        to be able to walk to the next seed."""
        # Cleared here as well as on reset: a solver that fails before it gets
        # to ``env.reset`` would otherwise inherit the last episode's verdict
        # and a figure would be written for an episode that never ran.
        self._success = False
        self._steps = 0
        error = None
        try:
            self.solve(self.env, seed=int(seed), debug=False, vis=False)
        except Exception as exc:                           # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        return Attempt(int(seed), bool(self._success), int(self._steps), error)
