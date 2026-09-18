"""Window alignment, chunk masking, replay parity, and the action timeline.

These are the properties that were wrong and silent: every one of them let
training run, produced a descending loss, and taught the wrong thing. They are
tested against behaviour -- the arrays a window actually contains -- rather
than against the source that builds them.

The numpy half runs anywhere. The torch half is skipped without torch and says
so; a skip here establishes nothing about training correctness.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from sim_vla.data import layout
from sim_vla.data.layout import Slice, assemble, conditioning_rows, rows

ACTION = 3


def episode(steps: int):
    """An episode whose every action says which index it came from.

    ``actions[i] = [i, -i, i/10]`` -- nonconstant, and readable back, so an
    off-by-one in the target alignment is visible rather than plausible.
    """
    index = np.arange(steps, dtype=np.float32)
    actions = np.stack([index, -index, index / 10.0], axis=-1)
    rewards = (index + 1.0).astype(np.float32)
    observations = np.stack([index, index], axis=-1)
    observations = np.concatenate(
        [observations, observations[-1:] + 1.0], axis=0).astype(np.float32)
    return actions, rewards, observations


def cut(actions, rewards, observations, *, start, stop, length, burn_in, burn,
        episode_end, lookahead=0):
    """One window of that episode, assembled exactly as a loader would."""
    steps = actions.shape[0]
    ahead_stop = min(stop + int(lookahead), steps)
    ahead = actions[stop:ahead_stop] if ahead_stop > stop else None
    return assemble(
        observations={"proprio": observations[start:stop + 1]},
        actions=actions[start:stop], rewards=rewards[start:stop],
        prev_action=actions[start - 1] if start > 0 else None,
        prev_reward=float(rewards[start - 1]) if start > 0 else None,
        piece=Slice(start=start, real=stop - start, burn=burn,
                    episode_end=stop >= steps),
        length=length, burn_in=burn_in, lookahead=int(lookahead),
        lookahead_actions=ahead)


class TestWindowTargets(unittest.TestCase):
    """``action`` is a_(t-1), ``action_target`` is a_t, and neither is faked."""

    def setUp(self):
        self.actions, self.rewards, self.observations = episode(40)

    def test_previous_action_is_the_action_before_the_row(self):
        window = cut(self.actions, self.rewards, self.observations,
                     start=10, stop=18, length=8, burn_in=2, burn=2,
                     episode_end=False)
        # Row i holds a_(start+i-1); row 0 holds what preceded the window.
        np.testing.assert_allclose(window["action"][0], self.actions[9])
        for row in range(1, 9):
            np.testing.assert_allclose(window["action"][row],
                                       self.actions[9 + row])

    def test_action_target_is_the_action_taken_at_the_row(self):
        window = cut(self.actions, self.rewards, self.observations,
                     start=10, stop=18, length=8, burn_in=2, burn=2,
                     episode_end=False)
        for row in range(8):
            np.testing.assert_allclose(window["action_target"][row],
                                       self.actions[10 + row])

    def test_target_and_previous_action_differ(self):
        """The bug was supervising on ``action``: the posterior's own input."""
        window = cut(self.actions, self.rewards, self.observations,
                     start=10, stop=18, length=8, burn_in=2, burn=2,
                     episode_end=False)
        available = window["action_valid"]
        self.assertTrue(available[:8].all())
        self.assertFalse(
            np.allclose(window["action"][:8], window["action_target"][:8]),
            "action and action_target are identical; one of them is wrong")

    def test_interior_window_final_row_has_no_loaded_target(self):
        """The regression: it was marked valid on every interior window."""
        window = cut(self.actions, self.rewards, self.observations,
                     start=10, stop=18, length=8, burn_in=2, burn=2,
                     episode_end=False, lookahead=0)
        # Row 8 is the window's last observation. No action was loaded for it.
        self.assertFalse(bool(window["action_valid"][8]),
                         "an unloaded action target was marked available")
        self.assertFalse(bool(conditioning_rows(window)[8]))

    def test_episode_end_window_final_row_has_no_loaded_target(self):
        window = cut(self.actions, self.rewards, self.observations,
                     start=32, stop=40, length=8, burn_in=2, burn=2,
                     episode_end=True)
        self.assertFalse(bool(window["action_valid"][8]))
        self.assertTrue(bool(window["is_last"][8]))

    def test_lookahead_supplies_the_missing_targets(self):
        window = cut(self.actions, self.rewards, self.observations,
                     start=10, stop=18, length=8, burn_in=2, burn=2,
                     episode_end=False, lookahead=2)
        for offset in range(2):
            self.assertTrue(bool(window["action_valid"][8 + offset]))
            np.testing.assert_allclose(window["action_target"][8 + offset],
                                       self.actions[18 + offset])
        # And it stops there: 8 loaded + 2 ahead = rows 0..9 of 11.
        self.assertEqual(int(window["action_valid"].sum()), 10)
        self.assertFalse(bool(window["action_valid"][10]))

    def test_lookahead_never_crosses_the_episode_boundary(self):
        window = cut(self.actions, self.rewards, self.observations,
                     start=34, stop=40, length=8, burn_in=2, burn=2,
                     episode_end=True, lookahead=5)
        # The episode has 40 actions; nothing past index 39 exists.
        self.assertFalse(bool(window["action_valid"][6]))

    def chunk_lengths(self, window, chunk):
        """How many real targets each eligible conditioning row would get."""
        eligible = conditioning_rows(window)
        available = np.asarray(window["action_valid"], dtype=bool)
        out = []
        for row in np.nonzero(eligible)[0]:
            run = 0
            for offset in range(chunk):
                index = row + offset
                if index >= available.shape[0] or not available[index]:
                    break
                run += 1
            out.append(run)
        return out

    def test_a_full_interior_window_supervises_whole_chunks(self):
        """The reproduction: with H=4 a full interior window used to give
        ``[4, 4, 3, 2, 1]`` at its tail, because the lookahead was capped by
        the observation window's spare capacity -- one slot, whatever H was."""
        chunk = 4
        window = cut(self.actions, self.rewards, self.observations,
                     start=10, stop=18, length=8, burn_in=2, burn=2,
                     episode_end=False, lookahead=chunk)
        lengths = self.chunk_lengths(window, chunk)
        self.assertTrue(lengths, "no eligible conditioning rows at all")
        self.assertEqual(set(lengths), {chunk},
                         f"truncated chunks on a full interior window: "
                         f"{lengths}")

    def test_a_window_at_the_episode_end_truncates_honestly(self):
        """The counterpart: there really are no actions past the last one, and
        the tail must be masked rather than invented."""
        chunk = 4
        window = cut(self.actions, self.rewards, self.observations,
                     start=30, stop=40, length=10, burn_in=2, burn=2,
                     episode_end=True, lookahead=chunk)
        lengths = self.chunk_lengths(window, chunk)
        self.assertEqual(lengths[-1], 1)
        self.assertEqual(lengths[0], chunk)
        # Strictly decreasing at the tail, never padded back up.
        self.assertEqual(lengths[-chunk:], [4, 3, 2, 1])

    def test_lookahead_adds_no_observations(self):
        plain = cut(self.actions, self.rewards, self.observations,
                    start=10, stop=18, length=8, burn_in=2, burn=2,
                    episode_end=False, lookahead=0)
        ahead = cut(self.actions, self.rewards, self.observations,
                    start=10, stop=18, length=8, burn_in=2, burn=2,
                    episode_end=False, lookahead=4)
        # Lookahead is action-only: no future observation may reach the latent.
        np.testing.assert_allclose(plain["proprio"], ahead["proprio"])
        np.testing.assert_array_equal(plain["valid"], ahead["valid"])
        np.testing.assert_array_equal(plain["loss_mask"], ahead["loss_mask"])

    def test_burn_in_rows_are_not_conditioning_rows(self):
        window = cut(self.actions, self.rewards, self.observations,
                     start=10, stop=18, length=8, burn_in=2, burn=2,
                     episode_end=False, lookahead=3)
        eligible = conditioning_rows(window)
        self.assertFalse(eligible[:2].any(),
                         "a burn-in row was offered as a conditioning row")
        self.assertTrue(eligible[2:8].all())

    def test_reset_row_has_no_incoming_reward(self):
        window = cut(self.actions, self.rewards, self.observations,
                     start=0, stop=8, length=8, burn_in=2, burn=0,
                     episode_end=False)
        self.assertFalse(bool(window["reward_valid"][0]))
        self.assertTrue(bool(window["is_first"][0]))

    def test_every_window_has_the_same_row_count(self):
        observation_axis, target_axis = set(), set()
        for start, stop, burn in ((0, 8, 0), (10, 18, 2), (34, 40, 2)):
            window = cut(self.actions, self.rewards, self.observations,
                         start=start, stop=stop, length=8, burn_in=2,
                         burn=burn, episode_end=stop >= 40, lookahead=3)
            layout.check(window, 8, 2, 3)
            observation_axis.add(window["action"].shape[0])
            target_axis.add(window["action_target"].shape[0])
        # Two axes, each one shape: that is what lets two sources stack.
        self.assertEqual(observation_axis, {rows(8, 2)})
        self.assertEqual(target_axis, {layout.target_rows(8, 2, 3)})

    def test_padding_is_never_a_valid_observation(self):
        window = cut(self.actions, self.rewards, self.observations,
                     start=36, stop=40, length=8, burn_in=2, burn=2,
                     episode_end=True)
        # 4 transitions -> 5 real observation rows out of 11.
        self.assertEqual(int(window["valid"].sum()), 5)
        self.assertFalse(window["valid"][5:].any())


class TestReplayParity(unittest.TestCase):
    """Online windows must be cut like demonstration windows, or the mixture
    is two distributions wearing one batch."""

    def make_episode(self, steps=30):
        from sim_vla.data.replay import OnlineEpisode

        actions, rewards, observations = episode(steps)
        ep = OnlineEpisode()
        ep.add_observation({"proprio": observations[0]})
        for index in range(steps):
            ep.add_transition(actions[index], float(rewards[index]),
                              False, index == steps - 1, False)
            ep.add_observation({"proprio": observations[index + 1]})
        return ep

    def test_scored_span_never_exceeds_the_demonstration_length(self):
        """The bug: a window starting at step 0 scored ``length + burn_in``.

        A demonstration window scores at most ``length + 1`` rows -- ``length``
        transitions plus the final observation. The old replay could score
        ``length + burn_in + 1``, because it sampled ``length + burn_in``
        transitions and then called only ``min(burn_in, start)`` of them
        burn-in, which is zero for every window that began at step 0.
        """
        from sim_vla.data.replay import OnlineReplay

        length, burn_in = 8, 3
        replay = OnlineReplay(seed=0)
        for _ in range(6):
            replay.add(self.make_episode(30))
        worst = 0
        for _ in range(40):
            batch = replay.sample(4, length=length, burn_in=burn_in)
            worst = max(worst, int(batch["loss_mask"].sum(axis=1).max()))
        self.assertLessEqual(
            worst, length + 1,
            f"a replay window scored {worst} rows where a demonstration "
            f"window scores at most {length + 1}")
        # And the old behaviour really is excluded by that bound.
        self.assertLess(length + 1, length + burn_in + 1)

    def test_a_window_at_the_reset_has_nothing_to_burn_in(self):
        """``plan_windows`` strides by ``length``, so a demonstration window
        either starts at the reset with no burn-in or starts clear of it. A
        uniform scored start also produced a third kind, and a mixed batch
        would then hold two window shapes the model cannot distinguish."""
        from sim_vla.data.replay import OnlineReplay

        replay = OnlineReplay(seed=5)
        for _ in range(4):
            replay.add(self.make_episode(30))
        for _ in range(60):
            batch = replay.sample(4, length=8, burn_in=3)
            for row in range(batch["is_first"].shape[0]):
                if batch["is_first"][row, 0]:
                    self.assertTrue(
                        batch["loss_mask"][row, 0],
                        "a window at the reset burned in its first row")

    def test_replay_rows_match_the_demonstration_row_count(self):
        from sim_vla.data.replay import OnlineReplay

        replay = OnlineReplay(seed=1)
        for _ in range(6):
            replay.add(self.make_episode(30))
        batch = replay.sample(3, length=8, burn_in=3)
        for key, value in batch.items():
            self.assertEqual(value.shape[1], rows(8, 3), key)

    def test_replay_action_target_alignment(self):
        from sim_vla.data.replay import OnlineReplay

        replay = OnlineReplay(seed=2)
        replay.add(self.make_episode(30))
        batch = replay.sample(2, length=8, burn_in=0, lookahead=2)
        for row in range(batch["action_target"].shape[0]):
            target = batch["action_target"][row]
            available = batch["action_valid"][row]
            # actions[i] = [i, -i, i/10]: the second component is the negated
            # first wherever a real action was loaded.
            np.testing.assert_allclose(target[available][:, 1],
                                       -target[available][:, 0], atol=1e-6)

    def test_mixed_batch_refuses_sources_that_disagree(self):
        from sim_vla.data.replay import OnlineReplay, mixed_batch

        class Sampler:
            def batch(self, size):
                return {"action": np.zeros((size, 5, ACTION), np.float32)}

        replay = OnlineReplay(seed=3)
        for _ in range(6):
            replay.add(self.make_episode(20))
        with self.assertRaises(KeyError) as caught:
            mixed_batch(Sampler(), replay, 8, 4, 0, 0.5)
        # Silently intersecting is what dropped actions and rewards before.
        self.assertIn("disagree", str(caught.exception))

    def test_mixed_batch_refuses_mismatched_lookaheads(self):
        """Same keys, different target-axis length: a silent concatenate error
        at best, and two distributions in one batch at worst."""
        from sim_vla.data.replay import OnlineReplay, mixed_batch

        demo_replay = OnlineReplay(seed=6)
        online = OnlineReplay(seed=7)
        for _ in range(6):
            demo_replay.add(self.make_episode(20))
            online.add(self.make_episode(20))

        class Sampler:
            lookahead = 4

            def batch(self, size):
                return demo_replay.sample(size, 4, 1, self.lookahead)

        with self.assertRaises(ValueError) as caught:
            mixed_batch(Sampler(), online, 8, 4, 1, 0.5, lookahead=0)
        self.assertIn("lookahead", str(caught.exception))

    def test_mixed_batch_stacks_when_the_lookaheads_agree(self):
        from sim_vla.data.replay import OnlineReplay, mixed_batch

        demo_replay = OnlineReplay(seed=8)
        online = OnlineReplay(seed=9)
        for _ in range(6):
            demo_replay.add(self.make_episode(20))
            online.add(self.make_episode(20))

        class Sampler:
            lookahead = 4

            def batch(self, size):
                return demo_replay.sample(size, 4, 1, self.lookahead)

        out = mixed_batch(Sampler(), online, 8, 4, 1, 0.5, lookahead=4)
        self.assertEqual(out["action"].shape[0], 8)
        self.assertEqual(out["action_target"].shape[1], rows(4, 1) + 4)
        self.assertEqual(out["loss_mask"].shape[1], rows(4, 1))

    def test_mixed_batch_falls_back_to_demonstrations_while_replay_is_small(self):
        from sim_vla.data.replay import OnlineReplay, mixed_batch

        calls = []

        class Sampler:
            def batch(self, size):
                calls.append(size)
                return {"action": np.zeros((size, 5, ACTION), np.float32)}

        replay = OnlineReplay(seed=4)
        replay.add(self.make_episode(20))
        out = mixed_batch(Sampler(), replay, 8, 4, 0, 0.5)
        self.assertEqual(calls, [8])
        self.assertEqual(out["action"].shape[0], 8)


class TestSamplerContract(unittest.TestCase):
    """What a real window actually exposes, against a real dataset file.

    Stage 1B used to read ``sampler.batch(1)["actions"]``. No window has ever
    had that key: ``layout.assemble`` splits the stored actions into ``action``
    and ``action_target``, so the default PickCube run raised ``KeyError`` at
    the first line of Stage 1B.
    """

    def setUp(self):
        from sim_vla.data.dataset import DemoDataset

        from tests.test_sim_vla_pipeline import make_dataset

        self.tmp = tempfile.TemporaryDirectory()
        # Registered first, so it runs last: the dataset handle added below
        # has to be closed before Windows will let the file be deleted.
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "demos.h5"
        make_dataset(self.path, episodes=((30, 24), (18, 14)))
        self.data = DemoDataset(self.path, graph_enabled=False)
        self.addCleanup(self.data.close)

    def sampler(self, **kwargs):
        from sim_vla.data.sequences import SequenceSampler

        return SequenceSampler(self.data, length=8, burn_in=2, seed=0,
                               **kwargs)

    def test_windows_expose_the_canonical_action_keys(self):
        batch = self.sampler().batch(2)
        for key in ("action", "action_target", "action_valid", "loss_mask",
                    "valid", "reward", "reward_valid"):
            self.assertIn(key, batch)
        self.assertNotIn("actions", batch,
                         "the storage name does not survive windowing; "
                         "inferring the action width from it cannot work")

    def test_action_width_comes_from_the_target_key(self):
        from sim_vla.data.sequences import SequenceSampler  # noqa: F401

        batch = self.sampler().batch(1)
        self.assertEqual(int(np.asarray(batch["action_target"]).shape[-1]),
                         int(np.asarray(batch["action"]).shape[-1]))

    def test_lookahead_extends_availability_without_extending_valid(self):
        # Two samplers over the same dataset with the same seed: the only
        # difference between the batches is the lookahead.
        plain = self.sampler().batch(4)
        ahead = self.sampler(lookahead=4).batch(4)
        self.assertGreater(int(ahead["action_valid"].sum()),
                           int(plain["action_valid"].sum()),
                           "lookahead supplied no additional targets")
        self.assertEqual(int(plain["valid"].sum()), int(ahead["valid"].sum()),
                         "lookahead changed which observations are real")
        np.testing.assert_array_equal(plain["loss_mask"], ahead["loss_mask"])

    def test_no_window_crosses_an_episode_boundary(self):
        from sim_vla.data.sequences import plan_windows

        sampler = self.sampler(lookahead=4)
        for ref in sampler.data.episodes:
            for window in plan_windows(ref, 8, 2):
                self.assertGreaterEqual(window.start, 0)
                self.assertLessEqual(window.stop, ref.steps)


class TestActionCoordinates(unittest.TestCase):
    """Three coordinate systems, and the RSSM's unit-ball projection."""

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable: coordinate conversions are "
                          "tensor operations")

    def test_the_rssm_clip_collapses_distinct_standardized_actions(self):
        """The reason a dynamics coordinate system exists at all."""
        import torch

        # rssm.py:65, reproduced. Two valid standardized actions, one forward
        # value: the dynamics cannot tell 2 sigma from 4 sigma.
        def rssm_clip(action):
            return action / torch.clip(torch.abs(action), min=1.0).detach()

        two = torch.tensor([[2.0]])
        four = torch.tensor([[4.0]])
        self.assertTrue(torch.allclose(rssm_clip(two), rssm_clip(four)))

    def test_dynamics_coordinates_survive_that_clip(self):
        import torch

        from sim_vla.models.action_space import to_dynamics

        def rssm_clip(action):
            return action / torch.clip(torch.abs(action), min=1.0).detach()

        two = to_dynamics(torch.tensor([[2.0]]))
        four = to_dynamics(torch.tensor([[4.0]]))
        self.assertTrue(float(two.abs().max()) < 1.0)
        self.assertTrue(float(four.abs().max()) < 1.0)
        # Unchanged by the clip, and still distinguishable.
        self.assertTrue(torch.allclose(rssm_clip(two), two))
        self.assertTrue(torch.allclose(rssm_clip(four), four))
        self.assertGreater(float((four - two).abs().min()), 1e-3)

    def test_dynamics_map_is_invertible(self):
        import torch

        from sim_vla.models.action_space import from_dynamics, to_dynamics

        values = torch.linspace(-8.0, 8.0, 65).reshape(-1, 1)
        recovered = from_dynamics(to_dynamics(values))
        self.assertTrue(torch.allclose(values, recovered, atol=1e-3))

    def test_normalization_round_trip_with_nonzero_mean_and_nonunit_scale(self):
        import torch

        from sim_vla.models.action_space import ActionCoordinates

        coords = ActionCoordinates(real_normalizer(mean=[5.0, -2.0],
                                                   std=[0.5, 4.0]))
        raw = torch.tensor([[5.0, -2.0], [6.0, 2.0], [4.0, -10.0]])
        normalized = coords.normalize(raw)
        self.assertTrue(torch.allclose(normalized[0],
                                       torch.zeros(2), atol=1e-6))
        self.assertTrue(torch.allclose(coords.denormalize(normalized), raw,
                                       atol=1e-5))
        # 1 sigma above the mean on a std of 0.5 is raw 5.5.
        self.assertAlmostEqual(float(normalized[1][0]), 2.0, places=5)

    def test_clipping_is_straight_through(self):
        import torch

        from sim_vla.models.action_space import (ActionBounds,
                                                 ActionCoordinates)

        coords = ActionCoordinates(
            real_normalizer(mean=[0.0], std=[0.5]),
            ActionBounds(low=np.array([-1.0], np.float32),
                         high=np.array([1.0], np.float32)))
        # Raw bounds +-1 with std 0.5 are +-2 in normalized coordinates.
        value = torch.tensor([[5.0]], requires_grad=True)
        out = coords.executed(value)
        self.assertAlmostEqual(float(out), 2.0, places=5)
        out.sum().backward()
        # A saturated dimension still receives a gradient: it is exactly the
        # one the actor has to be pushed back from.
        self.assertAlmostEqual(float(value.grad), 1.0, places=6)


def real_normalizer(mean, std, low=None, high=None, mode="mean_std"):
    """A genuine :class:`Normalizer`, so the tensor path can be compared to it."""
    from sim_vla.data.normalization import FieldStats, Normalizer

    low = [m - 1.0 for m in mean] if low is None else low
    high = [m + 1.0 for m in mean] if high is None else high
    stats = FieldStats(mean=list(mean), std=list(std), low=list(low),
                       high=list(high), minimum=list(low), maximum=list(high),
                       count=100)
    return Normalizer(fields={"actions": stats, "proprio": stats},
                      identity={"dataset": "toy"}, mode=mode)


class TestConstantDimensions(unittest.TestCase):
    """A dimension the robot never moves has ``std == 0``.

    ``fit_field`` reports that honestly. ``Normalizer.normalize`` divides by
    ``max(std, EPS)``; the tensor path divided by the raw statistic and
    produced ``inf`` and ``nan`` for previous actions, action targets and
    proprioception, beside a numpy path returning finite numbers.
    """

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable: the tensor path is under test")

    def test_actions_stay_finite_and_match_the_numpy_path(self):
        import torch

        from sim_vla.models.action_space import ActionCoordinates

        # Second dimension is constant at 5.0.
        norm = real_normalizer(mean=[0.0, 5.0], std=[2.0, 0.0])
        coords = ActionCoordinates(norm)
        raw = torch.tensor([[1.0, 5.0], [-3.0, 5.0], [0.0, 5.0]])

        out = coords.normalize(raw)
        self.assertTrue(torch.isfinite(out).all(),
                        f"nonfinite normalized actions: {out}")
        expected = norm.normalize("actions", raw.numpy())
        self.assertTrue(torch.allclose(out, torch.as_tensor(expected),
                                       atol=1e-6),
                        "the tensor path disagrees with Normalizer.normalize")

    def test_denormalize_stays_finite_and_round_trips(self):
        import torch

        from sim_vla.models.action_space import ActionCoordinates

        norm = real_normalizer(mean=[0.0, 5.0], std=[2.0, 0.0])
        coords = ActionCoordinates(norm)
        raw = torch.tensor([[1.0, 5.0], [-3.0, 5.0]])
        back = coords.denormalize(coords.normalize(raw))
        self.assertTrue(torch.isfinite(back).all())
        # The constant dimension cannot be recovered by scale, only by offset,
        # and the offset is what carries it.
        self.assertTrue(torch.allclose(back[:, 1], raw[:, 1], atol=1e-4))
        self.assertTrue(torch.allclose(back[:, 0], raw[:, 0], atol=1e-4))

    def test_normalized_bounds_stay_finite(self):
        import torch

        from sim_vla.models.action_space import (ActionBounds,
                                                 ActionCoordinates)

        norm = real_normalizer(mean=[0.0, 5.0], std=[2.0, 0.0])
        coords = ActionCoordinates(
            norm, ActionBounds(low=np.array([-1.0, -1.0], np.float32),
                               high=np.array([1.0, 1.0], np.float32)))
        low, high = coords.normalized_bounds(torch.zeros(1, 2))
        self.assertTrue(torch.isfinite(low).all())
        self.assertTrue(torch.isfinite(high).all())
        self.assertTrue(bool((low <= high).all()))

    def test_proprio_stays_finite_through_the_batch_path(self):
        import torch

        from sim_vla.data.batch import to_model_batch

        norm = real_normalizer(mean=[0.0, 5.0], std=[2.0, 0.0])
        batch = {"proprio": np.array([[[1.0, 5.0], [-3.0, 5.0]]], np.float32),
                 "actions": np.array([[[1.0, 5.0], [-3.0, 5.0]]], np.float32),
                 "action_target": np.array([[[1.0, 5.0], [0.0, 5.0]]],
                                           np.float32)}
        out = to_model_batch(batch, normalizer=norm)
        for key in ("proprio", "action", "action_target"):
            self.assertTrue(torch.isfinite(out[key]).all(),
                            f"nonfinite {key}: {out[key]}")

    def test_range_mode_matches_the_numpy_path_too(self):
        """A mode the tensor path ignored would put training and inference in
        different units."""
        import torch

        from sim_vla.models.action_space import ActionCoordinates

        norm = real_normalizer(mean=[0.0, 5.0], std=[2.0, 0.0],
                               low=[-2.0, 5.0], high=[2.0, 5.0], mode="range")
        coords = ActionCoordinates(norm)
        raw = torch.tensor([[1.0, 5.0], [-2.0, 5.0], [2.0, 5.0]])
        out = coords.normalize(raw)
        self.assertTrue(torch.isfinite(out).all())
        expected = norm.normalize("actions", raw.numpy())
        self.assertTrue(torch.allclose(out, torch.as_tensor(expected),
                                       atol=1e-6))
        # -2 and +2 are the fitted edges, so they land on -1 and +1.
        self.assertAlmostEqual(float(out[1, 0]), -1.0, places=5)
        self.assertAlmostEqual(float(out[2, 0]), 1.0, places=5)

    def test_an_unimplemented_mode_is_refused(self):
        from sim_vla.models.action_space import FieldScaler
        from sim_vla.data.normalization import FieldStats

        stats = FieldStats(mean=[0.0], std=[1.0], low=[-1.0], high=[1.0],
                           minimum=[-1.0], maximum=[1.0], count=1)
        with self.assertRaises(ValueError):
            FieldScaler(stats, mode="quantile")


class TestEnvValidation(unittest.TestCase):
    """The recording's own keys, compared against a live environment.

    The first version looked for ``robot_uid`` and top-level frequencies; the
    collector writes ``robot_uids`` at the top level and the frequencies inside
    ``controller``. Every compare saw ``None``, returned early, and the env
    reported ``validated=True`` for a deliberately mismatched robot.
    """

    def metadata(self, **overrides):
        base = {
            "env_id": "PickCube-v1",
            "robot_uids": "panda",
            "reward_mode": "normalized_dense",
            "camera_keys": {"base_camera": "image_base"},
            "image_size": [16, 16],
            "proprio_fields": [],
            "controller": {"control_mode": "pd_joint_pos", "action_dim": 8,
                           "control_freq": 20, "sim_freq": 100,
                           "action_low": [-1.0] * 8, "action_high": [1.0] * 8},
        }
        base.update(overrides)
        return base

    def live(self, **overrides):
        from types import SimpleNamespace

        space = SimpleNamespace(shape=(8,), low=np.full(8, -1.0),
                                high=np.full(8, 1.0))
        unwrapped = SimpleNamespace(
            control_mode="pd_joint_pos", control_freq=20, sim_freq=100,
            robot_uids="panda", reward_mode="normalized_dense",
            single_action_space=space)
        for key, value in overrides.items():
            setattr(unwrapped, key, value)
        return SimpleNamespace(unwrapped=unwrapped, action_space=space)

    def env(self, metadata=None, live=None):
        from sim_vla.envs.maniskill import SimVlaEnv

        env = SimVlaEnv(metadata or self.metadata(), graph_enabled=False)
        env._env = live or self.live()
        return env

    def test_a_matching_environment_validates_and_says_what_it_compared(self):
        report = self.env().validate()
        self.assertTrue(report["validated"])
        for key in ("robot_uids", "controller.control_freq",
                    "controller.sim_freq", "controller.control_mode",
                    "controller.action_dim"):
            self.assertIn(key, report["checked"],
                          f"{key} was never compared")

    def test_a_different_robot_is_caught(self):
        env = self.env(live=self.live(robot_uids="xarm6_robotiq"))
        with self.assertRaises(SystemExit) as caught:
            env.validate()
        self.assertIn("robot_uids", str(caught.exception))

    def test_different_frequencies_are_caught(self):
        for key, value in (("control_freq", 10), ("sim_freq", 250)):
            with self.subTest(key=key):
                env = self.env(live=self.live(**{key: value}))
                with self.assertRaises(SystemExit) as caught:
                    env.validate()
                self.assertIn(key, str(caught.exception))

    def test_a_different_action_width_is_caught(self):
        from types import SimpleNamespace

        space = SimpleNamespace(shape=(7,), low=np.full(7, -1.0),
                                high=np.full(7, 1.0))
        env = self.env(live=self.live(single_action_space=space))
        with self.assertRaises(SystemExit) as caught:
            env.validate()
        self.assertIn("action_dim", str(caught.exception))

    def test_different_controller_bounds_are_caught(self):
        from types import SimpleNamespace

        space = SimpleNamespace(shape=(8,), low=np.full(8, -0.5),
                                high=np.full(8, 0.5))
        env = self.env(live=self.live(single_action_space=space))
        with self.assertRaises(SystemExit) as caught:
            env.validate()
        self.assertIn("action_low", str(caught.exception))

    def test_a_different_reward_mode_is_caught(self):
        env = self.env(live=self.live(reward_mode="sparse"))
        with self.assertRaises(SystemExit) as caught:
            env.validate()
        self.assertIn("reward_mode", str(caught.exception))


class TestBatchPreprocessingContract(unittest.TestCase):
    """Idempotence that actually holds, and tensors that stay put."""

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

    def raw(self):
        return {
            "image_base": np.full((1, 2, 4, 4, 3), 128, np.uint8),
            "actions": np.full((1, 2, 2), 4.0, np.float32),
            "action_target": np.full((1, 2, 2), 4.0, np.float32),
        }

    def test_three_passes_are_the_same_as_one(self):
        """The bug: the second pass consumed the marker, so the third
        normalized again."""
        import torch

        from sim_vla.data.batch import to_model_batch

        coords_source = real_normalizer(mean=[1.0, 1.0], std=[2.0, 2.0])
        once = to_model_batch(self.raw(), normalizer=coords_source)
        twice = to_model_batch(once, normalizer=coords_source)
        thrice = to_model_batch(twice, normalizer=coords_source)
        for key in ("image_base", "action", "action_target"):
            self.assertTrue(torch.allclose(once[key], twice[key]), key)
            self.assertTrue(torch.allclose(once[key], thrice[key]),
                            f"{key} was normalized a second time")

    def test_the_marker_survives_the_call(self):
        from sim_vla.data.batch import MARKER, is_preprocessed, to_model_batch

        once = to_model_batch(self.raw())
        self.assertTrue(is_preprocessed(once))
        self.assertIn(MARKER, once)

    def test_tensor_input_is_accepted(self):
        """np.asarray on a live tensor is a CPU round trip at best."""
        import torch

        from sim_vla.data.batch import to_model_batch

        batch = {key: torch.as_tensor(value)
                 for key, value in self.raw().items()}
        out = to_model_batch(batch)
        self.assertTrue(out["image_base"].is_floating_point())
        self.assertLessEqual(float(out["image_base"].max()), 1.0)

    def test_action_and_target_leave_in_different_coordinates(self):
        import torch

        from sim_vla.data.batch import to_model_batch
        from sim_vla.models.action_space import ActionCoordinates

        coords = ActionCoordinates(real_normalizer(mean=[0.0, 0.0],
                                                   std=[1.0, 1.0]))
        out = to_model_batch(self.raw(), coords=coords)
        # action is the RSSM's input and must be inside the unit ball;
        # action_target is the actor's supervision and must not be squashed.
        self.assertLess(float(out["action"].abs().max()), 1.0)
        self.assertAlmostEqual(float(out["action_target"].abs().max()), 4.0,
                               places=5)


if __name__ == "__main__":
    unittest.main()
