"""The pieces both backends stand on: imports, counting, units, masks.

Four of the earlier sim_vla review findings are properties of these utilities
rather than of either backend, so they are checked here before anything reuses
them:

* a standard deviation of exactly zero must not become a division by zero
* the future-action lookahead has to have room for a whole chunk
* conditioning eligibility and target availability are two different masks
* an action that crosses into a backend's units must come back unchanged
"""

from __future__ import annotations

import unittest

import numpy as np

from . import common as C


class VendoredImports(unittest.TestCase):
    """Two upstream trees, both claiming ``envs``, in one process."""

    def test_each_backend_owns_its_own_top_level_names(self):
        from ..vendor import SOLD, TDMPC2

        self.assertIn("common", TDMPC2.owned)
        self.assertIn("envs", TDMPC2.owned)
        self.assertIn("modeling", SOLD.owned)
        self.assertIn("envs", SOLD.owned)
        # The collision that makes this module necessary.
        self.assertIn("envs", TDMPC2.owned & SOLD.owned)

    def test_names_are_handed_back_after_the_block(self):
        import sys

        from ..vendor import TDMPC2

        before = dict(sys.modules)
        with TDMPC2.active():
            import common  # noqa: F401

            self.assertIn("common", sys.modules)
        self.assertNotIn("common", set(sys.modules) - set(before))
        self.assertEqual(sorted(set(sys.modules) - set(before)), [])

    def test_the_repository_keeps_its_own_modules(self):
        """A backend's ``envs`` must not survive to shadow the repo's."""
        import sys

        from ..vendor import TDMPC2

        sentinel = object()
        previous = sys.modules.get("envs", sentinel)
        try:
            sys.modules["envs"] = sentinel                  # type: ignore[assignment]
            with TDMPC2.active():
                self.assertIsNot(sys.modules.get("envs"), sentinel)
            self.assertIs(sys.modules["envs"], sentinel)
        finally:
            if previous is sentinel:
                sys.modules.pop("envs", None)
            else:
                sys.modules["envs"] = previous

    def test_re_entering_reuses_the_same_module_objects(self):
        """Classes defined inside the block must stay usable and identical."""
        from ..vendor import TDMPC2

        first = TDMPC2.get("common.world_model", "WorldModel")
        second = TDMPC2.get("common.world_model", "WorldModel")
        self.assertIs(first, second)

    def test_both_backends_load_in_one_process(self):
        C.require_torch()
        from ..vendor import SOLD, TDMPC2

        world_model = TDMPC2.get("common.world_model", "WorldModel")
        predictor = SOLD.get("modeling.sold.prediction", "GaussianPredictor")
        self.assertTrue(callable(world_model))
        self.assertTrue(callable(predictor))


class ParameterCounting(unittest.TestCase):
    def test_shared_parameters_are_counted_once(self):
        torch = C.require_torch()
        import torch.nn as nn

        from ..params import report

        inner = nn.Linear(4, 4)
        outer = nn.Sequential(inner, nn.Linear(4, 2))
        counted = report([("inner", inner), ("outer", outer)])
        by_name = {c["name"]: c for c in counted["components"]}
        self.assertEqual(by_name["inner"]["unique"], 20)
        self.assertEqual(by_name["outer"]["total"], 20 + 10)
        self.assertEqual(by_name["outer"]["unique"], 10)
        self.assertEqual(by_name["outer"]["shared_with"], ["inner"])
        self.assertEqual(counted["total"], 30)

    def test_a_deep_copy_counts_twice(self):
        """A target network is real memory; tied weights are not."""
        import copy

        torch = C.require_torch()
        import torch.nn as nn

        from ..params import report

        critic = nn.Linear(4, 4)
        target = copy.deepcopy(critic).requires_grad_(False)
        counted = report([("critic", critic), ("target", target)])
        self.assertEqual(counted["total"], 40)
        self.assertEqual(counted["trainable"], 20)

    def test_the_budget_can_exclude_a_component(self):
        torch = C.require_torch()
        import torch.nn as nn

        from ..params import report, split

        counted = split(report([("wm", nn.Linear(4, 4)),
                                ("smolvla", nn.Linear(100, 100))]),
                        exclude=("smolvla",))
        self.assertEqual(counted["budget_total"], 20)
        self.assertEqual(counted["excluded_total"], 10100)
        self.assertEqual(counted["total"], 10120)


class ActionUnits(unittest.TestCase):
    def test_a_constant_dimension_does_not_divide_by_zero(self):
        """The std-flooring finding, on the path both backends use.

        A joint the task never moves has a fitted std of exactly zero. Without
        the floor this produces inf on that dimension and nan on the loss one
        step later, while the numpy normalizer beside it returns finite values.
        """
        torch = C.require_torch()

        from ...data.normalization import FieldStats, Normalizer
        from ..action_space import ActionConverter

        width = 4
        stats = FieldStats(mean=[0.0] * width, std=[1.0, 0.0, 2.0, 0.0],
                           low=[-1.0] * width, high=[1.0, 1.0, 1.0, 1.0],
                           minimum=[-1.0] * width, maximum=[1.0] * width,
                           count=10)
        normalizer = Normalizer(fields={"actions": stats}, identity={},
                                mode="mean_std")
        converter = ActionConverter(action_dim=width, mode="mean_std",
                                    normalizer=normalizer)
        native = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        actor = converter.to_actor(native)
        self.assertTrue(torch.isfinite(actor).all(),
                        f"a zero-std dimension produced {actor}")
        back = converter.to_native(actor)
        self.assertTrue(torch.isfinite(back).all())

    def test_round_trip_in_every_mode(self):
        torch = C.require_torch()

        from ...data.normalization import FieldStats, Normalizer
        from ..action_space import ActionConverter

        width = 3
        stats = FieldStats(mean=[0.1, -0.2, 0.0], std=[0.5, 2.0, 1.0],
                           low=[-0.8, -1.0, -0.5], high=[0.9, 1.0, 0.5],
                           minimum=[-1.0] * width, maximum=[1.0] * width,
                           count=10)
        native = torch.tensor([[0.2, -0.4, 0.05], [-0.7, 0.9, 0.3]])
        for mode in ("identity", "mean_std", "range"):
            normalizer = None
            if mode != "identity":
                normalizer = Normalizer(fields={"actions": stats}, identity={},
                                        mode=mode)
            converter = ActionConverter(action_dim=width, mode=mode,
                                        normalizer=normalizer)
            actor = converter.to_actor(native)
            back = converter.to_native(actor)
            self.assertTrue(torch.allclose(native, back, atol=1e-5),
                            f"{mode}: {native} -> {actor} -> {back}")

    def test_clipping_is_straight_through(self):
        """Clipped forward, identity backward.

        A hard clamp zeroes the gradient of exactly the dimension the actor
        most needs to be pushed back from.
        """
        torch = C.require_torch()

        from ..action_space import ActionConverter

        converter = ActionConverter(action_dim=2)
        action = torch.tensor([[3.0, -0.25]], requires_grad=True)
        executed = converter.executed(action)
        self.assertTrue(torch.allclose(executed.detach(),
                                       torch.tensor([[1.0, -0.25]])))
        executed.sum().backward()
        self.assertTrue(torch.allclose(action.grad, torch.ones_like(action)))

    def test_no_dreamer_squashing(self):
        """``tanh(x / 3)`` belongs to the RSSM and to nothing here."""
        torch = C.require_torch()

        from ..action_space import ActionConverter

        converter = ActionConverter(action_dim=2)
        inside = torch.tensor([[0.5, -0.5]])
        self.assertTrue(torch.allclose(converter.to_env(inside), inside))


class ChunkMasks(unittest.TestCase):
    def build(self, *, chunk=4, length=8, burn_in=2, lookahead=None):
        from ...data.sequences import SequenceSampler
        from ..tdmpc2.data import DemoWindows

        source = C.fake_source()
        windows = DemoWindows(source, horizon=length, burn_in=burn_in, seed=0,
                              stride=1,
                              lookahead=chunk if lookahead is None else lookahead)
        windows.windows = list(windows.sampler.windows)
        return source, windows

    def test_eligibility_and_availability_are_different_masks(self):
        torch = C.require_torch()

        from .. import chunking

        _source, windows = self.build()
        raw = windows.raw_batch(4)
        batch = {k: torch.as_tensor(np.asarray(v)) for k, v in raw.items()}
        scored = batch["loss_mask"].bool()
        available = batch["action_valid"].bool()
        steps = scored.shape[1]
        eligible = chunking.eligibility(scored, available)
        # Burn-in rows are available (a real action was loaded there) and not
        # eligible (the batch never established the state they describe).
        self.assertTrue(available[:, :2].all())
        self.assertFalse(scored[:, :2].any())
        self.assertFalse(eligible[:, :2].any())
        # The masks live on two axes and the longer one is the target axis.
        self.assertGreater(available.shape[1], steps)

    def test_the_lookahead_supplies_a_whole_chunk_at_the_last_row(self):
        """The lookahead-capacity finding.

        A full interior window has no padding, so squeezing the lookahead into
        the observation axis leaves exactly one free slot however long the
        chunk is -- and the last rows get 4, 3, 2, 1 actions instead of 4, 4,
        4, 4.
        """
        torch = C.require_torch()

        from .. import chunking

        chunk = 4
        _source, windows = self.build(chunk=chunk, length=8, burn_in=2)
        # An interior window: one that neither starts at a reset nor ends at
        # the end of its episode.
        interior = [w for w in windows.windows
                    if w.start > 0 and w.stop < 24 - chunk]
        self.assertTrue(interior, "the fixture has no interior windows")
        loaded = windows.sampler.load(interior[0])
        batch = {k: torch.as_tensor(np.asarray(v))[None] for k, v in loaded.items()}
        selection = chunking.select(batch, chunk)
        self.assertFalse(selection.empty)
        self.assertTrue(selection.mask.all(),
                        f"an interior window left {(~selection.mask).sum()} "
                        "chunk offsets unsupervised")

    def test_targets_are_a_t_not_a_t_minus_one(self):
        """The chunk at row t starts with the action taken *at* row t."""
        torch = C.require_torch()

        from .. import chunking

        source, windows = self.build(chunk=3, length=6, burn_in=0)
        window = [w for w in windows.windows if w.start == 0][0]
        loaded = windows.sampler.load(window)
        batch = {k: torch.as_tensor(np.asarray(v))[None] for k, v in loaded.items()}
        selection = chunking.select(batch, 3)
        # Row 0 is the episode's first observation: a_0 is the counter value 0
        # for episode 0, while `action` there is the zero lead-in.
        first = selection.targets[0]
        self.assertTrue(torch.allclose(first[0], torch.zeros_like(first[0])))
        self.assertTrue(torch.allclose(first[1], torch.ones_like(first[1])))
        self.assertTrue(torch.allclose(first[2], 2 * torch.ones_like(first[2])))
        self.assertTrue(torch.allclose(batch["action"][0, 0],
                                       torch.zeros_like(first[0])))

    def test_chunks_stop_at_the_episode_boundary(self):
        torch = C.require_torch()

        from .. import chunking

        chunk = 5
        source, windows = self.build(chunk=chunk, length=6, burn_in=0)
        steps = source.dataset.episodes[0].steps
        tail = [w for w in windows.windows if w.stop >= steps]
        self.assertTrue(tail)
        loaded = windows.sampler.load(tail[0])
        batch = {k: torch.as_tensor(np.asarray(v))[None] for k, v in loaded.items()}
        selection = chunking.select(batch, chunk)
        self.assertFalse(selection.mask.all(),
                         "a window that reaches the end of its episode cannot "
                         "have every chunk offset available")
        # Whatever is masked has to be a suffix: availability is cumulative.
        for row in selection.mask:
            flips = (row[:-1] < row[1:]).sum()
            self.assertEqual(int(flips), 0,
                             f"{row.tolist()} turns availability back on")


class ImageContract(unittest.TestCase):
    def test_bytes_stay_bytes_and_layout_changes(self):
        torch = C.require_torch()

        from ..observations import contract_for

        demos = C.FakeDemos(image=32)
        contract = contract_for(demos.metadata, backend="tdmpc2", size=(64, 64))
        window = demos.read(demos.episodes[0], 0, 4)
        out = contract.apply({k: v for k, v in window.items()})
        self.assertEqual(out.dtype, torch.uint8)
        self.assertEqual(tuple(out.shape), (5, 3, 64, 64))

    def test_a_float_image_is_refused(self):
        """Because the backends do their own ``/255``."""
        torch = C.require_torch()

        from ..observations import ImageContract

        contract = ImageContract([C.CAMERA], (8, 8), backend="tdmpc2")
        with self.assertRaises(TypeError):
            contract.apply({C.CAMERA: torch.zeros(2, 8, 8, 3)})

    def test_the_source_array_is_not_written_into(self):
        torch = C.require_torch()

        from ..observations import ImageContract

        contract = ImageContract([C.CAMERA], (8, 8), backend="tdmpc2")
        original = torch.full((2, 8, 8, 3), 200, dtype=torch.uint8)
        keep = original.clone()
        out = contract.apply({C.CAMERA: original})
        out.zero_()
        self.assertTrue(torch.equal(original, keep))


class ActorCapabilities(unittest.TestCase):
    def test_log_prob_and_entropy_raise(self):
        """No fabricated density, and no flow loss standing in for one."""
        C.require_torch()

        from ..latent_actor import ActorCapabilityError

        actor = C.stub_latent_actor(8, 3)
        with self.assertRaises(ActorCapabilityError):
            actor.log_prob()
        with self.assertRaises(ActorCapabilityError):
            actor.entropy()
        with self.assertRaises(ActorCapabilityError):
            actor.rsample()

    def test_sampling_keeps_the_gradient_when_asked(self):
        torch = C.require_torch()

        from ..latent_actor import assert_gradient_reaches

        actor = C.stub_latent_actor(8, 3)
        feature = torch.zeros(2, 8)
        action = actor.sample_chunk(feature, differentiable=True)
        assert_gradient_reaches(action, actor.trainable_parameters())
        detached = actor.sample_chunk(feature, differentiable=False)
        self.assertFalse(detached.requires_grad)


if __name__ == "__main__":
    unittest.main()
