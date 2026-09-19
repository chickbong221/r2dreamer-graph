"""Fixtures for the integration tests.

Three things are faked and everything else is real.

**The demonstrations.** ``FakeDemos`` stands in for
:class:`sim_vla.data.dataset.DemoDataset` so the window, alignment and masking
tests run without HDF5 and without a collected dataset. Its arrays are
*countable*: action ``t`` of episode ``e`` is ``e * 1000 + t``, so an
off-by-one in the action/reward timeline shows up as a wrong integer rather
than as a slightly worse loss. It is not a stand-in for the real loader in the
tests that open one.

**The action expert.** ``StubExpert`` is a small linear velocity field with the
same call signature SmolVLA's has. It exists so the flow loss, the sampler,
the chunk masking and -- most importantly -- the *gradient path* from a native
objective back into the conditioning token can be tested without a 450M
checkpoint or a GPU. It is deliberately not constant in either its input or
its conditioning: a head whose output does not depend on what it was given
passes a gradient test that means nothing.

**Nothing else.** The world models, the planner, the losses and the trainers
under test are the real ones, imported from the vendored trees.

A missing dependency is reported as a skip that names it, never as a pass, and
``run_stage.py`` treats a stage whose required module skipped as incomplete.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def require(module: str):
    try:
        return __import__(module)
    except Exception as exc:                               # noqa: BLE001
        raise unittest.SkipTest(f"{module} unavailable: {exc}")


def require_torch():
    return require("torch")


def require_cuda():
    torch = require_torch()
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no CUDA device")
    return torch


def require_pretrained():
    """The real SmolVLA checkpoint, or a skip naming why not."""
    from ...tests.test_pretrained import load

    return load()


# ------------------------------------------------------------- demonstrations
CAMERA = "image_base"


class FakeRef:
    """What :class:`sim_vla.data.sequences.SequenceSampler` reads off a ref."""

    def __init__(self, episode_id: int, steps: int):
        self.episode_id = int(episode_id)
        self.steps = int(steps)
        self.group = f"traj_{episode_id}"
        self.recorded_steps = int(steps)

    @property
    def observations(self) -> int:
        return self.steps + 1


class _Fields:
    supervision = ("actions", "rewards", "terminated", "truncated", "success")


class FakeDemos:
    """A countable stand-in for the collected demonstrations.

    ``actions[t] = episode * 1000 + t`` on every dimension and
    ``rewards[t] = -(episode * 1000 + t)``, so a window that mixes up ``a_t``
    with ``a_(t-1)``, or reads the reward one row out, produces a number that
    names the mistake.
    """

    def __init__(self, *, episodes: int = 3, steps: int = 24,
                 image: int = 16, proprio: int = 9, action: int = 7,
                 cameras: Sequence[str] = (CAMERA,), seed: int = 0,
                 bounded_actions: bool = False):
        # ``bounded_actions`` maps the same counter through a sine so the
        # actions live in the controller's range. Still a bijection of
        # (episode, t) over a window, so alignment stays checkable, but a loss
        # computed on them is on the scale a real run sees.
        self.bounded_actions = bool(bounded_actions)
        self.path = Path("fake://demos.h5")
        self.cameras = tuple(cameras)
        self.image = int(image)
        self.proprio_dim = int(proprio)
        self.action_dim = int(action)
        self.ignore_terminations = True
        self.fields = _Fields()
        self.episodes: List[FakeRef] = [
            FakeRef(index, int(steps)) for index in range(int(episodes))]
        self.metadata: Dict[str, Any] = {
            "env_id": "PickCube-v1",
            "camera_keys": {name.replace("image_", ""): name
                            for name in self.cameras},
            "image_size": (self.image, self.image),
            "reward_mode": "normalized_dense",
            "proprio_names": [f"p{i}" for i in range(self.proprio_dim)],
            "controller": {"control_mode": "pd_joint_pos",
                           "action_dim": self.action_dim,
                           "action_low": [-1.0] * self.action_dim,
                           "action_high": [1.0] * self.action_dim},
            "versions": {"repo_revision": "fake"},
        }
        self._rng = np.random.default_rng(int(seed))
        self._images = {
            ref.episode_id: {
                name: self._rng.integers(
                    0, 256, (ref.steps + 1, self.image, self.image, 3)
                ).astype(np.uint8)
                for name in self.cameras}
            for ref in self.episodes}

    # -------------------------------------------------------------- interface
    def read(self, ref: FakeRef, start: int, stop: int) -> Dict[str, np.ndarray]:
        if not 0 <= start < stop <= ref.steps:
            raise IndexError(f"[{start}, {stop}) outside episode {ref.episode_id}")
        index = np.arange(start, stop, dtype=np.float32)
        tag = float(ref.episode_id) * 1000.0
        out: Dict[str, np.ndarray] = {}
        for name in self.cameras:
            out[name] = self._images[ref.episode_id][name][start:stop + 1]
        out["proprio"] = np.tile(
            (tag + index)[:, None], (1, self.proprio_dim)).astype(np.float32)
        out["proprio"] = np.concatenate(
            [out["proprio"], out["proprio"][-1:] + 1.0], axis=0)
        counter = np.tile((tag + index)[:, None],
                          (1, self.action_dim)).astype(np.float32)
        if self.bounded_actions:
            dims = np.arange(self.action_dim, dtype=np.float32)[None, :]
            out["actions"] = np.sin(0.37 * counter + 0.61 * dims).astype(np.float32)
        else:
            out["actions"] = counter
        out["rewards"] = -(tag + index).astype(np.float32)
        out["terminated"] = np.zeros(stop - start, dtype=bool)
        out["truncated"] = np.zeros(stop - start, dtype=bool)
        out["success"] = np.zeros(stop - start, dtype=bool)
        return out

    def close(self) -> None:
        pass


def fake_source(*, render_size: int = 64, include_state: bool = False,
                **kwargs):
    """A :class:`sim_vla.integrations.tdmpc2.data.DemoSource` over fake demos."""
    from ..observations import contract_for
    from ..tdmpc2.data import DemoSource

    demos = FakeDemos(**kwargs)
    images = contract_for(demos.metadata, backend="tdmpc2",
                          size=(int(render_size), int(render_size)))
    return DemoSource(dataset=demos, images=images,
                      include_state=bool(include_state),
                      action_dim=demos.action_dim,
                      proprio_dim=demos.proprio_dim,
                      episode_length=demos.episodes[0].steps)


# ----------------------------------------------------------------- the expert
_CLASSES: Dict[str, Any] = {}


def stub_classes() -> Dict[str, Any]:
    """``StubExpert`` and ``StubActor``, defined once torch is known to exist.

    They subclass ``nn.Module``, which cannot be referenced at import time on a
    machine without torch -- and these are the tests that skip there.
    """
    if _CLASSES:
        return _CLASSES
    torch = require_torch()
    import torch.nn as nn

    from ...models.latent_adapter import LatentAdapter

    class StubExpert(nn.Module):
        """A velocity field with SmolVLA's signature and a real dependency.

        Linear in ``(x_t, conditioning token, t)``, and every one of those
        actually moves the output. That is what makes a gradient test mean
        something: a head that ignored its conditioning would report exactly
        zero gradient into the adapter, and a test that only checked "not
        None" would pass while the conditioning path was dead.
        """

        def __init__(self, token_dim: int, action_dim: int, seed: int = 0):
            super().__init__()
            generator = torch.Generator().manual_seed(int(seed))
            self.linear = nn.Linear(token_dim + action_dim + 1, action_dim)
            with torch.no_grad():
                self.linear.weight.normal_(0.0, 0.3, generator=generator)
                self.linear.bias.normal_(0.0, 0.1, generator=generator)

        def forward(self, x_t, t, cond):
            token = cond["state_token"].squeeze(-2)
            batch, chunk, _dim = x_t.shape
            token = token.unsqueeze(1).expand(batch, chunk, token.shape[-1])
            time = t.reshape(batch, 1, 1).expand(batch, chunk, 1)
            return self.linear(torch.cat([x_t, token, time], dim=-1))

    class StubActor(nn.Module):
        """What ``LatentActor`` wraps: adapter, ``condition``, ``velocity_fn``.

        The same surface ``SmolVLAActor`` presents, small enough to run on CPU.
        """

        def __init__(self, feature_dim: int, action_dim: int, *,
                     token_dim: int = 32, chunk_size: int = 4,
                     flow_steps: int = 4, seed: int = 0):
            super().__init__()
            torch.manual_seed(int(seed))
            self.adapter = LatentAdapter(int(feature_dim), int(token_dim),
                                         hidden=64, layers=2)
            self.expert = StubExpert(int(token_dim), int(action_dim), seed=seed)
            self.chunk_size = int(chunk_size)
            self.flow_steps = int(flow_steps)
            self.action_dim = int(action_dim)

        @property
        def device(self):
            return next(self.parameters()).device

        def condition(self, features, instruction=None):
            features = features.to(self.device)
            return {"state_token": self.adapter(features),
                    "batch": int(features.shape[0])}

        def velocity_fn(self):
            return self.expert

        def trainable_report(self):
            return {"stub": True, "chunk_size": self.chunk_size}

    _CLASSES.update(expert=StubExpert, actor=StubActor)
    return _CLASSES


def stub_latent_actor(feature_dim: int, action_dim: int, **kwargs):
    from ..latent_actor import LatentActor

    actor = stub_classes()["actor"](feature_dim, action_dim, **kwargs)
    return LatentActor(actor, instruction="stub")


def stub_slot_actor(*, num_slots: int, slot_dim: int, action_dim: int,
                    context: int = 2, token_dim: int = 32, chunk_size: int = 4,
                    flow_steps: int = 4, seed: int = 0):
    """A stub expert behind SOLD's real slot-history adapter.

    The adapter is the integration's own ``SlotHistoryAdapter`` -- upstream's
    ``Predictor`` under the hood -- because that is the piece whose causality
    and context bounding are under test. Only the 450M expert behind it is
    replaced.
    """
    torch = require_torch()

    from ..latent_actor import LatentActor
    from ..sold.adapter import SlotHistoryAdapter

    classes = stub_classes()
    torch.manual_seed(int(seed))
    adapter = SlotHistoryAdapter(
        num_slots=int(num_slots), slot_dim=int(slot_dim),
        token_dim=int(token_dim), context=int(context),
        max_episode_steps=int(context), head_token_dim=32, hidden_dim=32,
        num_heads=2, num_layers=1, num_mlp_layers=1)
    actor = classes["actor"](1, int(action_dim), token_dim=int(token_dim),
                             chunk_size=int(chunk_size),
                             flow_steps=int(flow_steps), seed=int(seed))
    # Swap the flat-feature adapter for the slot one; everything else about
    # the stub -- condition(), velocity_fn() -- is unchanged.
    actor.adapter = adapter
    return LatentActor(actor, instruction="stub")


def unzero_critic(agent, scale: float = 0.05, seed: int = 0) -> None:
    """Make TD-MPC2's Q heads a non-constant function of their input.

    ``common/init.py`` zeroes the last layer of the reward head and of the Q
    ensemble -- ``init.zero_([self._reward[-1].weight, self._Qs.params[-2]])``
    -- which is upstream's initialisation and a good one. It also means a
    freshly built agent has ``dQ/da == 0`` **exactly**, for every action, by
    construction.

    A gradient test that ran against that would report no gradient reaching
    the adapter and would be right about the number and wrong about the
    conclusion: nothing is broken, the head is constant. Worse, the same test
    would keep reporting zero if the conditioning path really were cut. So
    the tests perturb those weights first, and this is the one place that
    says why.
    """
    torch = require_torch()

    generator = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        agent.model._Qs.params[-2].normal_(0.0, float(scale), generator=generator)
        agent.model._reward[-1].weight.normal_(0.0, float(scale),
                                               generator=generator)
        for parameter, target in zip(agent.model._Qs.parameters(),
                                     agent.model._target_Qs.parameters()):
            target.copy_(parameter)


# -------------------------------------------------------------- tdmpc2 config
def tiny_tdmpc2_cfg(*, render_size: int = 64, cameras: int = 1,
                    action_dim: int = 7, include_state: bool = False,
                    proprio_dim: int = 0, device: str = "cpu", **over):
    """A TD-MPC2 config small enough to build on CPU in a second.

    ``render_size`` stays 64 because ``layers.conv`` asserts 64 or 128; what
    is made small is the channel count, the latent and the MLP widths. The
    architecture family is upstream's -- these are interface, alignment and
    gradient tests, not a claim about what this size learns.
    """
    from ..tdmpc2.config import build_cfg

    shape: Dict[str, Tuple[int, ...]] = {
        "rgb": (3 * int(cameras), int(render_size), int(render_size))}
    if include_state:
        shape["rgb-state"] = (int(proprio_dim),)
    settings = {
        "model_size": 1, "obs": "rgb", "include_state": bool(include_state),
        "render_size": int(render_size), "device": device,
        "num_envs": 2, "num_eval_envs": 2, "batch_size": 4, "horizon": 3,
        "num_samples": 16, "num_elites": 4, "num_pi_trajs": 2, "iterations": 2,
        "latent_dim": 32, "enc_dim": 32, "mlp_dim": 32, "num_q": 2,
        "num_bins": 11, "simnorm_dim": 4, "num_channels": 8,
        "rgb_state_enc_dim": 16, "rgb_state_latent_dim": 8,
        "num_cameras": int(cameras), "proprio_dim": int(proprio_dim),
    }
    settings.update(over)
    return build_cfg(settings, obs_shape=shape, action_dim=int(action_dim),
                     episode_length=24)
