"""The three stages, for SOLD.

Stage 1A -- SAVi, on the target demonstrations
-----------------------------------------------

SOLD's two-stage recipe pretrains the autoencoder before anything else, and
the ``checkpoints/`` directory in the vendored tree holds SAVi models for
``reach_red`` and ``push_red``. Those are multi-object-fetch scenes with a
different robot, a different action space and a different camera; a slot
decomposition fitted to them is not a starting point for a ManiSkill table-top
task. So Stage 1A trains SAVi here, on the dataset the other two stages use.

The objective is upstream's: reconstruct every frame of the sequence from the
slots, ``F.mse_loss(reconstructions, images)``, with the slot predictor
advanced by the action taken at each frame. When Lightning is installed the
run uses ``AutoencoderModule.compute_reconstruction_loss`` itself; without it
the same two lines run here, and ``test_sold_native`` asserts the two agree.

Stage 1B -- slot dynamics and the reward head
-----------------------------------------------

``SOLDModule.compute_dynamics_loss`` and ``compute_reward_loss``, called as
upstream wrote them. They are methods, but they touch only attributes, so they
run against a holder that carries the same ones -- which is how the native
losses are used verbatim without standing up Lightning and a simulator.

Slots are detached before the dynamics loss, exactly as ``training_step``
detaches them, so no gradient reaches the autoencoder from here. Whether the
autoencoder trains at all is ``finetune_autoencoder``, and that setting is
honoured rather than overridden.

Stage 2 -- adapter and action expert, frozen world model
----------------------------------------------------------

Every parameter of SAVi, the dynamics and the reward head is frozen and the
slots are computed under ``no_grad``. Conditioning at row ``t`` is the causal
slot history ending at ``t``, bounded to the adapter's context; the chunk
supervised there is ``[a_t .. a_(t+H-1)]``, masked by availability.

Burn-in is ``context - 1`` rows, so every eligible row has a full history --
a row conditioned on two frames when the policy will always be given three is
a different question from the one it will be asked.

Stage 3 -- native online SOLD, with SmolVLA as the actor
----------------------------------------------------------

Upstream's ``SOLDModule`` and Lightning loop, with the flow policy attached.
The imagined rollout, the lambda returns, the critic's regularized objective
and the target EMA are untouched; what changes is where the imagined actions
come from and that the entropy bonus is absent. See ``policy.py``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .. import chunking
from ..checkpoint import StageMeta, save as save_checkpoint
from ..latent_actor import LatentActor, assert_gradient_reaches
from ..vendor import SOLD as VENDOR
from . import data as demo_data
from . import model as build
from .config import architecture, context_bounds


def freeze(*modules) -> int:
    count = 0
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad:
                parameter.requires_grad_(False)
                count += 1
    return count


def thaw(*modules) -> int:
    count = 0
    for module in modules:
        for parameter in module.parameters():
            if not parameter.requires_grad:
                parameter.requires_grad_(True)
                count += 1
    return count


class Holder:
    """The attributes upstream's loss methods read, and nothing else.

    ``SOLDModule.compute_dynamics_loss`` and ``compute_reward_loss`` are
    methods on a ``LightningModule``, but what they touch is a handful of
    attributes and ``self.log``. Binding them to this means the *native*
    functions compute the offline losses -- not a reimplementation of them --
    on a machine that has torch and nothing else.
    """

    def __init__(self, **attributes):
        self.__dict__.update(attributes)
        self.after_eval = False

    def log(self, *args, **kwargs):                        # noqa: D401
        """Swallowed: the visualisations are Lightning's, not the loss."""
        return None


# ------------------------------------------------------------------ stage 1A
def reconstruction_loss(autoencoder, images: torch.Tensor,
                        actions: torch.Tensor) -> Dict[str, Any]:
    """SAVi's own objective.

    Upstream's ``AutoencoderModule.compute_reconstruction_loss`` is these two
    lines with ``self.autoencoder``. It is used directly when Lightning is
    installed -- see :func:`native_reconstruction_loss` -- and
    ``test_sold_native`` asserts the two produce the same number.
    """
    outputs = autoencoder(images, actions[:, 1:])
    return {**outputs,
            "reconstruction_loss": F.mse_loss(outputs["reconstructions"], images),
            "images": images}


def native_reconstruction_loss(autoencoder, images, actions):
    """The same, through upstream's own function. Needs Lightning."""
    with VENDOR.active():
        from train_autoencoder import AutoencoderModule

        return AutoencoderModule.compute_reconstruction_loss(
            Holder(autoencoder=autoencoder), images, actions)


@dataclass
class AutoencoderConfig:
    steps: int = 20_000
    batch_size: int = 8
    sequence_length: int = 8
    lr: float = 1e-4
    grad_clip: float = 0.05
    log_every: int = 200


def pretrain_autoencoder(autoencoder, source, config: AutoencoderConfig, *,
                         device="cuda", seed: int = 0, converter=None,
                         log: Optional[Callable[[str], None]] = print
                         ) -> Dict[str, float]:
    """Fit SAVi to the target demonstrations."""
    autoencoder = autoencoder.to(device).train()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=float(config.lr))
    loader = demo_data.DemoLoader(
        source, sequence_length=int(config.sequence_length),
        batch_size=int(config.batch_size), device=device, seed=int(seed),
        converter=converter)
    if log:
        log(f"[sold:1a] {len(loader)} windows of {config.sequence_length} "
            f"frames from {len(source.dataset.episodes)} episodes")
    started = time.time()
    last: Dict[str, float] = {}
    for step in range(int(config.steps)):
        batch = loader.sample()
        images = batch["obs"].float() / 255.0
        # The first row's previous action is NaN, and `actions[:, 1:]` is what
        # the slot predictor advances on, so it never reaches the model. The
        # substitution is here anyway: a NaN that slipped through would poison
        # every slot in the batch and report as a loss of nan several steps
        # later.
        actions = torch.nan_to_num(batch["action"])
        outputs = reconstruction_loss(autoencoder, images, actions)
        optimizer.zero_grad(set_to_none=True)
        outputs["reconstruction_loss"].backward()
        torch.nn.utils.clip_grad_norm_(autoencoder.parameters(),
                                       float(config.grad_clip))
        optimizer.step()
        last = {"reconstruction_loss": float(outputs["reconstruction_loss"])}
        if log and config.log_every and (step + 1) % int(config.log_every) == 0:
            log(f"[sold:1a] {step + 1}/{config.steps} "
                f"reconstruction {last['reconstruction_loss']:.5f} "
                f"({time.time() - started:.0f}s)")
    return last


# ------------------------------------------------------------------ stage 1B
@dataclass
class WorldModelConfig:
    steps: int = 50_000
    batch_size: int = 8
    log_every: int = 200


class WorldModelTrainer:
    """Slot dynamics and the reward head, through upstream's own losses."""

    def __init__(self, parts: Mapping[str, Any], world: Mapping[str, Any],
                 source, config: WorldModelConfig, *, device="cuda",
                 seed: int = 0, converter=None):
        self.parts = dict(parts)
        self.world = dict(world)
        self.source = source
        self.config = config
        self.device = torch.device(device)
        self.converter = converter

        low, high = context_bounds(world)
        self.min_num_context, self.max_num_context = low, high
        self.imagination_horizon = int(world.get("imagination_horizon", 15))
        self.sequence_length = self.imagination_horizon + high

        self.autoencoder = self.parts["autoencoder"].to(device)
        self.dynamics = self.parts["dynamics"].to(device)
        self.reward = self.parts["reward"].to(device)

        self.finetune_autoencoder = bool(world.get("finetune_autoencoder", False))
        self.autoencoder_optimizer = torch.optim.Adam(
            self.autoencoder.parameters(),
            lr=float(world.get("autoencoder_learning_rate", 1e-4)))
        self.autoencoder_grad_clip = float(
            world.get("autoencoder_grad_clip", 0.05))
        self.dynamics_optimizer = torch.optim.Adam(
            self.dynamics.parameters(),
            lr=float(world.get("dynamics_learning_rate", 1e-4)))
        self.dynamics_grad_clip = float(world.get("dynamics_grad_clip", 3.0))
        self.reward_optimizer = torch.optim.Adam(
            self.reward.parameters(),
            lr=float(world.get("reward_learning_rate", 1e-4)))
        self.reward_grad_clip = float(world.get("reward_grad_clip", 10.0))

        self.loader = demo_data.DemoLoader(
            source, sequence_length=self.sequence_length,
            batch_size=int(config.batch_size), device=device, seed=int(seed),
            converter=converter)
        self.step = 0

    def _holder(self) -> Holder:
        return Holder(autoencoder=self.autoencoder,
                      dynamics_predictor=self.dynamics,
                      reward_predictor=self.reward,
                      imagination_horizon=self.imagination_horizon,
                      min_num_context=self.min_num_context,
                      max_num_context=self.max_num_context)

    def update(self) -> Dict[str, float]:
        with VENDOR.active():
            from train_sold import SOLDModule

        batch = self.loader.sample()
        images = batch["obs"].float() / 255.0
        actions = torch.nan_to_num(batch["action"])
        rewards = batch["reward"]

        holder = self._holder()
        if self.finetune_autoencoder:
            self.autoencoder_optimizer.zero_grad(set_to_none=True)
        outputs = reconstruction_loss(self.autoencoder, images, actions)
        if self.finetune_autoencoder:
            outputs["reconstruction_loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(),
                                           self.autoencoder_grad_clip)
            self.autoencoder_optimizer.step()

        # Upstream detaches here; the dynamics never push gradient into the
        # autoencoder whether or not it is being fine-tuned.
        slots = outputs["slots"].detach()

        self.dynamics_optimizer.zero_grad(set_to_none=True)
        dynamics = SOLDModule.compute_dynamics_loss(holder, images, slots,
                                                    actions)
        dynamics["dynamics_loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(),
                                       self.dynamics_grad_clip)
        self.dynamics_optimizer.step()

        self.reward_optimizer.zero_grad(set_to_none=True)
        reward = SOLDModule.compute_reward_loss(
            holder, images, outputs["reconstructions"].detach(), slots, rewards)
        reward["reward_loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.reward.parameters(),
                                       self.reward_grad_clip)
        self.reward_optimizer.step()

        self.step += 1
        return {"reconstruction_loss": float(outputs["reconstruction_loss"]),
                "slot_loss": float(dynamics["slot_loss"]),
                "image_loss": float(dynamics["image_loss"]),
                "dynamics_loss": float(dynamics["dynamics_loss"]),
                "reward_loss": float(reward["reward_loss"]),
                "reward_mse_loss": float(reward["reward_mse_loss"])}

    def fit(self, steps: Optional[int] = None,
            log: Optional[Callable[[str], None]] = print) -> Dict[str, float]:
        total = int(steps if steps is not None else self.config.steps)
        started = time.time()
        last: Dict[str, float] = {}
        for _ in range(total):
            last = self.update()
            if log and self.config.log_every and \
                    self.step % int(self.config.log_every) == 0:
                log(f"[sold:1b] {self.step}/{total} "
                    + " ".join(f"{k}={v:.4f}" for k, v in last.items())
                    + f" ({time.time() - started:.0f}s)")
        return last


# ------------------------------------------------------------------- stage 2
@dataclass
class ImitationConfig:
    steps: int = 20_000
    batch_size: int = 8
    lr: float = 1e-4
    grad_clip: float = 1.0
    log_every: int = 100
    sequence_length: int = 16


class ImitationTrainer:
    """Adapter and action expert, on a frozen slot world model."""

    def __init__(self, parts: Mapping[str, Any], actor: LatentActor, source,
                 converter, *, config: ImitationConfig, device="cuda",
                 seed: int = 0):
        self.parts = dict(parts)
        self.autoencoder = self.parts["autoencoder"].to(device).eval()
        self.actor = actor
        self.source = source
        self.converter = converter
        self.config = config
        self.device = torch.device(device)

        self.frozen = freeze(self.autoencoder, self.parts["dynamics"],
                             self.parts["reward"], self.parts["critic"],
                             self.parts["actor"])
        self.context = int(self.actor.adapter.context)
        # Every eligible row has to carry a full slot history, or it is a
        # different question from the one the policy is asked online.
        self.burn_in = max(self.context - 1, 0)

        trainable = actor.trainable_parameters()
        if not trainable:
            raise RuntimeError(
                "nothing in the actor is trainable; the adapter and the action "
                "expert are supposed to be")
        self.optimizer = torch.optim.AdamW(trainable, lr=float(config.lr))
        self.windows = demo_data.SoldWindows(
            source, length=int(config.sequence_length), burn_in=self.burn_in,
            seed=int(seed), stride=1, lookahead=int(actor.chunk_size),
            full_only=False)
        self.step = 0

    @torch.no_grad()
    def slots(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Causal slots for every row. No gradient reaches the autoencoder.

        ``SAVi.encode`` walks the sequence forward, carrying each frame's slots
        into the next frame's initialisation through the action-conditioned
        predictor. Row ``t`` therefore depends on frames up to ``t`` and on
        nothing after it.
        """
        images = batch["obs"].float() / 255.0
        actions = torch.nan_to_num(batch["action"])
        return self.autoencoder.encode(images, actions[:, 1:])

    def loss(self, raw: Mapping[str, Any]):
        batch = demo_data.sold_batch(raw, self.source, device=self.device,
                                     converter=self.converter)
        masks = {key: torch.as_tensor(np.asarray(raw[key])).to(self.device)
                 for key in ("action_target", "action_valid", "loss_mask")}
        selection = chunking.select(masks, int(self.actor.chunk_size),
                                    converter=self.converter)
        if selection.empty:
            return torch.zeros((), device=self.device), {"eligible_rows": 0.0,
                                                         "skipped": 1.0}
        # Rows without a full slot history are not conditioning rows here.
        within = selection.rows % selection.steps
        keep = within >= self.burn_in
        rows = selection.rows[keep]
        if int(rows.numel()) == 0:
            return torch.zeros((), device=self.device), {"eligible_rows": 0.0,
                                                         "skipped": 1.0}
        slots = self.slots(batch)
        windows = self.actor.adapter.select_windows(slots, rows)
        loss, metrics = self.actor.flow_loss(
            windows, selection.targets[keep].to(windows.device),
            selection.mask[keep].to(windows.device))
        return loss, {**{k: float(v) for k, v in metrics.items()},
                      "eligible_rows": float(rows.numel()),
                      "target_fraction": float(selection.mask[keep].float().mean()),
                      "skipped": 0.0}

    def update(self) -> Dict[str, float]:
        raw = self.windows.raw_batch(int(self.config.batch_size))
        loss, metrics = self.loss(raw)
        if metrics.get("skipped"):
            self.step += 1
            return {"loss": float(loss.detach()), "grad_norm": 0.0, **metrics}
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clipped = torch.nn.utils.clip_grad_norm_(
            self.actor.trainable_parameters(), float(self.config.grad_clip))
        self.optimizer.step()
        self.step += 1
        return {"loss": float(loss.detach()), "grad_norm": float(clipped),
                **metrics}

    def fit(self, steps: Optional[int] = None,
            log: Optional[Callable[[str], None]] = print) -> Dict[str, float]:
        total = int(steps if steps is not None else self.config.steps)
        last: Dict[str, float] = {}
        for _ in range(total):
            last = self.update()
            if log and self.config.log_every and \
                    self.step % int(self.config.log_every) == 0:
                log(f"[sold:2] step {self.step}/{total} loss {last['loss']:.4f} "
                    f"rows {last.get('eligible_rows', 0):.0f}")
        return last

    def release(self) -> None:
        for parameter in self.actor.parameters():
            parameter.grad = None
        self.optimizer = None


# ------------------------------------------------------------------ metadata
def stage_meta(cfg: Mapping[str, Any], *, stage: str, parts, source,
               smolvla: bool, converter=None, actor=None, policy=None,
               parameters: Optional[Mapping[str, Any]] = None,
               max_episode_steps: int = 150, step: int = 0,
               extra: Optional[Mapping[str, Any]] = None) -> StageMeta:
    world = dict(cfg["world_model"])
    adapter_context = int(getattr(getattr(actor, "adapter", None), "context", 0))
    actor_meta: Dict[str, Any] = {}
    if actor is not None:
        inner = getattr(actor, "actor", None)
        loaded = getattr(inner, "loaded", None)
        actor_meta = {
            "repo_id": getattr(loaded, "repo_id", ""),
            "revision": getattr(loaded, "revision", ""),
            "lerobot_version": getattr(loaded, "lerobot_version", ""),
            "chunk_size": int(actor.chunk_size),
            "flow_steps": int(actor.flow_steps),
            "action_dim": int(actor.action_dim),
            "state_token_mode": getattr(inner, "state_token_mode", ""),
            "instruction": actor.instruction,
            "execute": int((cfg.get("smolvla") or {}).get("execute", 1)),
        }
    return StageMeta(
        backend="sold", stage=stage,
        env_id=str(source.metadata.get("env_id") or ""), smolvla=bool(smolvla),
        architecture=architecture(
            world, num_slots=int(parts["num_slots"]),
            slot_dim=int(parts["slot_dim"]), action_dim=int(source.action_dim),
            image_size=source.images.size, max_episode_steps=max_episode_steps,
            adapter_context=adapter_context),
        normalization=(converter.descriptor() if converter is not None else {}),
        dataset_identity=source.identity(),
        actor=actor_meta,
        policy=(policy.descriptor() if policy is not None else {}),
        parameters=dict(parameters or {}),
        step=int(step), extra=dict(extra or {}))


def save_stage(path: Path, meta: StageMeta, parts: Mapping[str, Any],
               actor: Optional[LatentActor] = None,
               optimizers: Optional[Mapping[str, Any]] = None) -> Path:
    modules = {
        "autoencoder": parts["autoencoder"],
        "dynamics": parts["dynamics"],
        "reward": parts["reward"],
        "actor_gaussian": parts["actor"],
        "critic": parts["critic"],
        "critic_target": parts["critic_target"],
    }
    if actor is not None:
        modules["adapter"] = actor.adapter
        modules["smolvla_actor"] = actor
    return save_checkpoint(path, meta, modules, optimizers)


def attach(module, actor: LatentActor, converter, cfg: Mapping[str, Any], *,
           descriptor: Optional[Dict[str, Any]] = None,
           log: Optional[Callable[[str], None]] = print):
    """Hand SOLD's actor sampling to the flow policy."""
    from .policy import LatentSlotPolicy

    smolvla = dict(cfg.get("smolvla") or {})
    low, _high = context_bounds(cfg["world_model"])
    policy = LatentSlotPolicy(
        actor, converter, lr=float(smolvla.get("online_lr", 1e-5)),
        execute=int(smolvla.get("execute", 1)),
        max_batch=int(smolvla.get("max_batch", 256)),
        min_num_context=low, descriptor=descriptor)
    module.attach_policy(policy)
    if log:
        log(f"[sold:3] SmolVLA is the actor; the Gaussian actor is retained "
            "for checkpoint shape and for smolvla.enabled=false, and is not "
            "trained or sampled after handoff")
        log(f"[sold:3] actor objective: lambda-return advantage under dynamics "
            f"gradients, entropy bonus disabled "
            f"(weight {cfg['world_model'].get('actor_entropy_loss_weight')} "
            "has no term to scale)")
        log(f"[sold:3] slot-history context bounded to {policy.context} "
            "frames in imitation, imagination and inference alike")

    device = next(module.parameters()).device
    probe = torch.zeros((2, policy.context, int(actor.adapter.num_slots),
                         int(actor.adapter.slot_dim)), device=device)
    action = policy.sample(probe, grad=True)
    assert_gradient_reaches(action, actor.trainable_parameters()[:4],
                            what="SOLD's imagined action")
    return policy
