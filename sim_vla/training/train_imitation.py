"""Stage 1B: adapter and action expert, on a frozen world model.

The world model is the object Stage 1A just trained, handed over in the same
process and frozen here. It is not reloaded from a checkpoint, because by
default Stage 1A does not write one. Episodes are encoded causally -- posterior
states from the reset observation forward, never from a future one -- and the
adapter and action expert are trained to reproduce the demonstrated action
chunk with a flow-matching loss.

What is supervised, and where it comes from
-------------------------------------------

The conditioning state at row ``t`` is the posterior ``s_t``, which consumed
``a_(t-1)``. The chunk supervised there is ``[a_t, ..., a_(t+H-1)]``, and those
come from ``action_target`` -- **not** from ``action``, which is the previous
action the posterior already ate. Training on ``action`` put the target inside
its own conditioning input and asked the policy to predict a value it had just
been shown.

Two masks, because they answer two questions:

* **eligibility** -- may the actor be trained at this row? It must be scored
  (not burn-in) and must have a target of its own. A chunk whose conditioning
  row is burn-in reaches forward into scored rows and would otherwise
  contribute loss from a state the batch never established.
* **availability** -- does a real action exist at this offset within the
  chunk? Past the end of an episode it does not, and that suffix is masked.

Eligible rows are selected *before* the actor call, so the expensive flow
forward runs on the rows that count rather than on a full ``B x T`` grid that
is then mostly masked away.

Chunking happens here rather than in the dataset. A stored dataset of
overlapping chunks is the same data written ``chunk_size`` times, and the
alignment that matters is a property of how it is cut, not of how it is stored.
The chunk dimension is kept all the way into the loss: flattening ``[B, T, H,
A]`` to ``[B*T, H, A]`` is a reshape, and collapsing ``H`` would quietly turn
chunk imitation into single-action imitation.

Posterior features are recomputed per batch rather than cached. A latent cache
is a function of the world model that produced it, and keeping one correct
across arms means carrying that model's identity with it; recomputing costs a
frozen forward pass and cannot go stale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ..data.batch import to_model_batch

from ..models.flow_sampler import flow_matching_loss
from ..runtime.checkpoint import CheckpointMeta, save

# The canonical key a window exposes for the action taken *at* each row.
TARGET_KEY = "action_target"


def chunk_targets(targets: torch.Tensor, available: torch.Tensor,
                  eligible: torch.Tensor, chunk: int
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(B, T, A) -> (B, T, chunk, A)`` with availability and eligibility.

    ``targets`` is ``action_target``: the action taken at each row.
    ``available`` is ``action_valid``: whether a real action was loaded there.
    ``eligible`` says which rows may be *conditioned* on -- scored, and with a
    target of their own.

    Returns the gathered chunk, a per-offset availability mask, and the
    per-row eligibility mask. The gather clamps past the end so the tensor
    stays rectangular; every clamped position is false in the mask, so a chunk
    half past the end contributes half a chunk of error rather than a full one
    against a repeated final action.

    ``targets`` and ``available`` are indexed by the **target axis**, which is
    ``lookahead`` rows longer than the observation axis that ``eligible``
    indexes. That is what lets a conditioning row near the end of a full
    interior window still be supervised on a whole chunk.
    """
    batch, span, dim = targets.shape
    steps = int(eligible.shape[1])
    index = torch.arange(chunk, device=targets.device).reshape(1, 1, chunk)
    base = torch.arange(steps, device=targets.device).reshape(1, steps, 1)
    offset = (base + index).clamp(max=span - 1)
    gather = offset.expand(batch, steps, chunk)
    gathered = torch.gather(
        targets.unsqueeze(2).expand(batch, span, chunk, dim), 1,
        gather.unsqueeze(-1).expand(batch, steps, chunk, dim))
    within = (base + index) < span
    mask = within & torch.gather(
        available.unsqueeze(2).expand(batch, span, chunk), 1, gather)
    # A chunk is only as long as its unbroken prefix of real actions: once a
    # window runs out of loaded targets, everything after that is padding even
    # if a later row happens to be marked available.
    mask = torch.cumprod(mask.to(torch.int8), dim=-1).bool()
    return gathered, mask, eligible


@dataclass
class ImitationConfig:
    chunk_size: int = 8
    flow_steps: int = 10
    lr: float = 1e-4
    steps: int = 20_000
    batch_size: int = 16
    log_every: int = 100
    grad_clip: float = 1.0


class ImitationTrainer:
    """Trains the adapter and the action expert; never the world model."""

    def __init__(self, world_model, actor, sampler, config: ImitationConfig,
                 *, device="cuda", normalizer=None, coords=None):
        self.world_model = world_model.eval()
        for parameter in self.world_model.parameters():
            parameter.requires_grad_(False)
        self.actor = actor
        self.sampler = sampler
        self.config = config
        self.device = torch.device(device)
        self.normalizer = normalizer
        self.coords = coords
        # The chunk the policy is supervised on has to be the chunk the
        # pretrained expert emits, or the loss is computed against a different
        # number of actions than the model produced.
        expert_chunk = int(getattr(actor, "chunk_size", config.chunk_size))
        if int(config.chunk_size) != expert_chunk:
            raise ValueError(
                f"ImitationConfig.chunk_size={config.chunk_size} but the actor "
                f"emits {expert_chunk} actions per call. Set them equal, or "
                "pass chunk_size through SmolVLAActor so the pretrained "
                "config is synchronised with it.")
        trainable = [p for p in self.actor.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                "nothing in the actor is trainable; the adapter and the action "
                "expert are supposed to be")
        self.optimizer = torch.optim.AdamW(trainable, lr=config.lr)
        self.step = 0

    def to_torch(self, batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Storage names to model names, in the one place that does it.

        The coordinates go through here too: ``action`` leaves in dynamics
        units for the RSSM and ``action_target`` in normalized units for the
        actor. Omitting them trained the world model on raw commands while
        inference fed it standardised ones.
        """
        return to_model_batch(batch, self.device, coords=self.coords,
                              normalizer=self.normalizer)

    def features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Causal posterior features. No gradient reaches the world model."""
        with torch.no_grad():
            out = self.world_model.observe(batch)
            return self.world_model.features(out["post"])

    def loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        if TARGET_KEY not in batch:
            raise KeyError(
                f"the batch has no {TARGET_KEY!r}; imitation is supervised on "
                "the action taken at each row, not on the previous action the "
                f"posterior consumed. Windows carry {sorted(batch)[:10]}")
        feat = self.features(batch)
        scored = batch["loss_mask"].bool()
        # The target axis is longer than the observation axis by the lookahead,
        # so eligibility is built from its leading rows only.
        available = batch["action_valid"].bool()
        steps = int(scored.shape[1])
        targets, target_mask, eligible = chunk_targets(
            batch[TARGET_KEY], available, scored & available[:, :steps],
            self.config.chunk_size)

        batch_size, steps = targets.shape[:2]
        flat_eligible = eligible.reshape(batch_size * steps)
        keep = torch.nonzero(flat_eligible, as_tuple=False).squeeze(-1)
        if keep.numel() == 0:
            # Explicit rather than a NaN mean over an empty selection. The
            # caller skips the optimizer step; a batch of pure burn-in is a
            # sampling accident, not a reason to stop.
            zero = feat.new_zeros((), requires_grad=False)
            return zero, {"eligible_rows": 0.0, "skipped": 1.0}

        # Selected before conditioning: the flow forward is the expensive part
        # and there is no reason to run it on rows that are masked out.
        flat_feat = feat.reshape(batch_size * steps, -1)[keep]
        flat_targets = targets.reshape(
            batch_size * steps, self.config.chunk_size, -1)[keep]
        flat_mask = target_mask.reshape(
            batch_size * steps, self.config.chunk_size)[keep]

        cond = self.actor.condition(flat_feat, batch.get("instruction"))
        loss, metrics = flow_matching_loss(
            self.actor.velocity_fn(), flat_targets, cond, mask=flat_mask)
        metrics = dict(metrics)
        metrics |= {"eligible_rows": float(keep.numel()),
                    "skipped": 0.0,
                    "target_fraction": float(flat_mask.float().mean())}
        return loss, metrics

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        tensors = self.to_torch(batch)
        loss, metrics = self.loss(tensors)
        if metrics.get("skipped"):
            self.step += 1
            return {"loss": float(loss.detach()), "grad_norm": 0.0,
                    **{k: float(v) for k, v in metrics.items()}}
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clipped = torch.nn.utils.clip_grad_norm_(
            [p for p in self.actor.parameters() if p.requires_grad],
            self.config.grad_clip)
        self.optimizer.step()
        self.step += 1
        return {"loss": float(loss.detach()), "grad_norm": float(clipped),
                **{k: float(v) for k, v in metrics.items()}}

    def fit(self, sampler, steps: Optional[int] = None) -> Dict[str, float]:
        total = int(steps or self.config.steps)
        last: Dict[str, float] = {}
        for _ in range(total):
            last = self.update(sampler.batch(self.config.batch_size))
            if self.step % self.config.log_every == 0:
                print(f"[imitation] step {self.step} loss {last['loss']:.4f}",
                      flush=True)
        return last

    def release(self) -> None:
        """Drop the optimizer state and any retained gradients.

        Stage 2 builds its own actor optimizer. Keeping this one alive holds a
        second copy of Adam's moments for every trainable parameter, on the
        device, for the whole of online training.
        """
        for parameter in self.actor.parameters():
            parameter.grad = None
        self.optimizer = None


@dataclass
class Stage1B:
    """What Stage 1B produces, in memory: the imitation-trained policy.

    Stage 2 takes ``actor`` directly. ``trainer`` is None once
    :meth:`ImitationTrainer.release` has run, which the pipeline does after any
    optional save and before online training starts.
    """

    actor: Any
    adapter: Any
    loaded: Any
    trainer: Any
    losses: Dict[str, float] = field(default_factory=dict)
    path: Optional[Path] = None


def action_width(cfg: Dict[str, Any], sampler) -> int:
    """The task's action width, from the window contract.

    Windows expose ``action``/``action_target``/``action_valid``; there is no
    ``actions`` key at this point, because ``layout.assemble`` renamed it when
    it split the previous action from the target. A configured width is checked
    against the data rather than trusted over it.
    """
    probe = sampler.batch(1)
    if TARGET_KEY not in probe:
        raise KeyError(
            f"the sampler produced {sorted(probe)[:10]} with no {TARGET_KEY!r}; "
            "sim_vla.data.layout.assemble is what every window source must "
            "go through")
    observed = int(np.asarray(probe[TARGET_KEY]).shape[-1])
    configured = int((cfg.get("task") or {}).get("action_dim") or 0)
    if configured and configured != observed:
        raise ValueError(
            f"task.action_dim={configured} but the demonstrations have "
            f"{observed} action dimensions. The dataset decides this.")
    return observed


def build_actor(cfg: Dict[str, Any], feature_dim: int, action_dim: int, *,
                device="cuda"):
    """The pretrained SmolVLA plus a fresh adapter sized for this arm.

    The adapter's width is decided by ``actor.state_token_mode`` and the
    checkpoint's own numbers, never by a constant here: ``embedding`` feeds the
    VLM hidden size and bypasses ``state_proj``, ``state_proj`` feeds
    ``max_state_dim`` through the frozen projection. ``feature_dim`` differs
    between the arms -- ``(h, z)`` against ``(h, z, g)`` -- which is why it is
    an argument rather than something read from a config.
    """
    from ..models.latent_adapter import LatentAdapter
    from ..models.pretrained import load_policy, model_facts
    from ..models.smolvla_actor import SmolVLAActor

    actor_cfg = dict(cfg["actor"])
    loaded = load_policy(str(actor_cfg["pretrained"]),
                         str(actor_cfg.get("revision") or "main"))
    # Placed before the actor is constructed: SmolVLAActor reads its device
    # from the pretrained weights and moves the adapter to match, so moving
    # the stack afterwards would leave the two apart.
    loaded.policy.to(device)

    facts = model_facts(loaded)
    mode = str(actor_cfg.get("state_token_mode") or "embedding")
    token_dim = int(facts["vlm_hidden_size"] if mode == "embedding"
                    else facts["max_state_dim"])
    adapter = LatentAdapter(int(feature_dim), token_dim)
    actor = SmolVLAActor(
        loaded, adapter, action_dim=int(action_dim),
        chunk_size=int(actor_cfg.get("chunk_size") or 0) or None,
        flow_steps=int(actor_cfg.get("flow_steps") or 0) or None,
        instruction=str(cfg["task"].get("instruction") or ""),
        state_token_mode=mode)

    execute = int(actor_cfg.get("execute") or 1)
    if not 1 <= execute <= int(actor.chunk_size):
        raise ValueError(
            f"actor.execute={execute} must be at least 1 and at most the "
            f"chunk size {actor.chunk_size}: more executed actions than the "
            "policy predicts would repeat or invent commands.")
    return actor


def run(cfg: Dict[str, Any], world_model, sampler, *, steps: int,
        device="cuda", actor=None, normalizer=None, coords=None,
        out: Optional[Path] = None, save_checkpoint: bool = False,
        meta: Optional[CheckpointMeta] = None,
        config: Optional[ImitationConfig] = None) -> Stage1B:
    """Train the adapter and action expert against a live world model.

    ``world_model`` is Stage 1A's object. ``save_checkpoint`` is off by
    default; Stage 2 takes ``Stage1B.actor`` from the return value.
    """
    dim = action_width(cfg, sampler)

    if actor is None:
        actor = build_actor(cfg, int(world_model.feature_dim), dim,
                            device=device)

    imitation = config or ImitationConfig(
        chunk_size=int(actor.chunk_size),
        flow_steps=int(actor.flow_steps),
        batch_size=int(cfg["data"]["batch_size"]),
        steps=int(steps))

    # Action-only lookahead so the last eligible rows of a window are
    # supervised on a whole chunk instead of a truncated one. Better
    # supervision, not a precondition: without it the suffix is simply masked.
    #
    # ``chunk_size``, not ``chunk_size - 1``. The last eligible conditioning
    # row is the window's final observation, and the action taken *there* is
    # already the first lookahead action -- so a full chunk at that row reaches
    # ``chunk_size`` actions past the window, not ``chunk_size - 1``.
    if hasattr(sampler, "lookahead"):
        sampler.lookahead = max(int(imitation.chunk_size), 0)

    trainer = ImitationTrainer(world_model, actor, sampler, imitation,
                               device=device, normalizer=normalizer,
                               coords=coords)
    last = trainer.fit(sampler, steps=int(steps))

    path: Optional[Path] = None
    if save_checkpoint:
        if out is None:
            raise ValueError(
                "save_checkpoint=True needs an output path; pass out=...")
        stage_meta = CheckpointMeta(
            **{**(meta.__dict__ if meta is not None
                  else {"graph_enabled": bool(cfg["model"]["graph"]["enabled"]),
                        "stage": "imitation"}),
               "stage": "imitation", "step": int(steps),
               "pretrained_revision": str(actor.loaded.revision)})
        path = save(Path(out), stage_meta,
                    {"adapter": actor.adapter, "actor": actor},
                    {"actor": trainer.optimizer})

    return Stage1B(actor=actor, adapter=actor.adapter, loaded=actor.loaded,
                   trainer=trainer, losses=last, path=path)
