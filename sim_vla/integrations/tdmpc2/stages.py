"""The three stages, for TD-MPC2.

Stage 1 -- native coupled pretraining
-------------------------------------

``TDMPC2.update(buffer)`` is called unchanged, against a buffer of
demonstrations instead of a live rollout. That is deliberate and not a
shortcut: the Q targets are ``r + gamma * Q_target(z', pi(z'))``, so the
auxiliary Gaussian prior has to be learning at the same time or the targets
bootstrap off a random policy. Coupling the two is TD-MPC2's design, and it is
preserved.

What trains in Stage 1, and with which optimizer:

==========================  =============================================
``model._encoder``          ``optim``, at ``lr * enc_lr_scale``
``model._dynamics``         ``optim``
``model._reward``           ``optim``
``model._Qs``               ``optim``
``model._pi`` (Gaussian)    ``pi_optim``, by ``update_pi``
``model._target_Qs``        no optimizer: Polyak, ``soft_update_target_Q``
==========================  =============================================

No SmolVLA is involved. The adapter does not exist yet and the flow policy is
not attached.

Stage 2 -- adapter and action expert, frozen world model
--------------------------------------------------------

Every parameter of the world model is set to ``requires_grad=False`` and the
model is put in ``eval``; the conditioning latent is computed under
``no_grad``. What trains is the adapter and SmolVLA's action expert, on a
flow-matching regression onto the demonstrated action chunk.

Conditioning at row ``t`` is ``z_t``, the existing encoder's output for
``o_t``. Nothing later than ``o_t`` reaches it. The chunk supervised there is
``[a_t .. a_(t+H-1)]``, taken from the target axis, masked by availability, and
the eligible rows are picked before the actor runs.

Stage 3 -- native online TD-MPC2, with SmolVLA as the policy
-------------------------------------------------------------

Upstream's ``OnlineTrainer``, ``Buffer``, ``Logger`` and environment, with the
flow policy attached to the agent. MPC stays the final action selector; what
changes is where the proposals, the bootstraps and the policy gradient come
from. See ``policy.py`` for the five call sites.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from ...data.normalization import fit_normalizer
from .. import chunking
from ..checkpoint import StageMeta, save as save_checkpoint
from ..latent_actor import LatentActor, assert_gradient_reaches
from ..vendor import TDMPC2 as VENDOR
from . import agent as build
from . import data as demo_data
from .config import architecture, build_cfg
from .policy import LatentPolicy, planner_cost


# ----------------------------------------------------------------- utilities
def freeze(model) -> int:
    """``requires_grad=False`` everywhere, without cutting the graph.

    Frozen means no gradient is *stored*; it does not mean detached. Nothing
    downstream of the world model needs a gradient into it here, but the same
    rule holds for SmolVLA's frozen transformer, where the gradient must pass
    through into the adapter.
    """
    count = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.requires_grad_(False)
            count += 1
    return count


def normalizer_for(cfg: Mapping[str, Any], source) -> Optional[Any]:
    """Statistics, only when a run actually asked to standardise actions."""
    mode = str((cfg.get("smolvla") or {}).get("action_normalization") or "identity")
    if mode == "identity":
        return None
    normalizer = fit_normalizer(source.dataset, fields=("actions", "proprio"))
    normalizer.mode = mode
    return normalizer


def stage_meta(cfg: Mapping[str, Any], *, stage: str, node, source,
               smolvla: bool, converter=None, actor=None, policy=None,
               parameters: Optional[Mapping[str, Any]] = None,
               step: int = 0, extra: Optional[Mapping[str, Any]] = None
               ) -> StageMeta:
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
        backend="tdmpc2", stage=stage,
        env_id=str(node.env_id), smolvla=bool(smolvla),
        architecture=architecture(node),
        normalization=(converter.descriptor() if converter is not None else {}),
        dataset_identity=source.identity(),
        actor=actor_meta,
        policy=(policy.descriptor() if policy is not None else {}),
        parameters=dict(parameters or {}),
        step=int(step), extra=dict(extra or {}))


# -------------------------------------------------------------------- stage 1
@dataclass
class WorldModelResult:
    agent: Any
    node: Any
    source: Any
    converter: Any
    normalizer: Any
    metrics: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None


def pretrain_world_model(cfg: Mapping[str, Any], *, steps: int,
                         device: str = "cuda", out: Optional[Path] = None,
                         save: bool = False,
                         log: Optional[Callable[[str], None]] = print
                         ) -> WorldModelResult:
    """Upstream's coupled offline update, on the target demonstrations."""
    world = dict(cfg.get("world_model") or {})
    smolvla = dict(cfg.get("smolvla") or {})
    data_cfg = dict(cfg.get("data") or {})

    source = demo_data.open_demos(
        str(data_cfg.get("dataset") or cfg["task"]["dataset"]),
        render_size=int(world.get("render_size", 64)),
        include_state=bool(world.get("include_state", False)),
        cameras=data_cfg.get("cameras"))
    # The camera count and the proprioception width are properties of the
    # dataset, not of a preset, and they change the encoder's input. Recorded
    # in the config so they land in the checkpoint's architecture block.
    node = build_cfg({**world, "device": device,
                      "num_cameras": len(source.images.cameras),
                      "proprio_dim": source.proprio_dim
                      if source.include_state else 0},
                     obs_shape=source.obs_shape(),
                     action_dim=source.action_dim,
                     episode_length=source.episode_length)
    normalizer = normalizer_for(cfg, source)
    converter = build.build_converter(source.metadata,
                                      action_dim=source.action_dim,
                                      smolvla=smolvla, normalizer=normalizer,
                                      device=device)

    agent = build.build_agent(node)
    buffer = demo_data.DemoBuffer(
        source, horizon=int(node.horizon), batch_size=int(node.batch_size),
        device=device, seed=int(data_cfg.get("seed", 0)), converter=converter)

    parameters = build.parameters(agent)
    if log:
        from ..params import render

        log(render(parameters, title="TD-MPC2 world model (stage 1)"))
        log(f"[stage1] {len(buffer)} unpadded windows of {node.horizon} "
            f"transitions from {len(source.dataset.episodes)} episodes")

    every = int((cfg.get("stages") or {}).get("world_model", {}).get("log_every", 500))
    metrics: Dict[str, float] = {}
    started = time.time()
    for step in range(int(steps)):
        metrics = agent.update(buffer)
        if log and every and (step + 1) % every == 0:
            log(f"[stage1] {step + 1}/{steps} "
                + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                           if isinstance(v, float))
                + f" ({time.time() - started:.0f}s)")

    path = None
    if save:
        if out is None:
            raise ValueError("save=True needs an output path; pass out=...")
        meta = stage_meta(cfg, stage="world_model", node=node, source=source,
                          smolvla=False, converter=converter,
                          parameters=parameters, step=int(steps))
        path = save_checkpoint(Path(out) / "tdmpc2_world_model.pt", meta,
                               {"model": agent.model},
                               {"optim": agent.optim, "pi_optim": agent.pi_optim})
    return WorldModelResult(agent=agent, node=node, source=source,
                            converter=converter, normalizer=normalizer,
                            metrics=metrics, parameters=parameters, path=path)


# -------------------------------------------------------------------- stage 2
@dataclass
class ImitationConfig:
    steps: int = 20_000
    batch_size: int = 16
    lr: float = 1e-4
    grad_clip: float = 1.0
    log_every: int = 100
    sequence_length: int = 32
    burn_in: int = 0


class ImitationTrainer:
    """Adapter and action expert, on a frozen TD-MPC2 world model."""

    def __init__(self, agent, actor: LatentActor, source, converter, *,
                 config: ImitationConfig, device="cuda", seed: int = 0):
        self.agent = agent
        self.model = agent.model
        self.actor = actor
        self.source = source
        self.converter = converter
        self.config = config
        self.device = torch.device(device)
        self.frozen = freeze(self.model)
        self.model.eval()

        if int(config.batch_size) < 1:
            raise ValueError("imitation batch_size must be at least 1")
        trainable = actor.trainable_parameters()
        if not trainable:
            raise RuntimeError(
                "nothing in the actor is trainable; the adapter and the action "
                "expert are supposed to be")
        self.optimizer = torch.optim.AdamW(trainable, lr=float(config.lr))
        # The lookahead is the chunk length, not chunk - 1: the last eligible
        # conditioning row is the window's final observation, and the action
        # taken *there* is already the first lookahead action.
        self.windows = demo_data.DemoWindows(
            source, horizon=int(config.sequence_length),
            burn_in=int(config.burn_in), seed=int(seed), stride=1,
            lookahead=int(actor.chunk_size))
        # Padding is masked, so imitation does not need whole windows the way
        # the native update does; using every window is more data, and the
        # masks are what keep the padding out of the loss.
        self.windows.windows = list(self.windows.sampler.windows)
        self.step = 0

    @torch.no_grad()
    def latents(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """``z_t`` for every row, causal by construction.

        ``model.encode`` sees one observation at a time; there is no recurrence
        and no window, so row ``t`` cannot depend on anything after ``o_t``.
        The random-shift augmentation upstream applies inside the encoder is
        left in place -- it is part of TD-MPC2's encoder, in training and at
        action-selection time alike.
        """
        obs = demo_data.observations(batch, self.source, device=self.device)
        z = self.model.encode(obs, None)                   # (T, B, latent)
        return z.movedim(0, 1).contiguous()                # (B, T, latent)

    def loss(self, batch: Mapping[str, Any]):
        tensors = {k: (v if isinstance(v, torch.Tensor)
                       else torch.as_tensor(np.asarray(v)))
                   for k, v in batch.items()}
        selection = chunking.select(
            {k: (v.to(self.device) if v.dtype != torch.uint8 else v)
             for k, v in tensors.items()},
            int(self.actor.chunk_size), converter=self.converter)
        if selection.empty:
            zero = torch.zeros((), device=self.device)
            return zero, {"eligible_rows": 0.0, "skipped": 1.0}
        feature = self.latents(batch)
        rows = selection.features(feature)
        loss, metrics = self.actor.flow_loss(
            rows, selection.targets.to(rows.device),
            selection.mask.to(rows.device))
        return loss, {**{k: float(v) for k, v in metrics.items()},
                      **selection.stats(), "skipped": 0.0}

    def update(self) -> Dict[str, float]:
        batch = self.windows.raw_batch(int(self.config.batch_size))
        loss, metrics = self.loss(batch)
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
                log(f"[stage2] step {self.step}/{total} loss {last['loss']:.4f} "
                    f"rows {last.get('eligible_rows', 0):.0f}")
        return last

    def release(self) -> None:
        """Drop the optimizer state before Stage 3 builds its own.

        Two copies of Adam's moments for a 450M action expert is 3.6 GB of
        device memory doing nothing for the rest of the run.
        """
        for parameter in self.actor.parameters():
            parameter.grad = None
        self.optimizer = None


@dataclass
class ImitationResult:
    actor: LatentActor
    trainer: ImitationTrainer
    converter: Any
    metrics: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    path: Optional[Path] = None


def train_imitation(cfg: Mapping[str, Any], world: WorldModelResult, *,
                    steps: int, device: str = "cuda",
                    actor: Optional[LatentActor] = None,
                    out: Optional[Path] = None, save: bool = False,
                    log: Optional[Callable[[str], None]] = print
                    ) -> ImitationResult:
    smolvla = dict(cfg.get("smolvla") or {})
    stage_cfg = dict((cfg.get("stages") or {}).get("imitation") or {})
    data_cfg = dict(cfg.get("data") or {})

    if actor is None:
        actor = build.build_actor(
            smolvla, feature_dim=build.latent_dim(world.agent),
            action_dim=world.source.action_dim, device=device)

    config = ImitationConfig(
        steps=int(steps),
        batch_size=int(stage_cfg.get("batch_size", 16)),
        lr=float(stage_cfg.get("lr", 1e-4)),
        grad_clip=float(stage_cfg.get("grad_clip", 1.0)),
        log_every=int(stage_cfg.get("log_every", 100)),
        sequence_length=int(data_cfg.get("sequence_length", 32)),
        burn_in=int(data_cfg.get("burn_in", 0)))

    trainer = ImitationTrainer(world.agent, actor, world.source,
                               world.converter, config=config, device=device,
                               seed=int(data_cfg.get("seed", 0)))
    parameters = build.parameters(world.agent, actor=actor)
    if log:
        from ..params import render

        log(render(parameters, title="TD-MPC2 + SmolVLA (stage 2)"))
        log(f"[stage2] world model frozen: {trainer.frozen} tensors; "
            f"trainable actor parameters "
            f"{sum(p.numel() for p in actor.trainable_parameters()):,}")

    metrics = trainer.fit(int(steps), log=log)

    path = None
    if save:
        if out is None:
            raise ValueError("save=True needs an output path; pass out=...")
        meta = stage_meta(cfg, stage="imitation", node=world.node,
                          source=world.source, smolvla=True,
                          converter=world.converter, actor=actor,
                          parameters=parameters, step=int(steps))
        path = save_checkpoint(Path(out) / "tdmpc2_imitation.pt", meta,
                               {"model": world.agent.model,
                                "adapter": actor.adapter,
                                "actor": actor},
                               {"actor": trainer.optimizer})
    return ImitationResult(actor=actor, trainer=trainer,
                           converter=world.converter, metrics=metrics,
                           parameters=parameters, path=path)


# -------------------------------------------------------------------- stage 3
def check_environment_contract(env, source, node,
                               log: Optional[Callable[[str], None]] = print
                               ) -> Dict[str, Any]:
    """Refuse an environment that is not the one the demonstrations came from.

    Stage 1 and 2 trained an encoder on a particular camera set at a
    particular resolution and a policy on a particular action width. A live
    environment that differs in any of those produces observations the encoder
    has never seen, and nothing about it fails -- the returns are simply bad.
    """
    wanted = source.contract()
    observed: Dict[str, Any] = {"env_id": str(node.env_id)}
    shapes = {str(k): tuple(int(x) for x in v)
              for k, v in dict(node.obs_shape).items()}
    observed["obs_shape"] = shapes
    problems: List[str] = []

    rgb = shapes.get("rgb")
    if rgb is None:
        problems.append("the environment produces no 'rgb' observation")
    else:
        if int(rgb[0]) != int(wanted["rgb_channels"]):
            problems.append(
                f"rgb channels: demonstrations {wanted['rgb_channels']} "
                f"({len(wanted['cameras'])} camera(s)) but the env gives "
                f"{rgb[0]}")
        if [int(rgb[1]), int(rgb[2])] != list(wanted["render_size"]):
            problems.append(
                f"render size: demonstrations {wanted['render_size']} but the "
                f"env gives {[int(rgb[1]), int(rgb[2])]}")
    if wanted["include_state"]:
        state = shapes.get("rgb-state")
        if state is None:
            problems.append("include_state is on but the env gives no state")
        elif int(state[0]) != int(wanted["proprio_dim"]):
            problems.append(
                f"state width: demonstrations {wanted['proprio_dim']} but the "
                f"env gives {int(state[0])}")
    elif "rgb-state" in shapes:
        problems.append(
            "the env supplies a state vector the world model was not built to "
            "read; set include_state to match the pretraining run")
    if int(node.action_dim) != int(wanted["action_dim"]):
        problems.append(
            f"action_dim: demonstrations {wanted['action_dim']} but the env "
            f"gives {int(node.action_dim)}")
    if str(node.env_id) != str(wanted["env_id"]):
        problems.append(
            f"env_id: demonstrations {wanted['env_id']!r} but this run builds "
            f"{str(node.env_id)!r}")

    if problems:
        raise SystemExit(
            "the online environment is not the one these weights were trained "
            "against:\n  - " + "\n  - ".join(problems)
            + "\nThe demonstrations' contract is "
            + repr(wanted) + ". Fix the env settings (render_size, "
            "control_mode, camera set) rather than the check.")
    if log:
        log(f"[stage3] environment contract matches the demonstrations: {wanted}")
    return wanted


def attach(agent, actor: LatentActor, converter, cfg: Mapping[str, Any], *,
           descriptor: Optional[Dict[str, Any]] = None,
           log: Optional[Callable[[str], None]] = print) -> LatentPolicy:
    """Hand TD-MPC2's learned-policy call sites to the flow policy."""
    smolvla = dict(cfg.get("smolvla") or {})
    policy = build.build_policy(actor, converter, smolvla,
                                descriptor=descriptor)
    deviations = build.apply_planner_overrides(agent.cfg, smolvla)
    agent.attach_policy(policy)

    cost = planner_cost(agent.cfg, sites=policy.sites,
                        flow_steps=int(actor.flow_steps),
                        proposal_flow_steps=policy.proposal_flow_steps)
    if log:
        log(f"[stage3] SmolVLA serves {list(policy.sites)}")
        log(f"[stage3] Gaussian prior after handoff: "
            f"{policy.descriptor()['gaussian_after_handoff']}")
        log(f"[stage3] flow-sampler cost per environment step: "
            f"{cost['total_rows_per_env_step']:,} sampled chunks, "
            f"{cost['total_denoise_passes_per_env_step']:,} denoising passes "
            f"({cost['rows_per_env_step']})")
        if deviations:
            log(f"[stage3] planner settings changed from native: {deviations}")

    # Once, loudly, before a run that would otherwise train for hours without
    # moving: a sampler built under no_grad gives an actor update that costs
    # the same and learns nothing.
    probe = torch.zeros((2, int(agent.cfg.true_latent_dim)),
                        device=next(agent.model.parameters()).device)
    action = policy.sample(probe, None, grad=True)
    assert_gradient_reaches(action, actor.trainable_parameters()[:4],
                            what="TD-MPC2's policy proposal")
    return policy
