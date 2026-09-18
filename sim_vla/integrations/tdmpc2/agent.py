"""Building the TD-MPC2 agent, the adapter and the flow policy.

Nothing here is an algorithm. It constructs upstream's ``TDMPC2`` from
upstream's ``WorldModel``, builds one small adapter from that model's own
latent to one SmolVLA conditioning token, and wires the two together through
the hooks in ``sim_vla/tdmpc2/tdmpc2.py``.

The adapter's input width is ``cfg.true_latent_dim``, which ``WorldModel``
computes and writes back onto the config: ``latent_dim`` on its own, plus
``rgb_state_latent_dim`` when the state encoder is present. Reading it from
the built model rather than from the config is deliberate -- the config value
does not exist until the model has been constructed.

``z`` is TD-MPC2's own latent, unchanged: the simplicial-normalised output of
the existing encoder. No second encoder, no extra branch, no new modality. The
adapter is the only thing between it and the action expert.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

from ..action_space import ActionConverter, converter_for
from ..latent_actor import LatentActor
from ..params import report as parameter_report, split as split_report
from ..vendor import TDMPC2 as VENDOR
from .policy import SITES, LatentPolicy


def build_world_model(cfg):
    """Upstream's ``WorldModel``, from upstream's code."""
    with VENDOR.active():
        from common.world_model import WorldModel

        return WorldModel(cfg)


def build_agent(cfg):
    """Upstream's ``TDMPC2``. Constructs its own ``WorldModel`` internally."""
    with VENDOR.active():
        from tdmpc2 import TDMPC2 as Agent

        return Agent(cfg)


def latent_dim(agent) -> int:
    """The width the adapter has to read, from the model that was built."""
    return int(agent.cfg.true_latent_dim)


def build_actor(smolvla: Mapping[str, Any], *, feature_dim: int,
                action_dim: int, device="cuda") -> LatentActor:
    """The pretrained SmolVLA plus a fresh adapter sized for this latent."""
    from ...models.latent_adapter import LatentAdapter
    from ...models.pretrained import load_policy, model_facts
    from ...models.smolvla_actor import SmolVLAActor

    loaded = load_policy(str(smolvla["pretrained"]),
                         str(smolvla.get("revision") or "main"))
    # Placed before SmolVLAActor is constructed: it reads its device from the
    # pretrained weights and moves the adapter to match.
    loaded.policy.to(device)

    facts = model_facts(loaded)
    mode = str(smolvla.get("state_token_mode") or "embedding")
    token_dim = int(facts["vlm_hidden_size"] if mode == "embedding"
                    else facts["max_state_dim"])
    adapter_cfg = dict(smolvla.get("adapter") or {})
    adapter = LatentAdapter(int(feature_dim), token_dim,
                            hidden=int(adapter_cfg.get("hidden", 1024)),
                            layers=int(adapter_cfg.get("layers", 2)))
    inner = SmolVLAActor(
        loaded, adapter, action_dim=int(action_dim),
        chunk_size=int(smolvla.get("chunk_size") or 0) or None,
        flow_steps=int(smolvla.get("flow_steps") or 0) or None,
        instruction=str(smolvla.get("instruction") or ""),
        state_token_mode=mode)

    execute = int(smolvla.get("execute") or 1)
    if not 1 <= execute <= int(inner.chunk_size):
        raise SystemExit(
            f"smolvla.execute={execute} must be at least 1 and at most the "
            f"chunk size {inner.chunk_size}. The chunk length, the number of "
            "executed actions and the planning horizon are three different "
            "quantities.")
    return LatentActor(inner, instruction=str(smolvla.get("instruction") or ""))


def build_converter(metadata: Optional[Mapping[str, Any]], *, action_dim: int,
                    smolvla: Mapping[str, Any], normalizer=None,
                    device=None) -> ActionConverter:
    return converter_for(metadata, action_dim=int(action_dim),
                         mode=str(smolvla.get("action_normalization") or "identity"),
                         normalizer=normalizer, device=device)


def build_policy(actor: LatentActor, converter: ActionConverter,
                 smolvla: Mapping[str, Any], *,
                 descriptor: Optional[Dict[str, Any]] = None) -> LatentPolicy:
    sites = tuple(smolvla.get("sites") or SITES)
    return LatentPolicy(
        actor, converter, sites=sites,
        lr=float(smolvla.get("online_lr", 1e-5)),
        max_batch=int(smolvla.get("max_batch", 256)),
        proposal_flow_steps=int(smolvla.get("proposal_flow_steps") or 0) or None,
        descriptor=descriptor)


def apply_planner_overrides(cfg, smolvla: Mapping[str, Any]) -> Dict[str, Any]:
    """The one planning setting a run is allowed to move, and it is reported.

    ``num_pi_trajs`` is TD-MPC2's, not this integration's. Lowering it is a
    cost decision for the flow sampler -- each proposal is a prefix pass plus
    ``flow_steps`` denoising passes through a 450M model -- and it changes the
    planner, so it comes back as a deviation rather than being applied quietly.
    """
    deviations: Dict[str, Any] = {}
    wanted = int(smolvla.get("pi_trajs") or 0)
    if wanted and wanted != int(cfg.num_pi_trajs):
        deviations["num_pi_trajs"] = {"native": int(cfg.num_pi_trajs),
                                      "run": wanted,
                                      "why": "flow-sampler cost per planning step"}
        cfg.num_pi_trajs = wanted
    return deviations


def parameters(agent, *, actor: Optional[LatentActor] = None) -> Dict[str, Any]:
    """Every component, counted once, with SmolVLA outside the budget.

    ``world_model(rest)`` catches anything the named components missed --
    a task embedding in a multi-task run, or a module a future upstream adds
    -- so the distinct total is the model's, not the sum of what was listed.
    """
    model = agent.model
    components = [
        ("encoder", model._encoder),
        ("dynamics", model._dynamics),
        ("reward", model._reward),
        ("policy_prior(gaussian)", model._pi),
        ("critic(Q ensemble)", model._Qs),
        ("critic_target", model._target_Qs),
        ("world_model(rest)", model),
    ]
    if actor is not None:
        # The adapter is listed before the actor that owns it, so its
        # parameters are attributed to it and counted once. It stays *inside*
        # the budget: it is new world-model-side capacity, not SmolVLA.
        components.append(("adapter", actor.adapter))
        components.append(("smolvla", actor))
    full = parameter_report(components)
    return split_report(full, exclude=("smolvla",) if actor is not None else ())
