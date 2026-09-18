"""SmolVLA as TD-MPC2's learned policy, at the five places it is asked for one.

TD-MPC2 asks ``model.pi`` for an action in five places, and they are not the
same kind of question:

=================  ==========================================================
``act``            the action to execute, when ``mpc=false``. With MPC on --
                   the default, and what Stage 3 keeps -- the planner is the
                   final action selector and this site is not reached.
``plan_proposals`` the ``num_pi_trajs`` policy trajectories that seed MPPI.
                   One proposal per latent, then the latent is advanced by it
                   and the next proposal comes from the *resulting* latent.
``estimate_value`` the terminal bootstrap ``Q(z_H, pi(z_H))`` inside the
                   planner's value estimate. Evaluated for every sampled
                   trajectory at every MPPI iteration, so this is by far the
                   most expensive of the five.
``td_target``      the Q-learning bootstrap ``r + gamma * Q_target(z', pi(z'))``.
``update_pi``      policy optimization: maximise the scaled Q of the policy's
                   own action.
=================  ==========================================================

All five are served by default. ``sites`` exists because the cost of the
planner sites is not a detail: with the shipped planning settings,
``estimate_value`` alone asks for ``num_envs * num_samples * iterations`` flow
samples *per environment step*, which for 32 envs is about 10^5. Turning a
site off falls back to the Gaussian prior there, which is a real deviation from
"SmolVLA is the policy" and is recorded in the checkpoint, reported at startup
and listed in the run's deviations -- and it makes the Gaussian prior a live
component again, so ``needs_gaussian`` turns its training back on.

What this class does not do
---------------------------

It does not implement ``model.pi``'s return signature. That returns
``(mu, pi, log_pi, log_std)``; a flow policy has the first two at best and
neither of the last two. :meth:`sample` returns an action, and the one
objective that read ``log_pi`` -- the entropy bonus in ``update_pi`` -- is
dropped in SmolVLA mode rather than handed a stand-in.

Coordinates
-----------

The actor samples in its own coordinates and TD-MPC2 consumes the
controller's. The conversion happens once, in :meth:`sample`, through the same
:class:`~sim_vla.integrations.action_space.ActionConverter` the imitation stage
normalizes its targets with -- so the planner, the Q functions, the dynamics
and the environment all see native units and the flow model only ever sees its
own. Clipping to the controller's bounds happens before the conversion and is
straight-through, so a saturated dimension still carries gradient in
``update_pi``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from ..latent_actor import LatentActor

SITES = ("act", "plan_proposals", "estimate_value", "td_target", "update_pi")


class LatentPolicy(nn.Module):
    """A latent-conditioned action-chunk policy behind TD-MPC2's hooks."""

    def __init__(self, actor: LatentActor, converter, *,
                 sites: Sequence[str] = SITES,
                 lr: float = 1e-5,
                 max_batch: int = 256,
                 proposal_flow_steps: Optional[int] = None,
                 deterministic_seed: int = 0,
                 descriptor: Optional[Dict[str, Any]] = None):
        super().__init__()
        unknown = [s for s in sites if s not in SITES]
        if unknown:
            raise ValueError(f"unknown policy site(s) {unknown}; known: {list(SITES)}")
        self.actor = actor
        self.converter = converter
        self.sites = tuple(sites)
        self.max_batch = int(max_batch)
        self.proposal_flow_steps = (int(proposal_flow_steps)
                                    if proposal_flow_steps else None)
        self._deterministic_seed = int(deterministic_seed)
        self._meta = dict(descriptor or {})

        trainable = self.actor.trainable_parameters()
        if not trainable:
            raise RuntimeError(
                "nothing in the actor is trainable; the adapter and the action "
                "expert are supposed to be")
        self.optimizer = torch.optim.AdamW(trainable, lr=float(lr))
        self.lr = float(lr)
        self.calls = {site: 0 for site in SITES}
        self.rows = {site: 0 for site in SITES}

    # ------------------------------------------------------------------ sites
    def serves(self, site: str) -> bool:
        return str(site) in self.sites

    @property
    def needs_gaussian(self) -> bool:
        """Whether TD-MPC2's Gaussian prior is still a live component.

        True when any site fell back to it. Then it has to keep training, or
        that site bootstraps off a policy frozen where pretraining left it.
        False when SmolVLA serves all five, and the prior is dormant: no
        gradient, no role in any target, retained only so the world model's
        state dict keeps its native shape and so `smolvla.enabled: false`
        restores the upstream agent exactly.
        """
        return any(site not in self.sites for site in SITES)

    @property
    def dormant_gaussian(self) -> bool:
        return not self.needs_gaussian

    # --------------------------------------------------------------- sampling
    def _generator(self, device) -> Optional[torch.Generator]:
        if self._deterministic_seed is None:
            return None
        generator = torch.Generator(device=device)
        generator.manual_seed(int(self._deterministic_seed))
        return generator

    def _sample_flat(self, features: torch.Tensor, *, grad: bool,
                     deterministic: bool, steps: Optional[int]) -> torch.Tensor:
        """``(N, latent) -> (N, action_dim)``, in the actor's coordinates.

        The chunk is sampled at this latent and its **first** action is the
        proposal. The rest of the chunk is discarded here on purpose: the next
        proposal belongs to the latent this action leads to, and reusing chunk
        entries as a trajectory would silently equate the chunk length with the
        planning horizon.
        """
        generator = self._generator(features.device) if deterministic else None
        chunks: List[torch.Tensor] = []
        size = max(int(self.max_batch), 1)
        for start in range(0, int(features.shape[0]), size):
            piece = features[start:start + size]
            chunk = self.actor.sample_chunk(
                piece, differentiable=grad, steps=steps, generator=generator)
            chunks.append(chunk[:, 0])
        return torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]

    def sample(self, z: torch.Tensor, task=None, *, deterministic: bool = False,
               grad: bool = False, site: Optional[str] = None) -> torch.Tensor:
        """An action per latent, in TD-MPC2's own coordinates.

        ``z`` may carry any number of leading axes -- ``(B, D)`` for one
        observation per env, ``(envs, trajs, D)`` inside the planner,
        ``(horizon + 1, B, D)`` in the policy update -- and the result has the
        same leading axes with ``action_dim`` last.

        ``deterministic`` does not mean "the mode": a flow policy has no
        closed-form mode. It means the sampler's starting noise is a fixed
        draw, which makes the action a deterministic, reproducible function of
        the latent. That is what an evaluation rollout needs and it is not the
        distribution's mean; the difference is stated rather than glossed.
        """
        if z.dim() < 2:
            raise ValueError(
                f"expected a latent with at least (batch, dim), got {tuple(z.shape)}")
        lead = z.shape[:-1]
        flat = z.reshape(-1, z.shape[-1])
        steps = (self.proposal_flow_steps
                 if site in ("plan_proposals", "estimate_value") else None)
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            actor_action = self._sample_flat(
                flat, grad=grad, deterministic=deterministic, steps=steps)
            native = self.converter.to_env(actor_action)
        if site is not None:
            self.calls[site] += 1
            self.rows[site] += int(flat.shape[0])
        return native.reshape(*lead, native.shape[-1])

    # ------------------------------------------------------------- optimizing
    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def step(self, grad_clip_norm: float) -> float:
        clipped = torch.nn.utils.clip_grad_norm_(
            self.actor.trainable_parameters(), float(grad_clip_norm))
        self.optimizer.step()
        return float(clipped)

    def parameters(self, recurse: bool = True):
        return super().parameters(recurse)

    # ------------------------------------------------------------ persistence
    def state_dict(self, *args, **kwargs):                 # type: ignore[override]
        return {"actor": self.actor.state_dict(*args, **kwargs),
                "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state, meta: Optional[Mapping] = None,  # type: ignore[override]
                        strict: bool = True):
        if meta is not None:
            self._check_meta(dict(meta))
        self.actor.load_state_dict(state["actor"], strict=strict)
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        return self

    def _check_meta(self, stored: Dict[str, Any]) -> None:
        current = self.descriptor()
        for key in ("action_normalization", "chunk_size", "action_dim",
                    "revision", "state_token_mode"):
            left, right = stored.get(key), current.get(key)
            if left in (None, "", 0) or right in (None, "", 0):
                continue
            if left != right:
                raise SystemExit(
                    f"refusing to load this policy: {key} is {left!r} in the "
                    f"checkpoint and {right!r} in this run.")

    # ----------------------------------------------------------------- report
    def descriptor(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "sites": list(self.sites),
            "gaussian_after_handoff": ("trained (a site still falls back to it)"
                                       if self.needs_gaussian else
                                       "dormant (retained for checkpoint shape "
                                       "and for smolvla.enabled=false)"),
            "chunk_size": self.actor.chunk_size,
            "action_dim": self.actor.action_dim,
            "flow_steps": self.actor.flow_steps,
            "proposal_flow_steps": self.proposal_flow_steps,
            "max_batch": self.max_batch,
            "online_lr": self.lr,
            "log_prob": "unavailable (flow policy); entropy term disabled",
        }
        out.update(self._meta)
        out["action_normalization"] = self.converter.mode
        return out

    def usage(self) -> Dict[str, Any]:
        return {"calls": dict(self.calls), "rows": dict(self.rows)}


def planner_cost(cfg, *, sites: Sequence[str], flow_steps: int,
                 proposal_flow_steps: Optional[int] = None) -> Dict[str, Any]:
    """How many flow-sampler rows one environment step will ask for.

    Printed at startup. The planner sites dominate by orders of magnitude and
    a run that does not know that before it starts finds out six hours later.
    """
    horizon = int(cfg.horizon)
    envs = int(cfg.num_envs)
    proposals = int(cfg.num_pi_trajs)
    samples = int(cfg.num_samples)
    iterations = int(cfg.iterations)
    batch = int(cfg.batch_size)
    per_update = max(1, int(envs / max(int(cfg.steps_per_update), 1)))
    steps = int(proposal_flow_steps or flow_steps)

    rows = {
        "plan_proposals": envs * proposals * horizon if "plan_proposals" in sites else 0,
        "estimate_value": envs * samples * iterations if "estimate_value" in sites else 0,
        "td_target": horizon * batch * per_update if "td_target" in sites else 0,
        "update_pi": (horizon + 1) * batch * per_update if "update_pi" in sites else 0,
    }
    denoise = {k: v * (steps if k in ("plan_proposals", "estimate_value")
                       else flow_steps) for k, v in rows.items()}
    return {"rows_per_env_step": rows,
            "denoise_passes_per_env_step": denoise,
            "total_rows_per_env_step": sum(rows.values()),
            "total_denoise_passes_per_env_step": sum(denoise.values()),
            "flow_steps": flow_steps,
            "proposal_flow_steps": steps}
