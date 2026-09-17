"""Implicit Q-learning on world-model latents, in the reference update order.

The objectives of IQL (Kostrikov, Nair and Levine, 2022), applied each step in
the order of the reference implementation:

1. **value**, by expectile regression onto the target critic's minimum, on
   recorded transitions: ``L_V = E[|tau - 1(delta < 0)| delta^2]`` with
   ``delta = min_i Qbar_i(z, a) - V(z)``;
2. the value is **recomputed with the updated value network**;
3. **actor**, by advantage-weighted likelihood of the recorded action, using
   that updated value: ``L_pi = -E[min(w_max, exp(beta A)) log pi(a | z)]``
   with ``A = min_i Qbar_i(z, a) - V(z)``. No other term: no behaviour-cloning
   loss is added and the actor is not initialised from one;
4. **critics**, by regression onto ``r + gamma c V(z')`` with the updated
   value, where ``c`` is one minus termination for recorded transitions and
   the predicted continuation for imagined ones;
5. the **target critic** follows the critic by Polyak averaging.

``batch`` holds recorded transitions and is the only input of steps 1-3.
``critic_batch`` may add imagined transitions; only step 4 reads it. That is
model-assisted IQL -- an extension of the algorithm, not the unchanged one.

Optional progress branch. A second critic and value learn the discounted
potential-difference reward ``F = gamma c Phi(z') - Phi(z)`` with the same
objectives and the same order, and the actor weights actions by
``A_env + beta A_progress``, both from updated values. Expectile regression is
non-linear, so this is *not* IQL on the summed reward ``r + beta F``; it is an
adaptation of the progress-aware method to IQL and is reported as such.

Initialisation is seeded per branch, independently of the global random
state: the actor, critic and value start from the same weights with and
without the progress branch, and fitting anything else first cannot move them.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.distributions import Independent, Normal

PROGRESS_SEED_OFFSET = 1_000_003


class MLP(nn.Module):
    def __init__(self, inp: int, out: int, hidden: Sequence[int], layernorm: bool = True, activation: str = "SiLU"):
        super().__init__()
        layers = []
        width = inp
        for size in hidden:
            layers.append(nn.Linear(width, int(size)))
            if layernorm:
                layers.append(nn.LayerNorm(int(size)))
            layers.append(getattr(nn, activation)())
            width = int(size)
        layers.append(nn.Linear(width, out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwinQ(nn.Module):
    def __init__(self, z_dim: int, a_dim: int, net: Mapping[str, Any]):
        super().__init__()
        self.q1 = MLP(z_dim + a_dim, 1, net["hidden"], net["layernorm"], net["activation"])
        self.q2 = MLP(z_dim + a_dim, 1, net["hidden"], net["layernorm"], net["activation"])

    def forward(self, z: torch.Tensor, a: torch.Tensor):
        x = torch.cat([z, a], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class ValueNet(nn.Module):
    def __init__(self, z_dim: int, net: Mapping[str, Any]):
        super().__init__()
        self.v = MLP(z_dim, 1, net["hidden"], net["layernorm"], net["activation"])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.v(z).squeeze(-1)


class GaussianPolicy(nn.Module):
    """Tanh-bounded mean and a learned state-independent log standard deviation.

    Executable commands are normalised into [-1, 1], so the mean lives there;
    likelihoods are evaluated on the recorded actions directly.
    """

    def __init__(self, z_dim: int, a_dim: int, net: Mapping[str, Any]):
        super().__init__()
        self.mean = MLP(z_dim, a_dim, net["hidden"], net["layernorm"], net["activation"])
        self.log_std = nn.Parameter(torch.zeros(a_dim))
        self.log_std_min = float(net["log_std_min"])
        self.log_std_max = float(net["log_std_max"])

    def distribution(self, z: torch.Tensor) -> Independent:
        mean = torch.tanh(self.mean(z))
        std = self.log_std.clamp(self.log_std_min, self.log_std_max).exp().expand_as(mean)
        return Independent(Normal(mean, std), 1)

    def log_prob(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.distribution(z).log_prob(a)

    @torch.no_grad()
    def act(self, z: torch.Tensor, deterministic: bool = True, noise_std: float = 0.0,
            generator: Optional[torch.Generator] = None) -> torch.Tensor:
        dist = self.distribution(z)
        loc, scale = dist.base_dist.loc, dist.base_dist.scale
        if deterministic:
            action = loc
        else:
            action = loc + scale * torch.randn(loc.shape, device=loc.device, dtype=loc.dtype, generator=generator)
        if noise_std > 0:
            action = action + noise_std * torch.randn(action.shape, device=action.device, dtype=action.dtype,
                                                      generator=generator)
        return action.clamp(-1.0, 1.0)


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.abs(expectile - (diff < 0).float())
    return (weight * diff.square()).mean()


def _soft_update(target: nn.Module, source: nn.Module, rate: float) -> None:
    with torch.no_grad():
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.mul_(1.0 - rate).add_(s.data, alpha=rate)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> Optional[float]:
    count = float(mask.sum())
    return float((values * mask).sum() / count) if count > 0 else None


class IQL(nn.Module):
    def __init__(self, z_dim: int, a_dim: int, cfg: Mapping[str, Any], total_steps: int, progress_beta: float = 0.0,
                 init_seed: int = 0):
        super().__init__()
        net = cfg["network"]
        self.cfg = dict(cfg)
        self.gamma = float(cfg["gamma"])
        self.expectile = float(cfg["expectile"])
        self.beta = float(cfg["beta"])
        self.max_weight = float(cfg["max_weight"])
        self.polyak = float(cfg["polyak"])
        self.progress_beta = float(progress_beta)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(init_seed))
            self.critic = TwinQ(z_dim, a_dim, net)
            self.value = ValueNet(z_dim, net)
            self.actor = GaussianPolicy(z_dim, a_dim, net)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        lr = float(cfg["lr"])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.actor_schedule = (torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_opt, T_max=max(1, int(total_steps)))
                               if bool(cfg.get("actor_cosine_decay", True)) else None)

        self.with_progress = self.progress_beta > 0.0
        if self.with_progress:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(init_seed) + PROGRESS_SEED_OFFSET)
                self.progress_critic = TwinQ(z_dim, a_dim, net)
                self.progress_value = ValueNet(z_dim, net)
            self.target_progress_critic = copy.deepcopy(self.progress_critic).requires_grad_(False)
            self.progress_critic_opt = torch.optim.Adam(self.progress_critic.parameters(), lr=lr)
            self.progress_value_opt = torch.optim.Adam(self.progress_value.parameters(), lr=lr)

    # ------------------------------------------------------------ steps
    @staticmethod
    def _step(optimizer, loss):
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    def update(self, batch: Mapping[str, torch.Tensor], critic_batch: Optional[Mapping[str, torch.Tensor]] = None
               ) -> Dict[str, float]:
        """One IQL step. ``batch``: recorded transitions. ``critic_batch``: what the critics learn from."""
        critic_batch = batch if critic_batch is None else critic_batch
        z, a = batch["z"], batch["action"]
        metrics: Dict[str, float] = {}

        # 1. value networks, on recorded transitions, against the target critics
        with torch.no_grad():
            q_target = torch.min(*self.target_critic(z, a))
        value_loss = expectile_loss(q_target - self.value(z), self.expectile)
        self._step(self.value_opt, value_loss)
        if self.with_progress:
            mask = batch["progress_valid"]
            with torch.no_grad():
                pq_target = torch.min(*self.target_progress_critic(z, a))
            p_diff = (pq_target - self.progress_value(z)) * mask
            progress_value_loss = expectile_loss(p_diff, self.expectile) * (mask.numel() / mask.sum().clamp_min(1))
            self._step(self.progress_value_opt, progress_value_loss)

        # 2. values recomputed with the updated networks
        with torch.no_grad():
            v = self.value(z)
            env_advantage = q_target - v
            advantage = env_advantage
            if self.with_progress:
                progress_advantage = (pq_target - self.progress_value(z)) * mask
                advantage = env_advantage + self.progress_beta * progress_advantage
            weight = torch.exp(self.beta * advantage).clamp(max=self.max_weight)

        # 3. actor: advantage-weighted likelihood of the recorded actions, nothing else
        log_prob = self.actor.log_prob(z, a)
        actor_loss = -(weight * log_prob).mean()
        self._step(self.actor_opt, actor_loss)
        if self.actor_schedule is not None:
            self.actor_schedule.step()

        # 4. critics, possibly with imagined transitions, against the updated values
        zc, ac = critic_batch["z"], critic_batch["action"]
        with torch.no_grad():
            target = critic_batch["reward"] + self.gamma * critic_batch["cont"] * self.value(critic_batch["z_next"])
        q1, q2 = self.critic(zc, ac)
        critic_loss = ((q1 - target).square() + (q2 - target).square()).mean()
        self._step(self.critic_opt, critic_loss)
        if self.with_progress:
            pmask = critic_batch["progress_valid"]
            with torch.no_grad():
                p_target = (critic_batch["progress_reward"]
                            + self.gamma * critic_batch["cont"] * self.progress_value(critic_batch["z_next"]))
            p1, p2 = self.progress_critic(zc, ac)
            p_loss = (((p1 - p_target).square() + (p2 - p_target).square()) * pmask).sum() / pmask.sum().clamp_min(1)
            self._step(self.progress_critic_opt, p_loss)

        # 5. target critics
        _soft_update(self.target_critic, self.critic, self.polyak)
        if self.with_progress:
            _soft_update(self.target_progress_critic, self.progress_critic, self.polyak)

        with torch.no_grad():
            td = 0.5 * ((q1 - target).abs() + (q2 - target).abs())
            synthetic = critic_batch.get("synthetic")
            synthetic = torch.zeros_like(td) if synthetic is None else synthetic.float()
            metrics.update({
                "loss/value": float(value_loss),
                "loss/actor": float(actor_loss),
                "loss/critic": float(critic_loss),
                "q/mean": float(torch.min(q1, q2).mean()),
                "v/mean": float(v.mean()),
                "v/std": float(v.std()),
                "adv/mean": float(advantage.mean()),
                "adv/std": float(advantage.std()),
                "adv/env_mean": float(env_advantage.mean()),
                "weight/mean": float(weight.mean()),
                "weight/max": float(weight.max()),
                "weight/clipped_fraction": float((weight >= self.max_weight).float().mean()),
                "weight/effective_sample_fraction": float(weight.sum().square()
                                                          / (weight.numel() * weight.square().sum()).clamp_min(1e-12)),
                "actor/log_std": float(self.actor.log_std.mean()),
                "critic/synthetic_fraction": float(synthetic.mean()),
            })
            recorded_td = _masked_mean(td, 1.0 - synthetic)
            imagined_td = _masked_mean(td, synthetic)
            if recorded_td is not None:
                metrics["critic/td_abs_recorded"] = recorded_td
            if imagined_td is not None:
                metrics["critic/td_abs_synthetic"] = imagined_td
            if self.with_progress:
                metrics["progress/value_loss"] = float(progress_value_loss)
                metrics["progress/critic_loss"] = float(p_loss)
                metrics["progress/advantage_abs"] = float(progress_advantage.abs().mean())
                metrics["progress/influence_raw"] = float(self.progress_beta * progress_advantage.abs().mean()
                                                          / (env_advantage.abs().mean() + 1e-8))
        return metrics

    # ------------------------------------------------------ diagnostics
    @torch.no_grad()
    def diagnostics(self, recorded: Iterable[Mapping[str, torch.Tensor]],
                    imagined: Iterable[Mapping[str, torch.Tensor]] = ()) -> Dict[str, float]:
        """Action error, TD error, value statistics and advantage weights on fixed batches.

        Recorded and imagined critic errors are reported separately. Nothing
        here selects a checkpoint: action error on recorded episodes measures
        imitation, not how well a policy performs.
        """
        sums: Dict[str, float] = {}
        rows = 0
        per_dim = None

        def add(key: str, value: torch.Tensor) -> None:
            sums[key] = sums.get(key, 0.0) + float(value.sum())

        for batch in recorded:
            z, a = batch["z"], batch["action"]
            n = int(z.shape[0])
            mean = self.actor.act(z, deterministic=True)
            error = (mean - a)
            q1, q2 = self.critic(z, a)
            q = torch.min(q1, q2)
            v = self.value(z)
            target = batch["reward"] + self.gamma * batch["cont"] * self.value(batch["z_next"])
            advantage = torch.min(*self.target_critic(z, a)) - v
            weight = torch.exp(self.beta * advantage).clamp(max=self.max_weight)
            add("action_mse", error.square().mean(-1))
            add("action_log_prob", self.actor.log_prob(z, a))
            add("td_abs_recorded", 0.5 * ((q1 - target).abs() + (q2 - target).abs()))
            add("q_mean", q)
            add("v_mean", v)
            add("v_sq", v.square())
            add("advantage_mean", advantage)
            add("advantage_sq", advantage.square())
            add("weight_mean", weight)
            add("weight_sq", weight.square())
            add("weight_clipped_fraction", (weight >= self.max_weight).float())
            per_dim = error.abs().sum(0) if per_dim is None else per_dim + error.abs().sum(0)
            rows += n
        out: Dict[str, float] = {}
        if rows:
            out = {key: value / rows for key, value in sums.items() if not key.endswith("_sq")}
            out["v_std"] = max(sums["v_sq"] / rows - out["v_mean"] ** 2, 0.0) ** 0.5
            out["advantage_std"] = max(sums["advantage_sq"] / rows - out["advantage_mean"] ** 2, 0.0) ** 0.5
            out["weight_effective_sample_fraction"] = (sums["weight_mean"] ** 2) / max(rows * sums["weight_sq"], 1e-12)
            for index, value in enumerate((per_dim / rows).tolist()):
                out[f"action_abs/{index}"] = float(value)
            out["recorded_rows"] = float(rows)

        imagined_rows, imagined_sums = 0, {"td_abs_synthetic": 0.0, "q_mean_synthetic": 0.0,
                                           "predicted_reward_mean": 0.0, "predicted_cont_mean": 0.0}
        for batch in imagined:
            z, a = batch["z"], batch["action"]
            q1, q2 = self.critic(z, a)
            target = batch["reward"] + self.gamma * batch["cont"] * self.value(batch["z_next"])
            imagined_sums["td_abs_synthetic"] += float((0.5 * ((q1 - target).abs() + (q2 - target).abs())).sum())
            imagined_sums["q_mean_synthetic"] += float(torch.min(q1, q2).sum())
            imagined_sums["predicted_reward_mean"] += float(batch["reward"].sum())
            imagined_sums["predicted_cont_mean"] += float(batch["cont"].sum())
            imagined_rows += int(z.shape[0])
        if imagined_rows:
            out.update({key: value / imagined_rows for key, value in imagined_sums.items()})
            out["synthetic_rows"] = float(imagined_rows)
        return out

    # ------------------------------------------------------ persistence
    def optimizer_state(self) -> Dict[str, Any]:
        state = {"critic": self.critic_opt.state_dict(), "value": self.value_opt.state_dict(),
                 "actor": self.actor_opt.state_dict()}
        if self.actor_schedule is not None:
            state["actor_schedule"] = self.actor_schedule.state_dict()
        if self.with_progress:
            state["progress_critic"] = self.progress_critic_opt.state_dict()
            state["progress_value"] = self.progress_value_opt.state_dict()
        return state

    def load_optimizer_state(self, state: Mapping[str, Any]) -> None:
        self.critic_opt.load_state_dict(state["critic"])
        self.value_opt.load_state_dict(state["value"])
        self.actor_opt.load_state_dict(state["actor"])
        if self.actor_schedule is not None and "actor_schedule" in state:
            self.actor_schedule.load_state_dict(state["actor_schedule"])
        if self.with_progress:
            self.progress_critic_opt.load_state_dict(state["progress_critic"])
            self.progress_value_opt.load_state_dict(state["progress_value"])


def potential_difference(phi: torch.Tensor, phi_next: torch.Tensor, cont: torch.Tensor, gamma: float) -> torch.Tensor:
    """``F = gamma c Phi(z') - Phi(z)``: the definition ``progress.potential_shaping`` uses, per transition."""
    return gamma * cont * phi_next - phi
