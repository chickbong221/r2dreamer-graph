"""Stage 2: collect, update the world model, imagine, train critic and actor.

The loop is identical for both arms. The only primary difference is whether the
world model's inference and dynamics carry the graph branch, which is decided
when the arm's world model is constructed and not here.

The world model and the actor arrive as objects from Stages 1A and 1B in the
same process -- nothing is reloaded from disk, because by default nothing was
written there. ``OnlineConfig.save_checkpoints`` turns writing on for a run
long enough that losing it would matter; it is off by default, and with it off
:meth:`OnlineTrainer.checkpoint` is a no-op rather than a silent partial write.

Three things this is careful about.

Posterior states are recomputed every update from the current world model. A
Stage 1 latent cache is a function of the model that produced it, and the world
model is being trained here, so reusing that cache would condition the policy
on states the model no longer produces.

Demonstrations and online experience are mixed by sequence count, and the
mixture waits for the replay to hold enough episodes to be worth sampling --
four online rollouts treated as half the distribution is worse than training on
demonstrations alone for another few minutes.

Acting is recurrent. :class:`LatentPolicy` advances the posterior on every
observation and re-plans a chunk every ``actor.execute`` steps, and its state is
reset per episode. A policy that carried the previous episode's state into the
next one would still produce plausible rollouts, and they would not mean
anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from ..data.batch import to_model_batch

from ..data.replay import OnlineEpisode, OnlineReplay, mixed_batch
from ..runtime.checkpoint import CheckpointMeta, save
from .actor_critic import ActorCriticConfig, ActorCriticTrainer
from .imagination import start_states


@dataclass
class OnlineConfig:
    total_steps: int = 1_000_000
    episodes_per_collect: int = 2
    updates_per_collect: int = 8
    demo_fraction: float = 0.5
    batch_size: int = 16
    sequence_length: int = 64
    burn_in: int = 8
    imagination_batch: int = 256
    max_episode_steps: int = 150
    seed: int = 0
    actor_every: int = 1
    # Action-only lookahead for replay windows, matching the demonstration
    # sampler's. Set from the actor's chunk size by run_online.
    lookahead: int = 0
    # One switch for the dataset, the replay, the env and imagination. A
    # replay that honoured terminations while the loader ignored them would
    # train one continuation head on two conventions.
    ignore_terminations: bool = True
    # Off by default. The stages hand their models on in memory, so a
    # checkpoint here is for resuming a long run, not for reaching the next
    # stage. checkpoint_every is only consulted when this is true.
    save_checkpoints: bool = False
    checkpoint_every: int = 10_000
    eval_every: int = 0             # 0 disables periodic evaluation
    eval_episodes: int = 10


def collect_episode(env, policy, *, max_steps: int = 150,
                    seed: Optional[int] = None) -> OnlineEpisode:
    """One rollout, keeping the final observation before the reset.

    ``policy`` takes an observation dict and returns an action. The final
    observation is appended after the loop: a buffer whose observation count
    equals its action count has lost the state the last action led to, and the
    replay refuses such an episode rather than storing it.
    """
    episode = OnlineEpisode()
    obs = env.reset(seed)
    episode.add_observation(obs)
    for _ in range(int(max_steps)):
        action = policy(obs)
        out = env.step(action)
        episode.add_transition(action, out["reward"], out["is_terminal"],
                               out["is_last"], out["success"])
        episode.add_observation(out["obs"])
        obs = out["obs"]
        if out["is_last"]:
            break
    return episode


def episode_metrics(episodes) -> Dict[str, float]:
    """The original pipeline's ``episode/`` panel, from the same rollouts.

    ``trainer.py`` logs ``episode/score`` and ``episode/length``, plus one
    entry per ``log_`` observation key taken as the per-episode *maximum* --
    a 0/1 flag maxed over an episode is "it happened at least once". The
    simulator's env supplies those keys through ManiSkill's ``final_info``;
    ``SimVlaEnv`` has no such channel, but ``collect_episode`` already records
    the per-step success flag, so the same numbers are derivable here and cost
    no extra simulation.

    ``success_once`` and ``success_at_end`` are kept apart deliberately: the
    flag flickers -- PickCube wants a static robot as well as a placed cube --
    so reaching success and still holding it at the end are different events,
    and neither is recoverable from the other.
    """
    if not episodes:
        return {}
    score, length, once, at_end = [], [], [], []
    for episode in episodes:
        flags = np.asarray(episode.success, dtype=bool)
        score.append(float(np.sum(episode.reward)))
        length.append(float(episode.steps))
        once.append(float(flags.any()) if flags.size else 0.0)
        at_end.append(float(flags[-1]) if flags.size else 0.0)
    return {
        "episode/score": float(np.mean(score)),
        "episode/length": float(np.mean(length)),
        "episode/success_once": float(np.mean(once)),
        "episode/success_at_end": float(np.mean(at_end)),
    }


class LatentPolicy:
    """Recurrent inference: one posterior state per episode, carried forward.

    Two things here are not conveniences.

    **Every observation is encoded**, even on steps where no new chunk is
    planned. Encoding only at re-plan boundaries would skip observations the
    recurrent state was trained to consume, so the state driving step ``k``
    would be one the model has never been in.

    **The state is reset per episode.** :meth:`reset` must be called before
    each rollout; otherwise the first action of an episode is driven by the
    state the previous one ended in.

    Actions cross the normalization boundary twice and in opposite directions.
    The policy was trained on normalized actions, so what it emits is
    denormalized before the environment sees it, and the raw action is what is
    fed back as the posterior's ``a_(t-1)`` -- where :func:`to_model_batch`
    normalizes it again, exactly as it does in training.
    """

    def __init__(self, world_model, actor, *, device="cuda", normalizer=None,
                 coords=None, instruction: Optional[str] = None,
                 execute: int = 1, flow_steps: Optional[int] = None):
        self.world_model = world_model
        self.actor = actor
        self.device = torch.device(device)
        self.normalizer = normalizer
        self.coords = coords
        self.instruction = instruction
        chunk = int(getattr(actor, "chunk_size", 1))
        execute = int(execute)
        if not 1 <= execute <= chunk:
            raise ValueError(
                f"execute={execute} must be at least 1 and at most the chunk "
                f"size {chunk}: executing more actions than the policy "
                "predicts would repeat or invent commands.")
        if execute != 1:
            # imagination.imagine replans at every transition. At execute=1
            # that is exactly what happens online; at anything larger the
            # imagined rollout models a policy that does not exist, and the
            # actor would be optimised for it. Refused rather than
            # approximated.
            raise NotImplementedError(
                f"execute={execute} is not supported for online training: "
                "sim_vla.training.imagination replans every transition, so "
                "imagined and executed policies would differ. Use execute=1, "
                "or implement matching imagined execution semantics first.")
        self.execute = execute
        self.chunk_size = chunk
        self.flow_steps = int(flow_steps or actor.flow_steps)
        self.action_dim = int(actor.action_dim)
        # The environment's clipping lives in coords, so this policy returns
        # the command that will actually be run. Declared so a caller does not
        # clip a second time behind its back.
        self.bounded = coords is not None
        self.clipped = 0
        self.reset()

    def reset(self) -> None:
        """Start an episode: no recurrent state, no queued actions."""
        self._state = None
        self._queue: List[np.ndarray] = []
        self._prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self._first = True

    def _window(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """One observation as a length-1 window in the training layout.

        ``_prev_action`` is in raw environment units -- the command that was
        actually sent -- and ``to_model_batch`` standardises it and maps it
        into dynamics coordinates, exactly as it does for a training window.
        """
        batch: Dict[str, Any] = {key: np.asarray(value)[None, None]
                                 for key, value in obs.items()}
        # Storage naming, so to_model_batch renames and normalizes it the way
        # it does for a training window.
        batch["actions"] = self._prev_action[None, None]
        batch["is_first"] = np.array([[self._first]], dtype=bool)
        return to_model_batch(batch, self.device, reward_dim=False,
                              normalizer=self.normalizer, coords=self.coords)

    @torch.no_grad()
    def _encode(self, obs: Dict[str, np.ndarray]) -> torch.Tensor:
        """Advance the posterior by one observation; return its feature."""
        from ..models.world_model import WorldModel

        out = self.world_model.observe(self._window(obs), self._state)
        post = out["post"]
        graph = bool(self.world_model.graph_enabled)
        stoch, deter, _logit, sem = WorldModel.unpack(post, graph)
        self._state = ((stoch[:, -1], deter[:, -1], sem[:, -1]) if graph
                       else (stoch[:, -1], deter[:, -1]))
        self._first = False
        return self.world_model.features(post)[:, -1]

    @torch.no_grad()
    def _plan(self, feat: torch.Tensor) -> List[np.ndarray]:
        """Sample one action chunk and return the part that gets executed.

        The clip happens here, in the actor's own coordinates, so that what is
        returned is the command the environment will run and what is fed back
        as ``a_(t-1)`` is that same command. Clipping downstream -- in the
        evaluation loop, say -- left the policy conditioning on an action that
        was never executed.
        """
        from ..models.flow_sampler import sample_actions

        cond = self.actor.condition(feat, self.instruction)
        chunk = sample_actions(
            self.actor.velocity_fn(), cond, batch=1,
            chunk=int(self.actor.chunk_size), dim=self.action_dim,
            steps=self.flow_steps,
            # The actor's device, not the feature's: condition() moved the
            # feature to the pretrained weights and the noise starts there.
            device=getattr(self.actor, "device", feat.device),
            dtype=feat.dtype, differentiable=False)
        normalized = chunk[0, : self.execute]
        if self.coords is not None:
            bounded = self.coords.executed(normalized)
            self.clipped += int(
                (bounded != normalized).any(dim=-1).sum().item())
            raw = self.coords.denormalize(bounded).float().cpu().numpy()
        else:
            raw = normalized.float().cpu().numpy()
        return [np.asarray(row, dtype=np.float32) for row in raw]

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        feat = self._encode(obs)
        if not self._queue:
            self._queue = self._plan(feat)
        action = self._queue.pop(0)
        # Exactly what goes to the environment is exactly what the next
        # posterior consumes.
        self._prev_action = np.asarray(action, dtype=np.float32)
        return action


class OnlineTrainer:
    """World-model updates, imagined actor-critic updates, and checkpoints."""

    def __init__(self, world_model, actor, critic, demo_sampler, *,
                 config: OnlineConfig, ac_config: ActorCriticConfig,
                 world_lr: float = 1e-4, device="cuda",
                 progress_head=None, checkpoint_dir: Optional[Path] = None,
                 meta: Optional[CheckpointMeta] = None, normalizer=None,
                 coords=None, seed: int = 0, potential=None,
                 progress_config=None, progress_lr: float = 3e-4):
        self.world_model = world_model
        self.actor = actor
        self.critic = critic
        self.demo_sampler = demo_sampler
        self.config = config
        self.device = torch.device(device)
        self.progress_head = progress_head
        self.normalizer = normalizer
        self.coords = coords
        # Seeded from the run's seed rather than defaulting to zero
        # independently, so two arms of the same experiment draw the same
        # replay windows.
        self.replay = OnlineReplay(seed=int(seed))
        self.world_opt = torch.optim.AdamW(world_model.parameters(), lr=world_lr)
        # The observed-graph potential, and the head that learns to predict it
        # from a latent so imagination can read it. Separate optimizer, so
        # fitting the head cannot move the world model or the critics.
        self.potential = potential
        self.progress_config = progress_config
        self.progress_opt = (
            torch.optim.AdamW(progress_head.parameters(), lr=progress_lr)
            if progress_head is not None else None)
        self.ac = ActorCriticTrainer(world_model, actor, critic, ac_config,
                                     coords=coords,
                                     progress_head=progress_head)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.meta = meta
        self.env_steps = 0
        self.updates = 0

    def to_torch(self, batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Storage names to model names, in the one place that does it."""
        return to_model_batch(batch, self.device, normalizer=self.normalizer,
                              coords=self.coords)

    def update(self) -> Dict[str, float]:
        """One world-model step, then one actor-critic step on fresh states."""
        batch = self.to_torch(mixed_batch(
            self.demo_sampler, self.replay, self.config.batch_size,
            self.config.sequence_length, self.config.burn_in,
            self.config.demo_fraction, lookahead=int(self.config.lookahead),
            ignore_terminations=bool(self.config.ignore_terminations)))

        total, _losses, _aux = self.world_model.loss(batch)
        self.world_opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
        self.world_opt.step()
        metrics = {"world_loss": float(total.detach())}

        metrics |= self.update_progress(batch)

        # Re-encoded after the step, not reused from before it. The posterior
        # in ``_aux`` came from the parameters that have just been replaced.
        start = start_states(self.world_model, batch,
                             limit=int(self.config.imagination_batch))
        metrics["imagination_starts"] = float(start[0].shape[0])
        self.updates += 1
        if self.updates % max(int(self.config.actor_every), 1) == 0:
            metrics |= self.ac.update(start, progress_beta=self.beta())
        return metrics

    def beta(self) -> Optional[float]:
        """The shaping weight at this point in the run, or None when off."""
        if self.progress_config is None or self.ac.progress_head is None:
            return None
        from .progress import beta_at

        return beta_at(self.progress_config, self.env_steps)

    def update_progress(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Fit the progress head to the observed-graph potential.

        The targets come from the *recorded* graph labels, not from the
        decoder's predictions, so the head is regressed onto something the
        dataset actually contains. Features are detached: this trains the head
        and nothing else, which is why it has its own optimizer.

        Rows whose potential is invalid -- a role that matched no node, a
        relation the frame never observed -- are masked rather than counted as
        zero progress. A schedule role that never resolves would otherwise
        teach the head that the task never advances.
        """
        if self.progress_opt is None or self.potential is None:
            return {}
        head = self.ac.progress_head
        phi, phi_valid = self.potential.targets(batch)
        with torch.no_grad():
            feat = self.world_model.features(
                self.world_model.observe(batch)["post"])
        mask = batch["loss_mask"].bool() & phi_valid
        if not bool(mask.any()):
            return {"progress_valid": 0.0}
        loss = head.loss(feat.detach(), phi, mask)
        self.progress_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),
                                       self.ac.config.grad_clip)
        self.progress_opt.step()
        return {"progress_loss": float(loss.detach()),
                "progress_valid": float(mask.float().mean()),
                "progress_target_mean": float(phi[mask].mean())}

    def checkpoint(self, tag: str = "latest") -> Optional[Path]:
        """Write the run, or do nothing if checkpointing is off.

        Off is the default. Returning None rather than writing a partial file
        is the point: a caller that asked for a checkpoint and did not enable
        them gets no file and no half-file.
        """
        if not self.config.save_checkpoints:
            return None
        if self.checkpoint_dir is None or self.meta is None:
            return None
        # The actor's resolved revision, not Stage 1A's empty string: these
        # weights include the pretrained expert, and a checkpoint that does not
        # say which revision they came from cannot be checked against one.
        revision = str(getattr(getattr(self.actor, "loaded", None), "revision",
                               "") or self.meta.pretrained_revision or "")
        meta = CheckpointMeta(**{**self.meta.__dict__, "stage": "online",
                                 "pretrained_revision": revision,
                                 "step": self.env_steps})
        return save(self.checkpoint_dir / f"online_{tag}.pt", meta,
                    {"world_model": self.world_model, "actor": self.actor,
                     "critic": self.critic, "progress": self.progress_head},
                    {"world": self.world_opt, "actor": self.ac.actor_opt,
                     "critic": self.ac.critic_opt})


def run_online(cfg: Dict[str, Any], world_model, actor, critic, demo_sampler,
               env, *, config: OnlineConfig, ac_config: ActorCriticConfig,
               device="cuda", normalizer=None, coords=None, progress_head=None,
               potential=None, progress_config=None,
               checkpoint_dir: Optional[Path] = None,
               meta: Optional[CheckpointMeta] = None,
               on_metrics: Optional[Callable[[Dict[str, float]], None]] = None,
               ) -> OnlineTrainer:
    """Collect and learn until ``config.total_steps`` environment steps.

    Env steps and updates are counted separately, because they are different
    budgets: ``updates_per_collect`` decides how hard the model is trained per
    unit of experience, and reporting one as the other hides that ratio.
    """
    # Replay windows are cut like demonstration windows, chunk lookahead
    # included, so a mixed batch is one distribution and not two. See
    # train_imitation.run for why this is the chunk size and not one less.
    config.lookahead = max(int(getattr(actor, "chunk_size", 1)), 0)
    if hasattr(demo_sampler, "lookahead"):
        demo_sampler.lookahead = config.lookahead

    trainer = OnlineTrainer(
        world_model, actor, critic, demo_sampler, config=config,
        ac_config=ac_config, device=device, progress_head=progress_head,
        checkpoint_dir=checkpoint_dir, meta=meta, normalizer=normalizer,
        coords=coords, seed=int(config.seed), potential=potential,
        progress_config=progress_config)
    policy = LatentPolicy(
        world_model, actor, device=device, normalizer=normalizer,
        coords=coords,
        instruction=str(cfg["task"].get("instruction") or "") or None,
        execute=int(cfg["actor"].get("execute") or 1))

    seed = int(config.seed)
    next_checkpoint = int(config.checkpoint_every)
    next_eval = int(config.eval_every)
    last: Dict[str, float] = {}

    while trainer.env_steps < int(config.total_steps):
        collected = []
        for _ in range(int(config.episodes_per_collect)):
            # Reset before every rollout, not inside collect_episode: the
            # policy's recurrent state is the policy's, and an env reset does
            # not clear it.
            policy.reset()
            episode = collect_episode(
                env, policy, max_steps=int(config.max_episode_steps),
                seed=seed)
            trainer.replay.add(episode)
            trainer.env_steps += int(episode.steps)
            collected.append(episode)
            seed += 1

        for _ in range(int(config.updates_per_collect)):
            last = trainer.update()
        last["env_steps"] = float(trainer.env_steps)
        last["updates"] = float(trainer.updates)
        last["replay_episodes"] = float(len(trainer.replay))
        # From the rollouts just collected, not a separate evaluation: these
        # are the training-time curves the original pipeline reports.
        episode = episode_metrics(collected)
        last |= episode
        if on_metrics is not None:
            on_metrics(dict(last))
        print(f"[online] env_steps {trainer.env_steps} "
              f"updates {trainer.updates} "
              f"success_once={episode.get('episode/success_once', 0.0):.2f} "
              f"score={episode.get('episode/score', 0.0):.3f} "
              + " ".join(f"{k}={v:.3f}" for k, v in sorted(last.items())[:5]),
              flush=True)

        if config.eval_every and trainer.env_steps >= next_eval:
            from ..evaluation.policy import evaluate_policy

            policy.reset()
            report = evaluate_policy(
                env, policy, episodes=int(config.eval_episodes),
                seed_start=int(cfg["eval"]["seeds_start"]),
                max_steps=int(config.max_episode_steps))
            print(f"[online] eval {report}", flush=True)
            if on_metrics is not None:
                # Success rate and environment return are what the arms are
                # compared on, so they go through the same sink as the losses --
                # against env_steps, and never mixed with the shaping reward.
                # per_episode is a list, which the sink drops on its own.
                on_metrics({"env_steps": float(trainer.env_steps),
                            **{f"eval_{key}": value
                               for key, value in report.items()
                               if isinstance(value, (int, float, bool))}})
            next_eval += int(config.eval_every)

        if config.save_checkpoints and trainer.env_steps >= next_checkpoint:
            written = trainer.checkpoint()
            if written is not None:
                print(f"[online] wrote {written}", flush=True)
            next_checkpoint += int(config.checkpoint_every)

    return trainer
