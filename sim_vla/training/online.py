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
anything. Imagination executes the same ``execute`` actions per chunk (see
:mod:`sim_vla.training.imagination`), so the policy the actor update optimises
is the one collecting.
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
from .precision import autocast
from .progress import PROGRESS_LR


@dataclass
class OnlineConfig:
    total_steps: int = 1_000_000
    episodes_per_collect: int = 2
    updates_per_collect: int = 8
    # Replay timesteps trained per environment step, as in trainer.py.
    # Zero preserves the legacy fixed updates_per_collect schedule.
    train_ratio: float = 0.0
    precision: str = "float32"
    demo_fraction: float = 0.5
    batch_size: int = 16
    sequence_length: int = 64
    burn_in: int = 8
    max_episode_steps: int = 150
    seed: int = 0
    actor_every: int = 1
    # Action-only lookahead for replay windows. Zero online, for both replay
    # sources: it existed to supervise whole demonstrated chunks, and nothing
    # in Stage 2 imitates. run_online sets it, and the demonstration sampler's,
    # so a mixed batch stays one shape.
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
    # Online episodes the replay must hold before a batch mixes them in. With
    # parallel envs it is also when updates begin: an episode only reaches the
    # replay whole, so the first round of rollouts trains nothing until it ends.
    min_replay: int = 4

    def updates_due(self, env_steps: int, updates: int) -> int:
        if not np.isfinite(self.train_ratio) or self.train_ratio < 0:
            raise ValueError("train_ratio must be finite and nonnegative")
        if self.batch_size <= 0 or self.sequence_length <= 0:
            raise ValueError("batch_size and sequence_length must be positive")
        if self.train_ratio == 0:
            return int(self.updates_per_collect)
        # Count whole windows, including burn-in, just as the original trainer
        # counts batch_size * batch_length. Carry fractional updates forward
        # by using cumulative steps instead of rounding each collection.
        target = int(env_steps * self.train_ratio
                     / (self.batch_size * self.sequence_length))
        return max(0, target - updates)


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


def _env_rows(obs: Dict[str, Any]) -> int:
    """How many envs a batched observation holds: its leading axis."""
    return int(np.asarray(next(iter(obs.values()))).shape[0])


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

    ``reset(batch=n)`` drives ``n`` envs in lockstep: observations and actions
    then carry a leading env axis, and each row has its own recurrent state and
    its own ``a_(t-1)``. ``reset()`` keeps the one-env interface -- one
    observation in, one action out -- and computes exactly what it did before
    the batched form existed, as a batch of one.

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
        # Imagination executes the same number of actions from one generated
        # chunk, so the imagined policy is this policy. The trainer passes its
        # own resolved value rather than reading the config twice.
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

    def reset(self, batch: Optional[int] = None) -> None:
        """Start an episode: no recurrent state, no queued actions.

        ``batch`` is the number of envs stepped together; None is one env
        through the unbatched interface.
        """
        if batch is not None and int(batch) < 1:
            raise ValueError(f"batch={batch} must be at least 1")
        self._batch = None if batch is None else int(batch)
        self._state = None
        self._queue: List[np.ndarray] = []
        shape = ((self.action_dim,) if self._batch is None
                 else (self._batch, self.action_dim))
        self._prev_action = np.zeros(shape, dtype=np.float32)
        self._first = True

    def _window(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """One observation per env as a length-1 window in the training layout.

        ``obs`` carries a leading env axis. ``_prev_action`` is in raw
        environment units -- the command that was actually sent -- and
        ``to_model_batch`` standardises it and maps it into dynamics
        coordinates, exactly as it does for a training window.
        """
        rows = _env_rows(obs)
        batch: Dict[str, Any] = {key: np.asarray(value)[:, None]
                                 for key, value in obs.items()}
        # Storage naming, so to_model_batch renames and normalizes it the way
        # it does for a training window.
        batch["actions"] = np.asarray(self._prev_action, np.float32).reshape(
            rows, 1, self.action_dim)
        batch["is_first"] = np.full((rows, 1), self._first, dtype=bool)
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
        """Sample one action chunk per env and return the part executed.

        Each returned entry is one step, with one row per env.

        The clip happens here, in the actor's own coordinates, so that what is
        returned is the command the environment will run and what is fed back
        as ``a_(t-1)`` is that same command. Clipping downstream -- in the
        evaluation loop, say -- left the policy conditioning on an action that
        was never executed.
        """
        from ..models.flow_sampler import sample_actions

        cond = self.actor.condition(feat, self.instruction)
        device = getattr(self.actor, "device", feat.device)
        rows = int(feat.shape[0])
        chunk = sample_actions(
            self.actor.velocity_fn(), cond, batch=rows,
            chunk=int(self.actor.chunk_size), dim=self.action_dim,
            steps=self.flow_steps,
            # The actor's device, not the feature's: condition() moved the
            # feature to the pretrained weights and the noise starts there.
            device=device,
            dtype=feat.dtype, differentiable=False)
        normalized = chunk[:, : self.execute]             # (rows, execute, A)
        if self.coords is not None:
            bounded = self.coords.executed(normalized)
            self.clipped += int(
                (bounded != normalized).any(dim=-1).sum().item())
            raw = self.coords.denormalize(bounded).float().cpu().numpy()
        else:
            raw = normalized.float().cpu().numpy()
        return [np.asarray(raw[:, step], dtype=np.float32)
                for step in range(raw.shape[1])]

    def __call__(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        single = self._batch is None
        if single:
            obs = {key: np.asarray(value)[None] for key, value in obs.items()}
        else:
            rows = _env_rows(obs)
            if rows != self._batch:
                raise ValueError(
                    f"{rows} observations for a policy reset to "
                    f"{self._batch} envs; each row carries its own recurrent "
                    "state, so the count cannot change mid-episode")
        feat = self._encode(obs)
        if not self._queue:
            self._queue = self._plan(feat)
        action = self._queue.pop(0)
        if single:
            action = action[0]
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
                 progress_config=None, progress_lr: float = PROGRESS_LR):
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
        # The demonstration sampler stays with this trainer: it feeds the
        # world model's mixed batches. The actor update never sees it -- there
        # is no online imitation term.
        self.ac = ActorCriticTrainer(world_model, actor, critic, ac_config,
                                     coords=coords,
                                     progress_head=progress_head,
                                     device=self.device)
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
            ignore_terminations=bool(self.config.ignore_terminations),
            min_replay=int(self.config.min_replay)))

        self.world_opt.zero_grad(set_to_none=True)
        with autocast(self.device, self.config.precision):
            total, _losses, _aux = self.world_model.loss(batch)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
        self.world_opt.step()
        metrics = {"world_loss": float(total.detach())}
        # These outputs and gradients are not needed for the actor update.
        del total, _losses, _aux
        self.world_opt.zero_grad(set_to_none=True)

        # Re-encoded after the step, not reused from before it: the posterior
        # in ``_aux`` came from the parameters that have just been replaced.
        # Encoded once and shared, because the progress head and imagination
        # want the same posterior of the same batch under the same model;
        # encoding twice paid for a second forward pass and gave the two
        # different latent samples.
        with torch.no_grad(), autocast(self.device, self.config.precision):
            post = self.world_model.observe(batch)["post"]
        with autocast(self.device, self.config.precision):
            metrics |= self.update_progress(batch, post)
            start = start_states(self.world_model, batch, post=post)
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

    def update_progress(self, batch: Dict[str, torch.Tensor],
                        post=None) -> Dict[str, float]:
        """Keep the progress head fitted to the observed-graph potential.

        The head arrives trained from Stage 1A, where it was trained jointly
        with the world model; it keeps training here because the features it
        reads move with the world model. Here the fit is detached, with the
        head's own optimizer (:func:`sim_vla.training.progress.fit_progress`):
        the same masked Huber objective Stage 1A adds to the world-model loss,
        on features re-encoded by the just-updated world model, and it moves
        the head alone.

        ``post`` is that re-encoding when the caller already has it; the
        features are taken from it rather than encoding the same batch again.
        """
        if self.progress_opt is None or self.potential is None:
            return {}
        from .progress import fit_progress

        with torch.no_grad():
            if post is None:
                post = self.world_model.observe(batch)["post"]
            feat = self.world_model.features(post)
        return fit_progress(self.ac.progress_head, self.progress_opt,
                            self.potential, feat, batch,
                            grad_clip=self.ac.config.grad_clip)

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
        ac = self.ac.config
        # What the run actually optimized, not what a config file said. Two
        # checkpoints with the same weights and different objectives are
        # different experiments, and a resume that switches between them is a
        # new experiment rather than a continuation -- so the identity is
        # written down where a later reader will find it.
        from .actor_critic import OBJECTIVE, RETURN, START_SELECTION

        extra = dict(self.meta.extra or {})
        extra |= {
            # What was optimised, and over what. A checkpoint whose weights
            # came from a different objective is a different experiment, not a
            # continuation, and this is where a later reader finds that out.
            "actor_objective": OBJECTIVE,
            "return": RETURN,
            "execute": int(ac.execute),
            "start_selection": START_SELECTION,
            "flow_steps": int(ac.flow_steps),
            "discount": float(ac.discount),
            "imagination_microbatch": int(ac.imagination_microbatch),
            "critic_warmup": int(ac.critic_warmup),
            "grad_clip": float(ac.grad_clip),
            "actor_lr": float(ac.actor_lr),
            "critic_lr": float(ac.critic_lr),
            "precision": str(ac.precision),
            "train_ratio": float(self.config.train_ratio),
            # Counters, so a later run can say how far this one got and a log
            # can be aligned to it.
            "actor_updates": int(self.ac.actor_steps),
            "actor_critic_updates": int(self.ac.step),
            "world_updates": int(self.updates),
            "env_steps": int(self.env_steps),
            # The coordinate identity: an action recorded under one
            # normalization is a different action under another.
            "action_coordinates": (self.coords.describe()
                                   if hasattr(self.coords, "describe")
                                   else None),
        }
        meta = CheckpointMeta(**{**self.meta.__dict__, "stage": "online",
                                 "pretrained_revision": revision,
                                 "step": self.env_steps, "extra": extra})
        return save(self.checkpoint_dir / f"online_{tag}.pt", meta,
                    {"world_model": self.world_model, "actor": self.actor,
                     "critic": self.critic, "progress": self.progress_head},
                    {"world": self.world_opt, "actor": self.ac.actor_opt,
                     "critic": self.ac.critic_opt,
                     "progress": self.progress_opt})


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
    unit of experience when train_ratio is zero. Otherwise the schedule uses
    replay timesteps per environment step, matching the original Dreamer.
    """
    # No lookahead online, for either source. It exists to supervise whole
    # demonstrated action chunks, which is Stage 1B's job; Stage 2 trains the
    # world model on these windows and imagines its own actions. Both sources
    # are set here because a mixed batch needs one target-axis length.
    config.lookahead = 0
    if hasattr(demo_sampler, "lookahead"):
        demo_sampler.lookahead = 0

    # One resolved value for both: the number of actions imagination executes
    # from a chunk is the number the environment executes before replanning.
    execute = int(ac_config.execute)
    configured = int((cfg.get("actor") or {}).get("execute") or execute)
    if configured != execute:
        raise ValueError(
            f"actor.execute={configured} but the actor-critic config was "
            f"built with execute={execute}. One setting decides how many "
            "actions a chunk contributes, in imagination and online alike.")

    trainer = OnlineTrainer(
        world_model, actor, critic, demo_sampler, config=config,
        ac_config=ac_config, device=device, progress_head=progress_head,
        checkpoint_dir=checkpoint_dir, meta=meta, normalizer=normalizer,
        coords=coords, seed=int(config.seed), potential=potential,
        progress_config=progress_config)
    # Collection samples from the policy being improved, with the same
    # deterministic flow sampler imagination uses.
    policy = LatentPolicy(
        world_model, actor, device=device, normalizer=normalizer,
        coords=coords,
        instruction=str(cfg["task"].get("instruction") or "") or None,
        execute=execute)
    # One policy: the evaluation runs exactly what is being trained.
    eval_policy, eval_prefix = policy, "eval"

    num_envs = int(getattr(env, "num_envs", 1))
    if num_envs > 1:
        _run_parallel(trainer, policy, env, config, num_envs, on_metrics)
        return trainer

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

        updates_due = config.updates_due(trainer.env_steps, trainer.updates)
        for _ in range(updates_due):
            last = trainer.update()
        last["env_steps"] = float(trainer.env_steps)
        last["updates"] = float(trainer.updates)
        last["replay_episodes"] = float(len(trainer.replay))
        last["train_ratio_actual"] = (
            trainer.updates * config.batch_size * config.sequence_length
            / max(trainer.env_steps, 1))
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

            eval_policy.reset()
            report = evaluate_policy(
                env, eval_policy, episodes=int(config.eval_episodes),
                seed_start=int(cfg["eval"]["seeds_start"]),
                max_steps=int(config.max_episode_steps))
            print(f"[online] {eval_prefix} {report}", flush=True)
            if on_metrics is not None:
                # Success rate and environment return are what the arms are
                # compared on, so they go through the same sink as the losses --
                # against env_steps, and never mixed with the shaping reward.
                # per_episode is a list, which the sink drops on its own.
                on_metrics({"env_steps": float(trainer.env_steps),
                            **{f"{eval_prefix}_{key}": value
                               for key, value in report.items()
                               if isinstance(value, (int, float, bool))}})
            next_eval += int(config.eval_every)

        if config.save_checkpoints and trainer.env_steps >= next_checkpoint:
            written = trainer.checkpoint()
            if written is not None:
                print(f"[online] wrote {written}", flush=True)
            next_checkpoint += int(config.checkpoint_every)

    return trainer


def _row(obs: Dict[str, np.ndarray], index: int) -> Dict[str, np.ndarray]:
    return {key: value[index] for key, value in obs.items()}


def _run_parallel(trainer: OnlineTrainer, policy: LatentPolicy, env,
                  config: OnlineConfig, num_envs: int,
                  on_metrics: Optional[Callable[[Dict[str, float]], None]]
                  ) -> None:
    """Step ``num_envs`` envs in lockstep and train between steps.

    The main trainer's schedule rather than collect-then-train: after every
    vector step the updates its environment steps are owed run at once, so the
    policy collecting the next step is the one just trained. Collecting whole
    rounds first would refresh a 128-env policy once per 19,200 steps.

    One round is one episode in every env. They reset together and all end at
    ``max_episode_steps``, because nothing terminates, so there is no partial
    reset. Episodes reach the replay whole at the end of their round -- a
    replay window must not cross a boundary it cannot see -- so updates wait
    for the replay to hold ``min_replay`` episodes, which is the end of the
    first round, and the updates owed for that round run then. The count is
    the same cumulative :meth:`OnlineConfig.updates_due` as the one-env loop,
    so the update budget is ``env_steps * train_ratio / (batch_size *
    sequence_length)`` whatever the env count.

    The run stops at ``total_steps``, mid-round if that is where it lands, and
    the unfinished episodes go with it: nothing would train on them.
    """
    if config.train_ratio == 0:
        raise ValueError(
            "train_ratio=0 selects the legacy updates_per_collect schedule, "
            "which is defined per collection of episodes_per_collect "
            "episodes; parallel envs train between steps and need a "
            "train_ratio")
    if config.eval_every:
        raise NotImplementedError(
            "periodic evaluation drives one env through evaluate_policy; with "
            f"{num_envs} parallel envs it would step env 0 alone. Evaluate "
            "separately, or set num_envs=1")
    total = int(config.total_steps)
    horizon = int(config.max_episode_steps)
    every = max(int(config.checkpoint_every), 1)
    next_checkpoint = every
    # One seed per episode, continuing the one-env loop's sequence.
    seed = int(config.seed)
    rounds = 0

    def train() -> None:
        if len(trainer.replay) < int(config.min_replay):
            return
        due = config.updates_due(trainer.env_steps, trainer.updates)
        if due <= 0:
            return
        last: Dict[str, float] = {}
        for _ in range(due):
            last = trainer.update()
        last["env_steps"] = float(trainer.env_steps)
        last["updates"] = float(trainer.updates)
        last["replay_episodes"] = float(len(trainer.replay))
        last["train_ratio_actual"] = (
            trainer.updates * config.batch_size * config.sequence_length
            / max(trainer.env_steps, 1))
        if on_metrics is not None:
            on_metrics(dict(last))
        print(f"[online] env_steps {trainer.env_steps} "
              f"updates {trainer.updates} "
              + " ".join(f"{k}={v:.3f}" for k, v in sorted(last.items())[:5]),
              flush=True)

    def checkpoint() -> None:
        nonlocal next_checkpoint
        if not config.save_checkpoints or trainer.env_steps < next_checkpoint:
            return
        written = trainer.checkpoint()
        if written is not None:
            print(f"[online] wrote {written}", flush=True)
        while next_checkpoint <= trainer.env_steps:
            next_checkpoint += every

    while trainer.env_steps < total:
        seeds = list(range(seed, seed + num_envs))
        seed += num_envs
        # Every row's recurrent state starts over with its episode.
        policy.reset(batch=num_envs)
        obs = env.reset_all(seeds)
        episodes = [OnlineEpisode() for _ in range(num_envs)]
        for index, episode in enumerate(episodes):
            episode.add_observation(_row(obs, index))

        complete = False
        for step in range(horizon):
            actions = policy(obs)
            out = env.step_all(actions)
            last_flags = np.asarray(out["is_last"], dtype=bool)
            if last_flags.any() and not last_flags.all():
                raise RuntimeError(
                    f"envs {np.flatnonzero(last_flags).tolist()} ended at step "
                    f"{step + 1} and the rest did not. Parallel collection "
                    "assumes one shared horizon and no terminations; stepping "
                    "on would append transitions after an episode's end.")
            for index, episode in enumerate(episodes):
                episode.add_transition(
                    actions[index], out["reward"][index],
                    out["is_terminal"][index], last_flags[index],
                    out["success"][index])
                episode.add_observation(_row(out["obs"], index))
            obs = out["obs"]
            trainer.env_steps += num_envs
            if last_flags.all() or step + 1 == horizon:
                complete = True
                break
            train()
            checkpoint()
            if trainer.env_steps >= total:
                break
        if not complete:
            break

        for episode in episodes:
            trainer.replay.add(episode)
        rounds += 1
        # From the rollouts just collected, as the one-env loop reports them.
        episode = episode_metrics(episodes)
        if on_metrics is not None:
            on_metrics({"env_steps": float(trainer.env_steps), **episode})
        print(f"[online] round {rounds}: {num_envs} episodes, env_steps "
              f"{trainer.env_steps} "
              f"success_once={episode.get('episode/success_once', 0.0):.2f} "
              f"score={episode.get('episode/score', 0.0):.3f}", flush=True)
        train()
        checkpoint()
