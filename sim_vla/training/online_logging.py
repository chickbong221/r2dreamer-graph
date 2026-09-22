"""Progress during replay-update backlogs, without changing their schedule."""

from __future__ import annotations

import time


PHASES = {
    "replay": 1, "world_forward": 2, "world_backward": 3,
    "posterior": 4, "progress_head": 5, "imagine": 6,
    "actor_backward": 7, "critic_backward": 8, "optimizer": 9,
}


def train_updates(trainer, due, on_metrics=None):
    """Report the backlog immediately and every completed optimizer update.

    Console phase reports are limited to once per 15 seconds, except that the
    first update announces each phase on first entry. They mark Python-side
    progress, not synchronized GPU timings; --profile-online measures those.
    """
    if due <= 0:
        return {}
    config = trainer.config
    started = time.monotonic()
    last_print = started
    last_summary = started
    initial_updates = trainer.updates
    completed = 0
    current_started = started
    announced = set()

    def emit(values):
        if on_metrics is not None:
            on_metrics({"env_steps": float(trainer.env_steps), **values})

    print(f"[online] training backlog: {due} updates due at "
          f"env_steps={trainer.env_steps}; completed_updates={initial_updates}; "
          f"train_ratio={config.train_ratio:g}, "
          f"replay_batch={config.batch_size}x{config.sequence_length}. "
          "Collection pauses while these updates run.", flush=True)
    emit({"online_progress/pending_updates": float(due),
          "online_progress/completed_updates": float(initial_updates)})

    def phase(name, **details):
        nonlocal last_print
        now = time.monotonic()
        first_entry = initial_updates == 0 and completed == 0 and name not in announced
        if not first_entry and now - last_print < 15.0:
            return
        announced.add(name)
        last_print = now
        micro = ""
        if "microbatch" in details:
            micro = (f" microbatch={details['microbatch']}/{details['microbatches']}"
                     f" starts={details['imagination_starts']}"
                     f" actor_training={bool(details['actor_training'])}")
        print(f"[online] update {completed + 1}/{due} "
              f"(global {initial_updates + completed + 1}) phase={name}"
              f"{micro} elapsed={now - current_started:.1f}s", flush=True)
        emit({"online_progress/phase": float(PHASES[name]),
              "online_progress/current_update": float(initial_updates + completed + 1),
              "online_progress/update_elapsed_s": now - current_started,
              "online_progress/pending_updates": float(due - completed),
              **{f"online_progress/{key}": float(value)
                 for key, value in details.items()}})

    last = {}
    for index in range(due):
        current_started = time.monotonic()
        last = trainer.update(on_progress=phase)
        now = time.monotonic()
        completed = index + 1
        remaining = due - completed
        elapsed = now - started
        eta = elapsed / completed * remaining
        last |= {
            "env_steps": float(trainer.env_steps),
            "updates": float(trainer.updates),
            "replay_episodes": float(len(trainer.replay)),
            "train_ratio_actual": (trainer.updates * config.batch_size
                                   * config.sequence_length / max(trainer.env_steps, 1)),
            "update_seconds": now - current_started,
            "online_progress/completed_updates": float(trainer.updates),
            "online_progress/pending_updates": float(remaining),
            "online_progress/backlog_elapsed_s": elapsed,
            "online_progress/backlog_eta_s": eta,
            "online_progress/phase": 0.0,
        }
        # Losses reach W&B on every completed update, even while env_steps
        # stays fixed throughout a long initial backlog.
        emit(dict(last))
        if completed == 1 or remaining == 0 or now - last_summary >= 15.0:
            losses = " ".join(f"{key}={last[key]:.4g}" for key in
                              ("world_loss", "critic_loss", "actor_loss", "actor_grad_norm")
                              if key in last)
            print(f"[online] trained {completed}/{due} "
                  f"global_update={trainer.updates} env_steps={trainer.env_steps} "
                  f"update_s={last['update_seconds']:.1f} "
                  f"pending={remaining} ETA={eta / 60:.1f}min {losses}", flush=True)
            last_print = now
            last_summary = now
    return last
