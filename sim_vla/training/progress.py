"""Graph-derived progress shaping, as a separate and optional reward stream.

Three things keep this from contaminating the primary comparison.

It requires the graph. Progress targets come from the task schedule the graph
is mined against, so ``progress.enabled`` without ``graph.enabled`` is refused
at config load (``sim_vla/config.py``) rather than silently producing zeros.

Its reward is never added to the environment's. The two are carried and logged
apart, and evaluation reports environment success and environment return; a
shaped run that scored better only on its own shaping would be visible as
exactly that.

The shaping is potential-based: ``beta * (gamma * phi(s') - phi(s))``. That form
leaves the optimal policy unchanged, which is what makes the arm a comparison
of learning speed rather than of a different objective.

The baseline arm receives none of this -- no targets, no head, no reward term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

import networks


# Where the repository keeps the compiled task schedules and the assets they
# resolve roles against. ``configs/model/_base_.yaml`` names the same directory.
DEFAULT_SCHEDULE_DIR = "scenegraph/configs/schedules"
DEFAULT_CONFIGS_DIR = "scenegraph/configs"

# What fraction of an online budget is spent before shaping starts, and before
# it reaches full strength.
WARMUP_START_FRACTION = 0.2
WARMUP_END_FRACTION = 0.6


@dataclass
class ProgressConfig:
    enabled: bool = False
    beta: float = 0.05
    warmup_start: int = 400_000
    warmup_end: int = 700_000


def warmup_for(total_steps: int,
               start_fraction: float = WARMUP_START_FRACTION,
               end_fraction: float = WARMUP_END_FRACTION) -> tuple:
    """A warm-up scaled to the run, rather than to a longer one.

    The repository's defaults start shaping at 400k environment steps and
    reach full beta at 700k. A 200k-step run under those numbers never turns
    shaping on at all: ``beta_at`` returns 0.0 for every step of it, the arm is
    identical to plain ``graph``, and the comparison reports that progress
    shaping does nothing. Scaling to the budget is what makes the arm the arm.
    """
    total = max(int(total_steps), 0)
    start = int(total * float(start_fraction))
    end = max(int(total * float(end_fraction)), start + 1)
    return start, end


def beta_at(config: ProgressConfig, step: int) -> float:
    """Linear warm-up, so shaping does not dominate an untrained value head."""
    if not config.enabled:
        return 0.0
    if step <= config.warmup_start:
        return 0.0
    if step >= config.warmup_end:
        return float(config.beta)
    span = max(config.warmup_end - config.warmup_start, 1)
    return float(config.beta) * (step - config.warmup_start) / span


class ProgressHead(nn.Module):
    """Predicts scalar task progress from a latent feature.

    Only ever constructed for the graph arm: its target is the graph schedule's
    phase, and there is no schedule without a graph.
    """

    def __init__(self, config, feature_dim: int):
        super().__init__()
        self.net = networks.MLPHead(config.critic, int(feature_dim))

    def potential(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).mode().squeeze(-1)

    def loss(self, feat: torch.Tensor, target: torch.Tensor,
             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        error = (self.potential(feat) - target.detach()) ** 2
        if mask is None:
            return error.mean()
        weight = mask.to(error.dtype)
        return (error * weight).sum() / weight.sum().clamp(min=1.0)


def shaping_reward(head: ProgressHead, feat: torch.Tensor, discount: float
                   ) -> torch.Tensor:
    """``gamma * phi(s') - phi(s)`` over an imagined rollout.

    Potential-based, so it cannot change which policy is optimal -- only how
    quickly one is found.
    """
    phi = head.potential(feat)
    return discount * phi[1:] - phi[:-1]


def build_progress(config, feature_dim: int, *, graph_enabled: bool,
                   progress_enabled: bool) -> Optional[ProgressHead]:
    """A head for the graph arm with shaping on, and None otherwise."""
    if not progress_enabled:
        return None
    if not graph_enabled:
        raise SystemExit(
            "progress shaping requires the graph arm: its targets come from "
            "the graph schedule, and a baseline has no schedule to read.")
    return ProgressHead(config, feature_dim)


class SchedulePotential:
    """Observed-graph ``Phi`` for one task, from the compiled task schedule.

    This is the supervision the progress head regresses onto. It reads the
    *observed* packed-graph labels the collector recorded -- ``node_ent``,
    ``edge_rel``, ``edge_abs`` and the edge index arrays -- and never the
    decoder's predictions, which is why it needs nothing from the world model
    and can be computed for a demonstration batch directly.

    ``dreamer.py`` builds the same object from a live graph-enabled env, which
    it needs because it reads the task identity and the whitelist directory off
    that env. Here both are recorded in the dataset's metadata, so no simulator
    has to be running to compile the schedule.
    """

    def __init__(self, env_id: str, whitelist_dir: str, n_abs: int,
                 *, schedule_dir: str = DEFAULT_SCHEDULE_DIR,
                 configs_dir: str = DEFAULT_CONFIGS_DIR, device=None):
        from scenegraph.adapters.graph_vocab import build_entity_vocab
        from scenegraph.core.schedule import maniskill_schedule_source, \
            compile_from_source

        from progress import TaskScheduleReplayPotential

        self.env_id = str(env_id)
        source = maniskill_schedule_source(
            self.env_id, str(configs_dir), str(schedule_dir),
            str(whitelist_dir or ""))
        self.source = source
        # The vocabulary comes from the directory the source resolved, so the
        # compiled roles index the rows the packer actually wrote.
        schedule = compile_from_source(
            source, build_entity_vocab(source.whitelist_dir))
        self.schedule = schedule
        self.scorer = TaskScheduleReplayPotential(schedule, int(n_abs))
        if device is not None:
            self.scorer = self.scorer.to(device)
        self.phases = len(schedule.phases)

    @torch.no_grad()
    def targets(self, batch):
        """``(phi, valid)`` shaped like the batch's ``(batch, time)``.

        ``compact_graph`` folds the batch and time axes into one graph axis, so
        the result is reshaped back rather than returned flat -- the masks it
        is combined with are per (batch, time).
        """
        from graph import compact_graph
        from scenegraph.adapters.graph_pack import GRAPH_KEYS

        packed = {key: batch[key] for key in GRAPH_KEYS}
        reference = packed["graph_node_ent"]
        shape = tuple(reference.shape[:-1])
        compact = compact_graph(packed)
        phi, valid = self.scorer(
            compact.node_ent, compact.edge_rel, compact.edge_abs,
            compact.edge_src_local, compact.edge_dst_local,
            compact.edge_graph, compact.graph_count)
        return phi.reshape(shape).float(), valid.reshape(shape).bool()

    def describe(self) -> Dict[str, Any]:
        return {"env_id": self.env_id, "phases": self.phases,
                "schedule": self.source.schedule_path,
                "whitelist_dir": self.source.whitelist_dir}


def availability(cfg, metadata=None) -> list:
    """What the progress arm is missing, or an empty list.

    Checked from paths and recorded metadata rather than by trying to build
    the thing, so it can run before Stage 1A.
    """
    import os

    progress = dict((cfg.get("model") or {}).get("progress") or {})
    graph_meta = dict((metadata or {}).get("graph") or {})
    env_id = str((cfg.get("task") or {}).get("env_id") or "")
    schedule_dir = str(progress.get("schedule_dir") or DEFAULT_SCHEDULE_DIR)
    configs_dir = str(progress.get("configs") or DEFAULT_CONFIGS_DIR)

    missing = []
    if not bool((cfg.get("model") or {}).get("graph", {}).get("enabled")):
        missing.append(
            "model.graph.enabled is false: the schedule resolves roles against "
            "the graph's entity vocabulary, and a baseline has no graph")
    if not env_id:
        missing.append("task.env_id is empty, so no schedule can be named")
    else:
        path = os.path.join(schedule_dir, f"{env_id}.json")
        if not os.path.isfile(path):
            missing.append(
                f"no task schedule at {path}; the repository ships one per "
                "supported task under " + DEFAULT_SCHEDULE_DIR)
    whitelist = graph_meta.get("whitelist_dir") or os.path.join(
        configs_dir, "subtask_whitelists", env_id)
    if not os.path.isdir(str(whitelist)):
        missing.append(
            f"no whitelist directory at {whitelist}; roles are resolved "
            "against the entity vocabulary it defines")
    if not graph_meta.get("absolute_tokens") and not graph_meta.get(
            "vocab_sizes"):
        missing.append(
            "the dataset metadata records no absolute-token vocabulary, so "
            "the number of spatial bins the scorer needs is unknown")
    return missing


def preflight(cfg, metadata=None) -> None:
    """Refuse the progress arm before anything expensive, if it cannot run.

    This used to refuse unconditionally on the belief that no schedule existed
    and that the scorer needed decoder probabilities. Both were wrong: the
    repository ships a schedule per task under ``scenegraph/configs/schedules``
    and :class:`TaskScheduleReplayPotential` reads observed labels. What is
    checked now is whether *this* run has the pieces.
    """
    if not bool((cfg.get("model") or {}).get("progress", {}).get("enabled")):
        return
    missing = availability(cfg, metadata)
    if not missing:
        return
    detail = "\n".join(f"  - {item}" for item in missing)
    raise SystemExit(
        "refusing to run the graph_progress arm: its supervision is not "
        "available for this run.\n" + detail + "\nRun --experiment graph "
        "instead, or supply the pieces above. This stops here rather than "
        "training a head on invented targets or silently running plain graph "
        "training under the graph_progress name.")


def build_potential(cfg, metadata, *, device=None):
    """The schedule potential for this run, or None when progress is off."""
    import os

    if not bool((cfg.get("model") or {}).get("progress", {}).get("enabled")):
        return None
    preflight(cfg, metadata)
    progress = dict(cfg["model"]["progress"])
    graph_meta = dict((metadata or {}).get("graph") or {})
    # Two numbers in the metadata differ by exactly one, and the scorer wants
    # the larger. ``vocab_sizes["absolute"]`` is ``len(Vocab)``, which counts
    # the pad slot at index 0 -- the same number configs/model/_base_.yaml
    # pins as graph.n_abs and dreamer.py passes to this scorer.
    # ``absolute_tokens`` is the raw token->id mapping and carries no pad
    # entry, while graph_vocab._index numbers its tokens from 1. So its
    # highest id equals its length, and allocating a mask of that length puts
    # that id one past the end -- which is the ValueError _label_mask raises.
    sizes = dict(graph_meta.get("vocab_sizes") or {})
    absolute = graph_meta.get("absolute_tokens") or {}
    n_abs = int(sizes.get("absolute") or 0)
    if not n_abs and absolute:
        n_abs = len(absolute) + 1
    if not n_abs:
        raise SystemExit(
            "the dataset records no absolute-token vocabulary; the scorer "
            "needs its size to allocate one slot per spatial bin")
    return SchedulePotential(
        str(cfg["task"]["env_id"]), str(graph_meta.get("whitelist_dir") or ""),
        n_abs,
        schedule_dir=str(progress.get("schedule_dir") or DEFAULT_SCHEDULE_DIR),
        configs_dir=str(progress.get("configs") or DEFAULT_CONFIGS_DIR),
        device=device)
