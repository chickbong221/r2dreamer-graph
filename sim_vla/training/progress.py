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

The head is the regular trainer's: :class:`networks.ProgressHead`, one sigmoid
scalar in ``[0, 1]``, regressed with the masked Huber loss of
``dreamer.py:_progress_model_loss``. Stage 1A trains it jointly with the world
model, as ``dreamer.py`` does -- its loss is added to the world-model loss on
attached posterior features. Stage 2 keeps it fitted on detached features with
its own optimizer, so online the world model is never moved by it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F

import networks


# Where the repository keeps the compiled task schedules and the assets they
# resolve roles against. ``configs/model/_base_.yaml`` names the same directory.
DEFAULT_SCHEDULE_DIR = "scenegraph/configs/schedules"
DEFAULT_CONFIGS_DIR = "scenegraph/configs"

# What fraction of an online budget is spent before shaping starts, and before
# it reaches full strength.
WARMUP_START_FRACTION = 0.2
WARMUP_END_FRACTION = 0.6

# Stage 2's own optimizer for the head. Stage 2 fits it on detached features,
# so neither number reaches the world model. Stage 1A uses neither: there the
# head is one more parameter group of the world model's optimizer, under the
# world model's learning rate and clipping.
PROGRESS_LR = 3e-4
PROGRESS_GRAD_CLIP = 1.0

# dreamer.py:_progress_model_loss. Huber rather than squared error because the
# target is a weighted step function: a frame that crosses two rungs at once
# is a real jump, not an outlier to chase.
PROGRESS_HUBER_DELTA = 0.1

# What a stored head is, beyond its tensors. Written into the world model's
# checkpoint metadata and required on restore. A head from before joint
# pretraining was a twohot readout fitted on detached features, beside a world
# model the progress loss never reached; neither half is what this trains.
HEAD_IDENTITY = {
    "architecture": "networks.ProgressHead",
    "output": "sigmoid_0_1",
    "objective": f"masked_huber_delta_{PROGRESS_HUBER_DELTA:g}",
    "stage_1a": "joint_with_world_model",
}


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


def predict(head: networks.ProgressHead, feat: torch.Tensor) -> torch.Tensor:
    """``phi(s)`` in ``[0, 1]``, shaped like ``feat`` without its last axis."""
    return head(feat).squeeze(-1)


def shaping_reward(head: networks.ProgressHead, feat: torch.Tensor,
                   discount: float, *, cont: Optional[torch.Tensor] = None
                   ) -> torch.Tensor:
    """``gamma * cont * phi(s') - phi(s)`` over an imagined rollout.

    Potential-based, so it cannot change which policy is optimal -- only how
    quickly one is found. The sigmoid bounds ``phi``, so ``|F_t| <= 1``.

    ``cont`` is the continuation of each imagined transition, read at the
    successor like its reward. Including it is what keeps the shaping
    potential-based when an episode can end: the successor's potential is only
    reachable if the episode continues into it, and a transition that
    terminates must not be credited with the potential of a state the agent
    never occupies. Under ``ignore_terminations`` it is one throughout and
    this is the plain ``gamma * phi(s') - phi(s)``.
    """
    phi = predict(head, feat)
    successor = phi[1:]
    if cont is not None:
        successor = cont.to(successor.dtype) * successor
    return discount * successor - phi[:-1]


def progress_mask(batch: Mapping[str, torch.Tensor],
                  phi_valid: torch.Tensor) -> torch.Tensor:
    """The ``(batch, time)`` rows a progress target may be scored at.

    ``loss_mask`` drops burn-in and padding, exactly as it does for every
    world-model term. ``valid`` is padding alone and already implied by it; it
    is ANDed in anyway so a loader that ever loosened ``loss_mask`` could not
    start scoring a repeated padding row. ``phi_valid`` drops frames the
    schedule cannot score -- a role that matched no node, a relation the frame
    never observed. Scoring those as zero would teach the head that an unseen
    target is a failed reach.
    """
    mask = batch["loss_mask"].bool() & phi_valid.to(
        batch["loss_mask"].device).bool()
    if "valid" in batch:
        mask = mask & batch["valid"].bool()
    return mask


def progress_loss(head: networks.ProgressHead, feat: torch.Tensor,
                  target: torch.Tensor, mask: torch.Tensor
                  ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Masked Huber between the head and the observed potential.

    The objective of ``dreamer.py:_progress_model_loss``, and the only one the
    head has in either stage. What it trains is decided by the caller through
    ``feat``: attached, the gradient reaches the world model too (Stage 1A);
    detached, only the head (Stage 2). The target is always detached.

    Invalid rows are replaced by zero before the loss rather than multiplied
    by the mask afterwards, so a NaN the scorer left in a row it marked invalid
    cannot reach the sum as ``NaN * 0``.
    """
    phi = predict(head, feat)
    if tuple(phi.shape) != tuple(target.shape):
        # F.huber_loss broadcasts a (B, T, 1) against a (B, T) with only a
        # warning, and the loss is then over B*T*T pairs.
        raise ValueError(
            f"progress target {tuple(target.shape)} does not match the head's "
            f"prediction {tuple(phi.shape)}")
    mask = mask.to(phi.device).bool()
    target = torch.where(mask, target.detach().to(phi.device, phi.dtype),
                         torch.zeros_like(phi))
    weight = mask.to(phi.dtype)
    count = weight.sum().clamp_min(1.0)
    error = F.huber_loss(phi, target, reduction="none",
                         delta=PROGRESS_HUBER_DELTA)
    loss = (error * weight).sum() / count
    with torch.no_grad():
        mean = (target * weight).sum() / count
        std = ((((target - mean) ** 2) * weight).sum() / count).sqrt()
        # The four dreamer.py logs. A low valid fraction reads as a
        # persistence bug; a near-zero target_std means behaviour produces no
        # spread to learn, and makes every other progress number meaningless.
        metrics = {
            "progress_valid": float(weight.mean()),
            "progress_target_mean": float(mean),
            "progress_target_std": float(std),
            "progress_head_mae": float(
                ((phi.detach() - target).abs() * weight).sum() / count),
        }
    return loss, metrics


def joint_progress_loss(head: networks.ProgressHead, potential,
                        feat: torch.Tensor, batch: Mapping[str, torch.Tensor]
                        ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Stage 1A's progress term, on the world model's own posterior features.

    ``feat`` must be the attached feature the world-model loss was computed
    from: that is what lets this term train the encoder, the recurrent state
    and the graph branch alongside the head, as the regular trainer does. The
    caller adds ``loss_scales.progress_model`` times the loss to the
    world-model loss before its single backward pass.
    """
    phi, phi_valid = potential.targets(batch)
    return progress_loss(head, feat, phi, progress_mask(batch, phi_valid))


def progress_weight(model_cfg) -> float:
    """``loss_scales.progress_model``: the head's weight in the world-model loss.

    Not ``progress.beta``. Beta weights the shaping reward the actor sees in
    Stage 2; this weights a supervised term in Stage 1A. The two happen to be
    configured near each other and must not be read for one another.
    """
    scales = getattr(model_cfg, "loss_scales", None)
    if scales is None or "progress_model" not in scales:
        raise SystemExit(
            "the model config has no loss_scales.progress_model; the "
            "graph_progress arm trains its head inside the world-model loss "
            "and needs that weight (configs/model/_base_.yaml sets it)")
    weight = float(scales["progress_model"])
    if not weight > 0.0:
        # dreamer.py refuses the same configuration.
        raise SystemExit(
            f"loss_scales.progress_model={weight:g} leaves the progress head "
            "untrained; Stage 2 would shape the actor with an unsupervised "
            "readout")
    return weight


def fit_progress(head: networks.ProgressHead, optimizer, potential,
                 feat: torch.Tensor, batch, *,
                 grad_clip: float = PROGRESS_GRAD_CLIP) -> Dict[str, float]:
    """Stage 2's regression step of the head onto the observed-graph potential.

    Stage 2 keeps the head tracking a world model that is still moving. Same
    objective as Stage 1A's joint term, but ``feat`` is detached here: this
    trains the head and nothing else, which is why it has its own optimizer.

    The targets come from the *recorded* graph labels, not from the decoder's
    predictions, so the head is regressed onto something the dataset actually
    contains. A batch with nothing scorable takes no step.
    """
    phi, phi_valid = potential.targets(batch)
    mask = progress_mask(batch, phi_valid)
    if not bool(mask.any()):
        return {"progress_valid": 0.0}
    loss, metrics = progress_loss(head, feat.detach(), phi, mask)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), float(grad_clip))
    optimizer.step()
    return {"progress_loss": float(loss.detach())} | metrics


def build_progress(config, feature_dim: int, *, graph_enabled: bool,
                   progress_enabled: bool
                   ) -> Optional[networks.ProgressHead]:
    """The regular trainer's head for the graph arm with shaping on, else None.

    ``config`` is the model config; the head is built from its
    ``progress.head`` block, as ``dreamer.py`` builds it.
    """
    if not progress_enabled:
        return None
    if not graph_enabled:
        raise SystemExit(
            "progress shaping requires the graph arm: its targets come from "
            "the graph schedule, and a baseline has no schedule to read.")
    progress = config.get("progress") if hasattr(config, "get") else None
    head_cfg = progress.get("head") if progress is not None else None
    if head_cfg is None:
        raise SystemExit(
            "the model config has no progress.head block; the progress head "
            "is networks.ProgressHead and is configured from it, as in "
            "configs/model/_base_.yaml")
    return networks.ProgressHead(head_cfg, int(feature_dim))


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


def recorded_schedules(cfg) -> list:
    """``[(path, weight)]`` from ``task.progress_schedules``; empty for sim tasks."""
    import os

    parts = list((cfg.get("task") or {}).get("progress_schedules") or [])
    folder = str(((cfg.get("model") or {}).get("progress") or {}).get(
        "schedule_dir") or DEFAULT_SCHEDULE_DIR)
    return [(os.path.join(folder, str(part["schedule"])), float(part["weight"]))
            for part in parts]


class RecordedSchedulePotential:
    """``Phi`` for annotated graphs: schedules compiled against recorded facts.

    There are no mined assets to say which relations a pair can carry, so each
    schedule is compiled against the facts the dataset records
    (``metadata.graph.facts``) and its entity vocabulary. Several schedules are
    summed with their weights; a frame is scorable only where every one is.
    """

    def __init__(self, parts, graph_meta: Mapping[str, Any], n_abs: int, *,
                 env_id: str = "", device=None):
        import json

        from scenegraph.adapters.graph_vocab import EE_TOKEN, PAD_TOKEN, EntityVocab
        from scenegraph.core.schedule import compile_schedule

        from progress import TaskScheduleReplayPotential

        vocab = EntityVocab(token_to_id={str(k): int(v) for k, v in
                                         graph_meta["entity_tokens"].items()})
        scorable: Dict[str, Dict[str, bool]] = {}
        for src, dst, relation in graph_meta["facts"]:
            scorable.setdefault(f"{src} / {dst}", {})[str(relation)] = True
        members = {key: {} for key in vocab.token_to_id
                   if key not in (PAD_TOKEN, EE_TOKEN)}
        self.env_id = str(env_id)
        self.paths = [str(path) for path, _ in parts]
        self.weights = [float(weight) for _, weight in parts]
        self.scorers = []
        self.phases = 0
        for path in self.paths:
            with open(path, encoding="utf-8") as handle:
                raw = json.load(handle)
            schedule = compile_schedule(raw, {}, members, {}, vocab,
                                        scorable=scorable)
            scorer = TaskScheduleReplayPotential(schedule, int(n_abs))
            self.scorers.append(scorer if device is None else scorer.to(device))
            self.phases += len(schedule.phases)

    @torch.no_grad()
    def targets(self, batch):
        """``(phi, valid)`` shaped like the batch's ``(batch, time)``."""
        from graph import compact_graph
        from scenegraph.adapters.graph_pack import GRAPH_KEYS

        packed = {key: batch[key] for key in GRAPH_KEYS}
        shape = tuple(packed["graph_node_ent"].shape[:-1])
        compact = compact_graph(packed)
        phi = valid = None
        for weight, scorer in zip(self.weights, self.scorers):
            part, ok = scorer(
                compact.node_ent, compact.edge_rel, compact.edge_abs,
                compact.edge_src_local, compact.edge_dst_local,
                compact.edge_graph, compact.graph_count)
            phi = weight * part if phi is None else phi + weight * part
            valid = ok if valid is None else valid & ok
        return phi.reshape(shape).float(), valid.reshape(shape).bool()

    def describe(self) -> Dict[str, Any]:
        return {"env_id": self.env_id, "phases": self.phases,
                "schedule": ",".join(self.paths),
                "weights": ",".join(f"{w:g}" for w in self.weights)}


def recorded_availability(cfg, metadata=None) -> list:
    """:func:`availability` for a task that names ``progress_schedules``."""
    import os

    graph_meta = dict((metadata or {}).get("graph") or {})
    missing = []
    if not bool((cfg.get("model") or {}).get("graph", {}).get("enabled")):
        missing.append(
            "model.graph.enabled is false: the schedule resolves roles against "
            "the graph's entity vocabulary, and a baseline has no graph")
    parts = recorded_schedules(cfg)
    for path, weight in parts:
        if not os.path.isfile(path):
            missing.append(f"no task schedule at {path}")
        if not weight > 0:
            missing.append(f"{path} has weight {weight}; weights must be positive")
    total = sum(weight for _, weight in parts)
    if abs(total - 1.0) > 1e-6:
        missing.append(f"task.progress_schedules weights sum to {total:g}, not 1, "
                       "so the potential would not end at 1")
    if not graph_meta.get("facts") or not graph_meta.get("entity_tokens"):
        missing.append(
            "the dataset metadata records no graph facts or entity vocabulary; "
            "prepare it with python -m sim_vla.data.prepare_real")
    if not graph_meta.get("absolute_tokens") and not graph_meta.get(
            "vocab_sizes"):
        missing.append(
            "the dataset metadata records no absolute-token vocabulary, so "
            "the number of spatial bins the scorer needs is unknown")
    return missing


def availability(cfg, metadata=None) -> list:
    """What the progress arm is missing, or an empty list.

    Checked from paths and recorded metadata rather than by trying to build
    the thing, so it can run before Stage 1A.
    """
    import os

    if recorded_schedules(cfg):
        return recorded_availability(cfg, metadata)
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
    parts = recorded_schedules(cfg)
    if parts:
        return RecordedSchedulePotential(
            parts, graph_meta, n_abs, env_id=str(cfg["task"]["env_id"]),
            device=device)
    return SchedulePotential(
        str(cfg["task"]["env_id"]), str(graph_meta.get("whitelist_dir") or ""),
        n_abs,
        schedule_dir=str(progress.get("schedule_dir") or DEFAULT_SCHEDULE_DIR),
        configs_dir=str(progress.get("configs") or DEFAULT_CONFIGS_DIR),
        device=device)
