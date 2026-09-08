"""Drive the scripted solution and keep every frame's packed graph.

The simulator runs exactly once per dataset. What it produces is the encoder's
own input -- nine arrays per frame -- and nothing else; RGB and simulator state
are the expensive half and the probe reads neither.

Two things the collector must not get wrong. The graph is built on *every*
control step, because temporal labels difference over the last K frames and a
builder fed one step in five would report a change spanning five times the
horizon it was mined for. And history is reset at each episode boundary, or the
first frames of an episode carry deltas measured against the previous one.

Failed attempts contribute their frames. The scripted planner fails on some
seeds, and a half-completed approach is a real observation of the scene -- it is
only the *success* label that is missing, and the probe never reads it.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from scenegraph.adapters.graph_pack import GRAPH_KEYS, pack_graph
from scenegraph.adapters.graph_vocab import GraphVocab, build_graph_vocab
from scenegraph.figures.graph_source import FigureGraphSource
from scenegraph.figures.render_camera import make_figure_env
from scenegraph.figures.rollout import MotionPlanRunner

from .dataset import ShardWriter, git_revision, save_packed_sample

# The human-render camera is created by ``make_figure_env`` and never read here.
# Kept tiny rather than removed: the figure path owns that function and a fork
# of it would be a second env builder to keep in step.
_UNUSED_RENDER_SIZE = (128, 128)

# Frames kept in memory exactly as the packer emitted them, written beside the
# shards. Without them the round-trip check can only compare the cache against
# another copy of itself.
_SAMPLE_LIMIT = 32


@dataclass
class CollectSummary:
    """What one collection produced, and what was wrong with it."""

    env_id: str
    attempts: int = 0
    successes: int = 0
    episodes: int = 0
    frames: int = 0
    unique_frames: int = 0
    steps_seen: int = 0
    pack_errors: Counter = field(default_factory=Counter)
    nodes_dropped: int = 0
    edges_dropped: int = 0
    frames_without_target: int = 0
    relation_counts: Counter = field(default_factory=Counter)
    absolute_counts: Counter = field(default_factory=Counter)
    temporal_counts: Counter = field(default_factory=Counter)
    labels_per_relation: dict = field(default_factory=dict)
    attempt_log: list = field(default_factory=list)
    n_cams: int = 0
    cameras: list = field(default_factory=list)
    whitelist_dir: str = ""
    vocab_sizes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = {
            key: value
            for key, value in self.__dict__.items()
            if not isinstance(value, Counter)
        }
        payload |= {
            "pack_errors": dict(self.pack_errors),
            "relation_counts": dict(self.relation_counts),
            "absolute_counts": dict(self.absolute_counts),
            "temporal_counts": dict(self.temporal_counts),
        }
        payload["labels_per_relation"] = {
            key: sorted(value) for key, value in self.labels_per_relation.items()
        }
        return payload

    def report(self) -> str:
        """The end-of-collection summary. Short, and never silent about a drop."""
        lines = [
            f"[collect] {self.env_id}: {self.attempts} attempts "
            f"({self.successes} succeeded), {self.episodes} episodes, "
            f"{self.frames} graphs kept from {self.steps_seen} control steps",
            f"[collect] {self.unique_frames} distinct packed graphs "
            f"({self.unique_frames / max(self.frames, 1):.1%} of what was kept) -- "
            "stationary frames repeat, and a large count of near-identical ones "
            "is not variety",
            f"[collect] cameras {self.cameras} (n_cams={self.n_cams}), "
            f"vocab {self.vocab_sizes}",
        ]
        for name, counts in (
            ("absolute", self.absolute_counts),
            ("temporal", self.temporal_counts),
        ):
            observed = ", ".join(
                f"{label} {count}" for label, count in sorted(counts.items())
            )
            lines.append(f"[collect] {name} labels observed: {observed or '(none)'}")
        thin = sorted(
            relation for relation, labels in self.labels_per_relation.items()
            if len(labels) < 2
        )
        if thin:
            lines.append(
                f"[collect] WARNING: {thin} never changed absolute label. The "
                "absolute and assignment edits still have legal alternatives to "
                "write, but this task exercised only one state of these "
                "relations -- extend collection if a pair needs to be about them."
            )
        if self.pack_errors:
            lines.append(f"[collect] WARNING: frames rejected by the packer: {dict(self.pack_errors)}")
        if self.nodes_dropped or self.edges_dropped:
            lines.append(
                f"[collect] WARNING: capacity drops -- {self.nodes_dropped} nodes, "
                f"{self.edges_dropped} edges. Raise n_max/e_max: a dropped fact is "
                "a graph the encoder never saw."
            )
        if self.frames_without_target:
            lines.append(
                f"[collect] WARNING: {self.frames_without_target} frames carried no "
                "resolved target while use_target_flag was on"
            )
        return "\n".join(lines)


class _Session:
    """Per-episode buffer, wired into the runner's two hooks.

    Frames are held until the attempt ends because success is only known then,
    and every frame of an episode has to carry the same verdict.
    """

    def __init__(
        self,
        graphs: FigureGraphSource,
        vocab: GraphVocab,
        writer: ShardWriter,
        summary: CollectSummary,
        *,
        n_max: int,
        e_max: int,
        use_target_flag: bool,
        max_frames: int,
    ):
        self.graphs = graphs
        self.vocab = vocab
        self.writer = writer
        self.summary = summary
        self.n_max = int(n_max)
        self.e_max = int(e_max)
        self.use_target_flag = bool(use_target_flag)
        self.max_frames = int(max_frames)
        self.seed = 0
        self.episode = -1
        self.capped = False
        # (packed, frame, meta counters). The label histograms are computed at
        # commit, not at capture: an attempt whose frames are thrown away must
        # not appear in the coverage the summary reports.
        self._buffer: list[tuple[dict, int, dict]] = []
        self._names = {
            "relation": {i: t for t, i in vocab.relation.token_to_id.items()},
            "absolute": {i: t for t, i in vocab.absolute.token_to_id.items()},
            "temporal": {i: t for t, i in vocab.temporal.token_to_id.items()},
        }
        self._hashes: set[str] = set()
        # (cache row, packed arrays) straight from the packer, for the
        # round-trip check.
        self.samples: list[tuple[int, dict]] = []

    # ------------------------------------------------------------- the hooks
    def on_reset(self, obs: dict) -> None:
        # A solver that resets twice inside one attempt starts a second episode
        # here rather than appending to the first one's history.
        self.discard()
        self.graphs.on_reset()
        self.episode += 1
        self._capture(obs)

    def on_step(self, obs: dict, _info: dict) -> None:
        self.summary.steps_seen += 1
        self._capture(obs)

    # --------------------------------------------------------------- capture
    def _capture(self, obs: dict) -> None:
        if self.capped:
            return
        graph = self.graphs.step(obs)
        if self.summary.n_cams == 0:
            self.summary.cameras = list(self.graphs.cameras)
            self.summary.n_cams = len(self.summary.cameras)
        try:
            packed = pack_graph(
                graph,
                self.vocab,
                n_max=self.n_max,
                e_max=self.e_max,
                n_cams=self.summary.n_cams,
                use_target_flag=self.use_target_flag,
            )
        except Exception as exc:                           # noqa: BLE001
            # Counted, not swallowed: a capacity overflow or an unencodable key
            # is a graph the dataset is missing, and the summary has to say so.
            self.summary.pack_errors[f"{type(exc).__name__}: {str(exc)[:120]}"] += 1
            return
        counters = {
            "nodes_dropped": int(graph.meta.get("n_nodes_dropped", 0) or 0),
            "edges_dropped": int(graph.meta.get("n_edges_dropped", 0) or 0),
            "target_packed": bool(graph.meta.get("target_packed", False)),
        }
        self._buffer.append((packed, int(graph.frame), counters))
        if self.max_frames and self.writer.frames + len(self._buffer) >= self.max_frames:
            self.capped = True

    def _account(self, counters: Mapping[str, object], packed: Mapping[str, np.ndarray]) -> None:
        self.summary.nodes_dropped += int(counters["nodes_dropped"])
        self.summary.edges_dropped += int(counters["edges_dropped"])
        if self.use_target_flag and not counters["target_packed"]:
            self.summary.frames_without_target += 1

        rel = np.asarray(packed["graph_edge_rel"])
        real = rel != 0
        for rid, sid, tid in zip(
            rel[real],
            np.asarray(packed["graph_edge_abs"])[real],
            np.asarray(packed["graph_edge_temp"])[real],
        ):
            relation = self._names["relation"][int(rid)]
            label = self._names["absolute"][int(sid)]
            self.summary.relation_counts[relation] += 1
            self.summary.absolute_counts[label] += 1
            self.summary.labels_per_relation.setdefault(relation, set()).add(label)
            if int(tid):
                self.summary.temporal_counts[self._names["temporal"][int(tid)]] += 1

        digest = hashlib.sha1()
        for key in GRAPH_KEYS:
            digest.update(np.ascontiguousarray(packed[key]).tobytes())
        self._hashes.add(digest.hexdigest())

    # ------------------------------------------------------------- lifecycle
    def discard(self) -> None:
        self._buffer.clear()

    def commit(self, success: bool) -> int:
        """Write the attempt's frames, all carrying its verdict."""
        if not self._buffer:
            return 0
        for packed, frame, counters in self._buffer:
            self._account(counters, packed)
            if len(self.samples) < _SAMPLE_LIMIT:
                # The row this frame lands on in the cache, recorded before the
                # write that creates it.
                self.samples.append((int(self.writer.frames), packed))
            self.writer.add(
                packed,
                {
                    "episode": self.episode,
                    "seed": self.seed,
                    "frame": frame,
                    "success": bool(success),
                },
            )
        written = len(self._buffer)
        self._buffer.clear()
        self.summary.episodes += 1
        self.summary.frames += written
        self.summary.unique_frames = len(self._hashes)
        return written


def collect(cfg: Mapping, out_dir: str) -> CollectSummary:
    """Run the collection and write the cache. Returns its summary."""
    env_id = str(cfg["env_id"])
    env = make_figure_env(
        env_id,
        render_size=_UNUSED_RENDER_SIZE,
        sensor_size=tuple(int(v) for v in cfg["sensor_size"]),
        control_mode=str(cfg["control_mode"]),
    )
    graphs = FigureGraphSource(
        env,
        env_id=env_id,
        cameras=list(cfg.get("cameras") or []) or None,
        thresholds_path=str(cfg.get("thresholds_path") or ""),
        whitelist_dir=str(cfg.get("whitelist_dir") or ""),
        use_target_flag=bool(cfg.get("use_target_flag", False)),
        object_object_spatial=bool(cfg.get("object_object_spatial", True)),
        visibility_policy=str(cfg.get("visibility_policy", "keep_tabletop")),
    )
    vocab = build_graph_vocab(graphs.whitelist_dir)
    summary = CollectSummary(env_id=env_id, whitelist_dir=graphs.whitelist_dir)
    summary.vocab_sizes = dict(vocab.sizes)

    writer = ShardWriter(out_dir, int(cfg.get("shard_size", 4096)))
    session = _Session(
        graphs, vocab, writer, summary,
        n_max=int(cfg["n_max"]),
        e_max=int(cfg["e_max"]),
        use_target_flag=bool(cfg.get("use_target_flag", False)),
        max_frames=int(cfg.get("max_frames", 0) or 0),
    )
    runner = MotionPlanRunner(env, env_id, on_reset=session.on_reset, on_step=session.on_step)

    keep_failed = bool(cfg.get("keep_failed_episodes", True))
    first_seed = int(cfg["first_seed"])
    episodes = int(cfg["episodes"])
    try:
        for i in range(episodes):
            seed = first_seed + i
            session.seed = seed
            attempt = runner.attempt(seed)
            summary.attempts += 1
            summary.successes += int(attempt.success)
            written = (
                session.commit(attempt.success)
                if (attempt.success or keep_failed)
                else (session.discard() or 0)
            )
            summary.attempt_log.append(attempt.to_dict() | {"frames": int(written)})
            note = f" ({attempt.error})" if attempt.error else ""
            print(
                f"[collect] {i + 1}/{episodes} seed={seed} success={attempt.success} "
                f"steps={attempt.steps} frames={written} total={summary.frames}{note}",
                flush=True,
            )
            if session.capped:
                print(f"[collect] frame cap {cfg.get('max_frames')} reached", flush=True)
                break
    except KeyboardInterrupt:
        session.discard()
        print("[collect] interrupted; the in-flight episode was dropped", flush=True)
    finally:
        session.discard()
        env.close()

    meta = {
        "collect": {
            key: cfg[key] for key in sorted(cfg) if key not in ("out_dir",)
        },
        "env_id": env_id,
        "cameras": summary.cameras,
        "n_cams": summary.n_cams,
        "n_max": int(cfg["n_max"]),
        "e_max": int(cfg["e_max"]),
        "whitelist_dir": graphs.whitelist_dir,
        "entity_tokens": dict(vocab.entity.token_to_id),
        "relation_tokens": dict(vocab.relation.token_to_id),
        "absolute_tokens": dict(vocab.absolute.token_to_id),
        "temporal_tokens": dict(vocab.temporal.token_to_id),
        "vocab_sizes": dict(vocab.sizes),
        "seeds": [first_seed + i for i in range(summary.attempts)],
        "revision": git_revision(),
        "summary": summary.to_dict(),
    }
    writer.close(meta)
    save_packed_sample(out_dir, session.samples)
    print(summary.report(), flush=True)
    if summary.frames == 0:
        raise RuntimeError(
            f"collection produced no graphs for {env_id}. Check that the scripted "
            "solution runs on this install and that the whitelist covers the scene."
        )
    return summary
