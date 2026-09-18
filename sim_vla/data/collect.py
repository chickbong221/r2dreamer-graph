"""Collect demonstrations whose labels all come from the same simulated step.

The scripted motion planner drives the task; every control step renders RGB,
builds the scene graph and takes the reward from the *same* ``env.step`` that
produced the action's effect. Nothing is replayed and nothing is restored, and
that is the point.

The obvious cheaper design is to record actions alone and re-render later,
pinning each recorded simulator state before rendering so the pixels match. It
works for pixels, and it quietly does not work for this dataset:

* ``set_state_dict`` restores poses and velocities. It does not re-run
  narrowphase, so the contact impulses the graph reads are still the ones the
  *previous simulated step* left in the buffer -- see the comment at
  ``scenegraph/core/relation_rules.py:936``, where reading a stale buffer is
  named as the thing that produces a phantom touch. A pinned frame would carry
  geometry from the restored state and grasp/contact from a different one.
* The reward and ``info`` a step returns were computed before any restore, so
  on a replay that drifted they describe a state the stored image does not.

Both are worst exactly where the labels matter: the frames around grasp,
release and insertion, where a one-step disagreement flips an edge.

Paying for it: the render and the graph run on rejected attempts too, roughly
1.6 attempts per accepted demo on PegInsertionSide. In exchange there is no
consistency to validate, because there is no second source for the labels to
disagree with.

    python -m sim_vla.data.collect --env-id PickCube-v1 --num-traj 500

What is still cheap is re-rendering RGB at another resolution: images depend on
poses alone, and ``env_states`` is recorded for exactly that. The graph and the
reward are not re-derivable that way, and the dataset's metadata says so rather
than leaving it to be rediscovered.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from scenegraph.adapters.graph_pack import GRAPH_KEYS
from scenegraph.tools.collect_maniskill_interactions import success_flag
# One definition of "the step a demo can be cut at" for the whole repo: the
# offline filter finds it in a recorded success array, this finds it in a live
# one, and a disagreement would mean two different datasets.
from scenegraph.tools.filter_demo_trajectories import settled_step

from .config import DEFAULT_ENV_CONFIG, DEFAULT_MODEL_CONFIG
from .graph_export import GraphJsonlExporter
from .schema import (
    END_SOLVER_ERROR, END_SOLVER_FINISHED, END_SUCCESS_CUT,
    build_metadata, flatten_proprio, privileged_fields, proprio_fields, unbatch,
)
from .writer import DatasetWriter, atomic_replace, field_kinds

DEFAULT_STRIDE_FACTOR = 8


def scalar(value, env_idx: int = 0) -> float:
    arr = unbatch(value, env_idx)
    return float(np.asarray(arr).reshape(-1)[0])


def state_to_numpy(state: Mapping[str, Any], env_idx: int = 0) -> Dict[str, Any]:
    """The simulator state tree, unbatched, as plain arrays."""
    out: Dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, Mapping):
            out[str(key)] = state_to_numpy(value, env_idx)
        else:
            out[str(key)] = np.asarray(unbatch(value, env_idx))
    return out


def stack_states(frames: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Stack a list of state trees into one tree of leading-axis arrays."""
    if not frames:
        return {}
    out: Dict[str, Any] = {}
    for key, value in frames[0].items():
        if isinstance(value, Mapping):
            out[key] = stack_states([f[key] for f in frames])
        else:
            out[key] = np.stack([np.asarray(f[key]) for f in frames])
    return out


class _EpisodeBuffer:
    """One attempt's frames, held until its verdict is known.

    Frames stop being kept past ``frame_cap``, which is the step budget: an
    episode is accepted only if success settles inside it, and the trim is
    capped there too, so no row beyond it can ever be written. The env goes on
    rendering those steps because the solver goes on running, but none of them
    gets a graph built for it, which is the expensive half.
    """

    def __init__(self, env, graphs, vocab, *, camera_keys, proprio_spec,
                 n_max: int, e_max: int, use_target_flag: bool,
                 frame_cap: int, keep_graph_objects: bool = False):
        self.env = env
        self.graphs = graphs
        self.vocab = vocab
        self.camera_keys = dict(camera_keys)
        self.proprio_spec = tuple(proprio_spec)
        self.n_max, self.e_max = int(n_max), int(e_max)
        self.use_target_flag = bool(use_target_flag)
        self.frame_cap = int(frame_cap)
        self.keep_graph_objects = bool(keep_graph_objects)
        self.n_cams = 0
        self.cameras: List[str] = []
        self.proprio_names: List[str] = []
        self.privileged_spec: List[Tuple[str, str]] = []
        self.clear()

    def clear(self) -> None:
        self.images: Dict[str, List[np.ndarray]] = {
            key: [] for key in self.camera_keys.values()}
        self.proprio: List[np.ndarray] = []
        self.packed: List[Dict[str, np.ndarray]] = []
        self.graph_objects: List[Any] = []
        self.states: List[Dict[str, Any]] = []
        # Rebuilt from the spec rather than emptied: the spec is learned once,
        # from the first frame, and survives the per-episode clear. Emptying
        # this to {} instead would leave the second episode appending to keys
        # that no longer exist.
        self.privileged: Dict[str, List[np.ndarray]] = {
            f"{group}.{key}": [] for group, key in self.privileged_spec}
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.terminated: List[bool] = []
        self.truncated: List[bool] = []
        self.success: List[bool] = []
        self.success_at_reset = False
        self.capture_error: Optional[str] = None

    # ------------------------------------------------------------- the hooks
    def on_reset(self, obs: Mapping[str, Any], info: Mapping[str, Any]) -> None:
        self.clear()
        # History is reset with the episode: temporal edge labels difference
        # over the last K frames, and a builder carried across a reset reports
        # a change against the previous episode's scene.
        self.graphs.on_reset()
        self.success_at_reset = success_flag(info) if info else False
        self._capture(obs)

    def on_step(self, action, obs, reward, terminated, truncated, info) -> None:
        self.actions.append(np.asarray(action).reshape(-1).astype(np.float32))
        self.rewards.append(scalar(reward))
        self.terminated.append(bool(scalar(terminated)))
        self.truncated.append(bool(scalar(truncated)))
        self.success.append(bool(success_flag(info)))
        self._capture(obs)

    # --------------------------------------------------------------- capture
    def _capture(self, obs: Mapping[str, Any]) -> None:
        if len(self.proprio) > self.frame_cap or self.capture_error:
            return
        try:
            graph = self.graphs.step(obs)
            if self.n_cams == 0:
                self.cameras = list(self.graphs.cameras)
                self.n_cams = len(self.cameras)
            from scenegraph.adapters.graph_pack import pack_graph

            packed = pack_graph(
                graph, self.vocab, n_max=self.n_max, e_max=self.e_max,
                n_cams=self.n_cams, use_target_flag=self.use_target_flag)
            vector, names = flatten_proprio(obs, self.proprio_spec)
        except Exception as exc:                           # noqa: BLE001
            # Recorded on the episode rather than raised: one unpackable frame
            # should cost that episode, not the run. The attempt is rejected
            # below because a half-captured episode cannot be trimmed honestly.
            self.capture_error = f"{type(exc).__name__}: {exc}"
            return

        if not self.proprio_names:
            self.proprio_names = names
            self.privileged_spec = privileged_fields(obs, self.proprio_spec)
            self.privileged = {f"{g}.{k}": [] for g, k in self.privileged_spec}

        sensors = obs.get("sensor_data") or {}
        for camera, key in self.camera_keys.items():
            self.images[key].append(
                np.asarray(unbatch(sensors[camera]["rgb"]), dtype=np.uint8))
        self.proprio.append(vector)
        self.packed.append(packed)
        if self.keep_graph_objects:
            self.graph_objects.append(graph)
        self.states.append(state_to_numpy(self.env.get_state_dict()))
        for group, key in self.privileged_spec:
            self.privileged[f"{group}.{key}"].append(
                np.asarray(unbatch((obs.get(group) or {})[key])))

    # -------------------------------------------------------------- verdict
    def verdict(self, max_steps: int) -> Dict[str, Any]:
        """Whether this attempt is a demo, and where it would be cut."""
        flags = np.asarray(self.success, dtype=bool)
        settled = settled_step(flags) if flags.size else None
        degenerate = bool(flags.size and flags[0]) or self.success_at_reset
        reason = ""
        if self.capture_error:
            reason = f"capture failed: {self.capture_error}"
        elif settled is None:
            reason = "did not end successful"
        elif degenerate:
            reason = "already successful at the first step"
        elif settled > int(max_steps):
            reason = f"settles at {settled} > {max_steps}"
        return {
            "keep": not reason,
            "reason": reason,
            "settled": settled,
            "recorded_steps": int(len(self.actions)),
        }

    def episode(self, keep_steps: int) -> Dict[str, Any]:
        """The arrays for an accepted episode, trimmed to ``keep_steps``."""
        rows = int(keep_steps) + 1
        graphs = {
            key: np.stack([frame[key] for frame in self.packed[:rows]])
            for key in GRAPH_KEYS
        }
        return {
            "images": {key: np.stack(values[:rows])
                       for key, values in self.images.items()},
            "proprio": np.stack(self.proprio[:rows]),
            "graphs": graphs,
            "actions": np.stack(self.actions[:keep_steps]),
            "rewards": np.asarray(self.rewards[:keep_steps], dtype=np.float32),
            "terminated": np.asarray(self.terminated[:keep_steps], dtype=bool),
            "truncated": np.asarray(self.truncated[:keep_steps], dtype=bool),
            "success": np.asarray(self.success[:keep_steps], dtype=bool),
            "env_states": stack_states(self.states[:rows]),
            "privileged": {key: np.stack(values[:rows])
                           for key, values in self.privileged.items()},
        }


def capture_wrapper(env, buffer: _EpisodeBuffer):
    """Wrap ``env`` so the buffer sees the action as well as the transition.

    ``MotionPlanRunner``'s own hooks carry the observation and the info but not
    the action, and the action is the label. Wrapping below the runner rather
    than widening its hooks keeps the figure exporters on the signature they
    were written against.
    """
    import gymnasium as gym

    class _Capture(gym.Wrapper):
        def reset(self, **kwargs):
            out = self.env.reset(**kwargs)
            buffer.on_reset(out[0], out[1] if len(out) > 1 else {})
            return out

        def step(self, action):
            out = self.env.step(action)
            buffer.on_step(action, *out)
            return out

    return _Capture(env)


def make_env(args) -> Tuple[Any, str]:
    """A single CPU env rendering what the graph needs and the policy sees.

    ``rgb+segmentation`` is not a choice about what to store: the graph builder
    reads segmentation masks, and the dataset keeps RGB only. Batch size one on
    CPU is not a tunable -- the scripted solutions read ``pose.sp``, which
    exists only for an unbatched pose.
    """
    import gymnasium as gym                                # noqa: F401
    import mani_skill.envs                                 # noqa: F401
    from envs.maniskill import _make_with_supported_reward

    sensors: Dict[str, Any] = dict(shader_pack=args.shader)
    if args.sensor_size:
        sensors |= dict(width=int(args.sensor_size[1]),
                        height=int(args.sensor_size[0]))
    kwargs: Dict[str, Any] = dict(
        id=args.env_id,
        obs_mode="rgb+segmentation",
        control_mode=args.control_mode,
        render_mode="rgb_array",
        sensor_configs=sensors,
        sim_backend=args.sim_backend,
        reward_mode=str(args.reward_mode),
        # The budget is the horizon, not just an acceptance filter. Left at the
        # task's registration -- PickCube ends at 50, PegInsertionSide at 100 --
        # the planner keeps stepping past it and every step after carries
        # ``truncated=True``. A 130-step peg demo would then hold a truncation
        # at step 100, and a loader that honours episode boundaries would cut
        # the demonstration in half before the peg is inserted.
        max_episode_steps=int(args.max_steps),
    )
    env = _make_with_supported_reward(kwargs, list(args.reward_fallback or []))
    return env, str(kwargs["reward_mode"])


def _run_one(args: Namespace, proc_id: int, start_seed: int, target: int,
             max_attempts: int) -> Tuple[str, Dict[str, Any]]:
    """One process: sample seeds forward until ``target`` demos are accepted."""
    from mani_skill.utils import gym_utils

    from envs.maniskill import camera_obs_key, rendered_cameras
    from scenegraph.adapters.graph_vocab import build_graph_vocab
    from scenegraph.configs.loader import default_temporal_k
    from scenegraph.figures.graph_source import FigureGraphSource
    from scenegraph.figures.rollout import MotionPlanRunner

    env, reward_mode = make_env(args)
    horizon = gym_utils.find_max_episode_steps_value(env)
    cameras = rendered_cameras(env)
    camera_keys = {camera: camera_obs_key(camera) for camera in cameras}

    graphs = FigureGraphSource(
        env,
        env_id=args.env_id,
        thresholds_path=str(args.thresholds_path or ""),
        whitelist_dir=str(args.whitelist_dir or ""),
        use_target_flag=bool(args.use_target_flag),
        object_object_spatial=bool(args.object_object_spatial),
        visibility_policy=str(args.visibility_policy),
    )
    vocab = build_graph_vocab(graphs.whitelist_dir)
    exporter = GraphJsonlExporter(
        Path(args.out_dir) / args.env_id / "graphs_jsonl",
        sample=(args.graph_sample if proc_id == 0 else 0))
    buffer = _EpisodeBuffer(
        env.unwrapped, graphs, vocab,
        camera_keys=camera_keys,
        proprio_spec=proprio_fields(args.env_id),
        n_max=args.n_max, e_max=args.e_max,
        use_target_flag=bool(args.use_target_flag),
        # Nothing past the budget can ever be saved -- the trim is capped there
        # -- so nothing past it is worth building a graph for.
        frame_cap=int(args.max_steps),
        keep_graph_objects=args.graph_sample > 0 and proc_id == 0,
    )
    runner = MotionPlanRunner(capture_wrapper(env, buffer), args.env_id)

    shard = Path(args.out_dir) / args.env_id / (
        f"{args.name}.h5" if args.num_procs == 1 else f"{args.name}.{proc_id}.h5")
    writer: Optional[DatasetWriter] = None
    kept = attempts = successes = 0
    settled_kept: List[int] = []
    rejected: Dict[str, int] = {}
    rejected_seeds: List[Dict[str, Any]] = []
    seed = int(start_seed)

    try:
        while kept < target and attempts < max_attempts:
            attempts += 1
            attempt = runner.attempt(seed)
            seed += 1
            successes += int(attempt.success)
            verdict = buffer.verdict(args.max_steps)
            if not verdict["keep"]:
                reason = verdict["reason"] or (attempt.error or "unknown")
                rejected[reason] = rejected.get(reason, 0) + 1
                rejected_seeds.append({
                    "seed": int(attempt.seed), "reason": reason,
                    "settled": verdict["settled"],
                    "recorded_steps": verdict["recorded_steps"],
                })
                buffer.clear()
                continue

            # The pad fits in what is left of the budget rather than extending
            # past it: success at exactly ``max_steps`` is still a demo, and it
            # is one of ``max_steps`` actions, not ``max_steps + pad``.
            steps = min(int(verdict["settled"]) + int(args.pad),
                        int(verdict["recorded_steps"]),
                        int(args.max_steps))
            payload = buffer.episode(steps)
            if writer is None:
                # Built from the first accepted episode, because the camera
                # set, the proprio column names and the graph's camera count
                # are all read back from a built env and a packed frame rather
                # than assumed from config.
                writer = DatasetWriter(shard, build_metadata(
                    env=env, env_id=args.env_id,
                    env_kwargs=dict(getattr(env.unwrapped.spec, "kwargs", {}) or {}),
                    reward_mode=reward_mode,
                    reward_fallback=args.reward_fallback,
                    cameras=buffer.cameras or cameras,
                    camera_keys=camera_keys,
                    image_size=args.sensor_size or (0, 0),
                    proprio_names=buffer.proprio_names,
                    proprio_spec=buffer.proprio_spec,
                    privileged_spec=buffer.privileged_spec,
                    graph_cfg=graphs.cfg,
                    vocab=vocab,
                    whitelist_dir=graphs.whitelist_dir,
                    thresholds_path=str(args.thresholds_path or ""),
                    temporal_k=default_temporal_k(
                        str(args.thresholds_path or "") or None),
                    n_max=args.n_max, e_max=args.e_max, n_cams=buffer.n_cams,
                    visibility_policy=args.visibility_policy,
                    use_target_flag=bool(args.use_target_flag),
                    object_object_spatial=bool(args.object_object_spatial),
                    max_steps=args.max_steps, pad=args.pad, horizon=horizon,
                    field_kinds=field_kinds(camera_keys.values(), GRAPH_KEYS),
                    config_source=getattr(args, "config_source", {}),
                ), overwrite=bool(args.overwrite))
            episode_id = writer.add(**payload, info={
                "seed": int(attempt.seed),
                "reset_kwargs": {"seed": int(attempt.seed)},
                "settled_steps": int(verdict["settled"]),
                "recorded_steps": int(verdict["recorded_steps"]),
                "success_at_reset": bool(buffer.success_at_reset),
                "task": str(args.env_id),
                "instruction": str(args.instruction or args.env_id),
                "end_reason": (END_SUCCESS_CUT
                               if steps < int(verdict["recorded_steps"])
                               else (END_SOLVER_ERROR if attempt.error
                                     else END_SOLVER_FINISHED)),
                "solver_error": attempt.error,
            })
            if exporter.wants(kept) and buffer.graph_objects:
                exporter.write(episode_id, buffer.graph_objects[:steps + 1])
            settled_kept.append(int(verdict["settled"]))
            kept += 1
            buffer.clear()
            if kept == target or attempts % args.log_every == 0:
                print(f"[proc {proc_id}] {kept}/{target} kept, {attempts} "
                      f"attempts, {successes} solved, seed at {seed}", flush=True)
    except KeyboardInterrupt:
        print(f"[proc {proc_id}] interrupted at {kept}/{target}", flush=True)
    finally:
        if writer is not None:
            writer.close()
        env.close()

    stats = {
        "proc_id": proc_id, "target": int(target), "kept": kept,
        "attempts": attempts, "successes": successes,
        "first_seed": int(start_seed), "last_seed": seed - 1,
        "exhausted": kept < target, "settled": settled_kept,
        "rejected": rejected, "rejected_seeds": rejected_seeds,
        "shard": str(shard) if writer is not None else "",
    }
    if stats["exhausted"]:
        print(f"[proc {proc_id}] WARNING: seed block exhausted at "
              f"{kept}/{target} after {attempts} attempts", flush=True)
    return stats["shard"], stats


def attach_summary(path: Path, summary: Mapping[str, Any]) -> None:
    """Record how a dataset was selected, into the sidecar it belongs to.

    Which attempts were rejected and why is the one thing about a demonstration
    set that cannot be reconstructed from the set itself: what is in the file
    is the episodes that passed.
    """
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["metadata"] = dict(payload.get("metadata") or {})
    payload["metadata"]["collection"] = dict(summary)
    tmp = sidecar.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    atomic_replace(tmp, sidecar)


def merge_shards(out: Path, shards: Sequence[Path], *,
                 overwrite: bool = False,
                 summary: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """One file per task, from one file per process.

    Episode ids are renumbered because the h5 group name is the id and every
    shard starts at zero. Metadata is taken from the first shard and checked
    against the rest: two shards that disagree about the vocabulary or the
    camera set are not one dataset, and merging them would hide that.
    """
    import h5py

    from .schema import merge_conflicts, sanitize_metadata

    if out.exists() and not overwrite:
        raise SystemExit(
            f"{out} already exists; pass --overwrite to replace it")

    # Every sidecar is read and compared before a byte is written. The earlier
    # version opened the destination first and raised afterwards, which left a
    # half-merged HDF5 beside the previous run's sidecar -- a dataset whose
    # metadata described data that was no longer in it.
    sidecars = [json.loads(shard.with_suffix(".json").read_text(encoding="utf-8"))
                for shard in shards]
    # Sanitised before it is compared or stored: one worker's resolved scene
    # state is not a property of the merged dataset, and keeping it would make
    # the merged file disagree with the seven shards it did not come from.
    metadata = sanitize_metadata(sidecars[0]["metadata"])
    conflicts = {
        shard.name: bad
        for shard, side in zip(shards[1:], sidecars[1:])
        if (bad := merge_conflicts(metadata, side["metadata"]))
    }
    if conflicts:
        raise SystemExit(f"refusing to merge shards that disagree: {conflicts}")

    # Built beside the destination and moved over it, so an interrupted merge
    # leaves the previous dataset intact rather than a truncated one.
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_h5 = out.with_suffix(".h5.tmp")
    tmp_json = out.with_suffix(".json.tmp")
    episodes: List[Dict[str, Any]] = []
    with h5py.File(tmp_h5, "w") as dst:
        for shard, side in zip(shards, sidecars):
            with h5py.File(shard, "r") as src:
                for entry in side["episodes"]:
                    new_id = len(episodes)
                    src.copy(f"traj_{entry['episode_id']}", dst, f"traj_{new_id}")
                    episodes.append(dict(entry) | {"episode_id": new_id})
    tmp_json.write_text(json.dumps(
        {"metadata": metadata | {"collection": summary or {}},
         "episodes": episodes, "count": len(episodes)},
        indent=2, default=str), encoding="utf-8")
    os.replace(tmp_h5, out)
    atomic_replace(tmp_json, out.with_suffix(".json"))
    return {"episodes": len(episodes)}


def split_targets(total: int, procs: int) -> List[int]:
    """Per-process targets that add up to exactly ``total``.

    Rounding every process up overshoots: 500 across eight workers becomes
    8 x 63 = 504. The remainder is handed to the first few instead, so the run
    collects the number that was asked for.
    """
    base, extra = divmod(int(total), int(procs))
    return [base + (1 if i < extra else 0) for i in range(int(procs))]


def collect(args) -> int:
    procs = max(int(args.num_procs), 1)
    targets = split_targets(int(args.num_traj), procs)
    per_proc = max(targets)
    stride = int(args.seed_stride) or (per_proc * DEFAULT_STRIDE_FACTOR)
    # The attempt cap is the block width, so a process cannot reach the next
    # process's seeds however badly the solver does.
    starts = [int(args.start_seed) + i * stride for i in range(procs)]
    print(f"[collect] {args.env_id}: {sum(targets)} demos across {procs} "
          f"processes {targets}, <= {args.max_steps} steps (+{args.pad} pad, "
          f"capped at the budget), seed blocks "
          f"{starts[0]}..{starts[-1] + stride - 1}", flush=True)

    jobs = [(args, i, starts[i], targets[i], stride) for i in range(procs)]
    results = ([_run_one(*jobs[0])] if procs == 1
               else _parallel(jobs, procs))
    shards = [Path(path) for path, _ in results if path]
    stats = [entry for _, entry in results]
    if not shards:
        print("[collect] nothing was written", flush=True)
        return 1

    settled = sorted(s for entry in stats for s in entry["settled"])
    attempts = sum(e["attempts"] for e in stats)
    kept = sum(e["kept"] for e in stats)
    successes = sum(e["successes"] for e in stats)
    seeds = [e["first_seed"] for e in stats]
    rejected: Dict[str, int] = {}
    for entry in stats:
        for reason, count in (entry["rejected"] or {}).items():
            rejected[reason] = rejected.get(reason, 0) + count
    # Kept in the dataset, not only on the terminal. What fraction of solved
    # episodes missed the budget, and which seeds they were, is the record of
    # how this dataset was selected -- and selection is the part of a
    # demonstration set that a later reader cannot reconstruct.
    summary = {
        "attempts": attempts,
        "successes": successes,
        "kept": kept,
        "solve_rate": successes / max(attempts, 1),
        "budget_yield": kept / max(successes, 1),
        "attempts_per_demo": attempts / max(kept, 1),
        "rejected_counts": rejected,
        "rejected_seeds": [row for entry in stats
                           for row in (entry.get("rejected_seeds") or [])],
        "settled_steps": settled,
        "per_process": [{k: v for k, v in e.items() if k != "settled"}
                        for e in stats],
    }

    out = Path(args.out_dir) / args.env_id / f"{args.name}.h5"
    if len(shards) > 1 or shards[0] != out:
        merge_shards(out, shards, overwrite=bool(args.overwrite),
                     summary=summary)
        for shard in shards:
            if shard != out:
                os.remove(shard)
                os.remove(shard.with_suffix(".json"))
    else:
        # A single process writes the final file directly, so there is no merge
        # to fold the summary into. Without this the selection record -- how
        # many solved episodes missed the budget, and which seeds -- would
        # exist only on the terminal for exactly the runs easiest to repeat.
        attach_summary(out, summary)

    print(f"\n[collect] wrote {kept} demos to {out}")
    print(f"[collect] {attempts} attempts -> {successes} solved "
          f"({successes / max(attempts, 1):.0%}) -> {kept} within "
          f"{args.max_steps} steps ({kept / max(successes, 1):.0%} of solved), "
          f"{attempts / max(kept, 1):.1f} attempts per demo")
    if settled:
        print(f"[collect] settled steps: min={settled[0]} "
              f"med={settled[len(settled) // 2]} max={settled[-1]}")
    for reason, count in sorted(rejected.items(), key=lambda kv: -kv[1])[:6]:
        print(f"[collect]   rejected {count}: {reason}")
    short = [e["proc_id"] for e in stats if e["exhausted"]]
    if short:
        # A different --name, not the same one: the top-up writes its own file
        # and the two are merged, so a recovery run can never land on top of
        # the demos that were already collected.
        print(f"[collect] WARNING: processes {short} ran out of seeds; "
              f"{int(args.num_traj) - kept} short. Top up with "
              f"--start-seed {max(seeds) + stride} "
              f"--num-traj {max(int(args.num_traj) - kept, 1)} "
              f"--name {args.name}_topup, then merge the two files.",
              flush=True)
        return 1
    return 0


def _parallel(jobs, procs):
    pool = mp.Pool(procs)
    try:
        return pool.starmap(_run_one, jobs)
    finally:
        pool.close()
        pool.join()


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Collect scripted demonstrations with RGB, proprioception, "
                    "packed scene graphs, rewards and actions from one live "
                    "rollout")
    p.add_argument("--env-id", default="PickCube-v1")
    p.add_argument("--num-traj", type=int, default=500,
                   help="demos to accept, not episodes to record")
    p.add_argument("--max-steps", type=int, default=150,
                   help="control steps a demo may take to reach the success "
                        "that holds to the end")
    p.add_argument("--pad", type=int, default=5,
                   help="steps kept after success settles")
    p.add_argument("--num-procs", type=int, default=8)
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--seed-stride", type=int, default=0)
    p.add_argument("--out-dir", default="data/sim_vla_demos")
    p.add_argument("--name", default="demos")
    p.add_argument("--instruction", default="",
                   help="task instruction stored per episode; the gym id by "
                        "default")

    p.add_argument("--overwrite", action="store_true",
                   help="replace an existing dataset of this name instead of "
                        "refusing; without it a repeated command is safe")
    p.add_argument("--control-mode", default="pd_joint_pos",
                   help="the scripted solutions are written for pd_joint_pos")
    p.add_argument("--sim-backend", default="cpu")
    p.add_argument("--reward-mode", default="normalized_dense")
    p.add_argument("--reward-fallback", nargs="*", default=["sparse"])

    # Everything below defaults to None and is filled from the training config
    # -- see sim_vla.data.config. A flag given explicitly still wins, and what
    # was taken from the config is recorded in the dataset's metadata.
    p.add_argument("--env-config", default=DEFAULT_ENV_CONFIG)
    p.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    p.add_argument("--shader", default=None,
                   help="default: env config's shader_dir")
    p.add_argument("--sensor-size", type=int, nargs=2, default=None,
                   metavar=("H", "W"), help="default: env config's size")
    p.add_argument("--n-max", type=int, default=None,
                   help="graph node capacity; default: model config")
    p.add_argument("--e-max", type=int, default=None,
                   help="graph edge capacity; default: model config")
    p.add_argument("--visibility-policy", default=None)
    p.add_argument("--use-target-flag", action="store_const", const=True,
                   default=None)
    p.add_argument("--no-object-object-spatial", dest="object_object_spatial",
                   action="store_const", const=False, default=None)
    p.add_argument("--thresholds-path", default=None)
    p.add_argument("--whitelist-dir", default=None)
    p.add_argument("--graph-sample", type=int, default=3,
                   help="episodes whose graphs are also written as JSONL")
    p.add_argument("--log-every", type=int, default=25)
    return p.parse_args(argv)


# Fallbacks for anything the training config does not supply. They match what
# configs/env/maniskill.yaml and configs/model/size50M_graph_simple.yaml hold
# today; the config is what makes them stay matched.
_FALLBACKS = {
    "shader": "minimal",
    "sensor_size": [112, 112],
    "visibility_policy": "keep_tabletop",
    "n_max": 8,
    "e_max": 168,
    "use_target_flag": False,
    "object_object_spatial": True,
    "thresholds_path": "",
    "whitelist_dir": "",
}


def resolve_settings(args) -> Namespace:
    """Fill the unset arguments from the training config, then from defaults."""
    from .config import apply_defaults

    source = apply_defaults(args)
    defaulted = []
    for key, value in _FALLBACKS.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
            defaulted.append(key)
    args.config_source = dict(source) | {"fell_back_to_builtin": defaulted}
    taken = source.get("from_config") or {}
    print(f"[collect] from {args.env_config} + {args.model_config}: "
          f"{taken or 'nothing'}"
          + (f"; built-in defaults for {defaulted}" if defaulted else ""),
          flush=True)
    return args


def main(argv=None) -> int:
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    return collect(resolve_settings(parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
