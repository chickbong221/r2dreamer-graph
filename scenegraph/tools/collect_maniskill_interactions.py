"""Success-gated interaction collection for normal ManiSkill tasks.

Drives ManiSkill's own motion-planning solutions and records what the robot
actually did, keeping only episodes that succeeded. No task registry: relation
buckets are discovered from the scene and from physics.

Single-environment by construction -- the official solutions read ``pose.sp``,
which requires batch size one, so the sim runs on CPU. Parallelism is multiple
processes writing separate shards.

Two stages. Discovery runs until no new bucket appears for ``--patience``
successful episodes; collection then fills the frozen set to ``--target``.

    python -m scenegraph.tools.collect_maniskill_interactions \
        --env-id PickCube-v1 --target 300 --out data/maniskill_evidence
"""

from __future__ import annotations

import argparse
import itertools
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from scenegraph.adapters.interaction_events import (
    EE_KEY, GROUP_GAP, GROUP_WINDOW, BucketStore, DiscoveryWindow,
    BinStats, EpisodeEvidence, GroupAccumulator, InteractionEvent,
    make_bucket,
)
from scenegraph.adapters.contact_geometry import (
    paired_contact_frame, symmetry_of,
)
from scenegraph.adapters.maniskill_containment import (
    containment_features, detect_capability,
)
from scenegraph.adapters.maniskill_scene import scene_entities
from scenegraph.adapters.privileged_state import (
    entity_pose_world_array, get_privileged_state, invalidate_scene_caches,
    set_merged_view_aliasing,
)
from scenegraph.core.entity_identity import stable_entity_key

SCHEMA_VERSION = 1


def _pose(entity, env_idx=0):
    arr = entity_pose_world_array(entity, env_idx)
    return None if arr is None else np.asarray(arr, dtype=float).tolist()


def _list(value):
    return None if value is None else np.asarray(value, dtype=float).tolist()


def success_flag(info, env_idx: int = 0) -> bool:
    value = info.get("success") if isinstance(info, dict) else None
    if value is None:
        return False
    arr = np.asarray(value.cpu() if hasattr(value, "cpu") else value)
    if arr.ndim == 0:
        return bool(arr)
    return bool(arr.reshape(-1)[min(env_idx, arr.size - 1)])


class InteractionRecorder:
    """Per-step relation detection behind an episode success gate.

    Emits one event per (bucket, step). Grouping those into one sample per
    physical interaction is the accumulators' job, not this class's.
    """

    def __init__(self, env, *, eps_force=0.05, min_vertical_ratio=0.5,
                 grasp_angle=30, env_idx=0, group_gap=GROUP_GAP,
                 group_window=GROUP_WINDOW):
        self.env = env
        self.env_idx = int(env_idx)
        self.eps_force = float(eps_force)
        self.min_vertical_ratio = float(min_vertical_ratio)
        self.grasp_angle = int(grasp_angle)
        self.episode = EpisodeEvidence()
        # Contact and support reduce around peak force; a grasp interval keeps
        # its last positive frame, which is the pose at release.
        self.groups = {
            "contact": GroupAccumulator("peak", group_gap, group_window),
            "support": GroupAccumulator("peak", group_gap, group_window),
            "grasp": GroupAccumulator("last", group_gap, group_window),
            "contain": GroupAccumulator("peak", group_gap, group_window),
        }
        self.bins = BinStats()
        self.capability: Optional[str] = None
        self.symmetry: Dict[str, Any] = {}
        self.entities: Optional[List[Any]] = None
        self.keys: Dict[int, str] = {}
        self.frame = 0

    def reset_episode(self) -> None:
        self.episode.reset()
        self.bins.new_episode()
        for acc in self.groups.values():
            acc.open.clear()
        self.frame = 0
        self.entities = None

    def finalize_episode(self) -> None:
        """Close groups still open when the episode ended -- a grasp held
        through success has no release frame to wait for."""
        for acc in self.groups.values():
            for event in acc.close_all():
                self.episode.add(event)

    def on_env_reset(self) -> None:
        """Called after every ``env.reset``, including the one inside solve().

        Tasks that randomize geometry reconfigure on each reset at num_envs=1
        (PegInsertionSide sets reconfiguration_freq=1), which destroys every
        actor captured earlier and drops the scene-level aliasing flag. Both
        have to be re-established here, not once at startup.
        """
        set_merged_view_aliasing(self.env, True)
        invalidate_scene_caches(self.env)
        self.entities = None

    def _capture(self) -> None:
        self.entities = scene_entities(self.env, self.env_idx)
        self.keys = {id(e): stable_entity_key(e) for e in self.entities}
        # Detected after reset: PlugCharger assigns goal_pose during episode
        # initialization, so it does not exist before one.
        self.capability = detect_capability(self.env)
        self.symmetry = {
            self.keys[id(e)]: symmetry_of(e, self.env_idx)
            for e in self.entities
        }

    def observe(self, info: Optional[dict] = None, state: Any = None) -> None:
        """Record one control step. Called once per ``env.step``.

        ``state`` is injectable so the detection rules can be tested without a
        simulator.
        """
        if info is not None:
            self.episode.observe_success(success_flag(info, self.env_idx))
        if self.entities is None:
            self._capture()
        if state is None:
            state = get_privileged_state(self.env, self.env_idx)
        self._ee_relations(state)
        self._object_relations(state)
        self._containment()
        self._spatial_stats(state)
        for acc in self.groups.values():
            for event in acc.tick(self.frame):
                self.episode.add(event)
        self.frame += 1

    @property
    def scene(self):
        base = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
        return base.scene

    def _row(self, entity) -> int:
        from scenegraph.adapters.privileged_state import _obj_index_for_env
        return _obj_index_for_env(entity, self.env_idx) or 0

    def _entity_for(self, key):
        for ent in self.entities or ():
            if self.keys.get(id(ent)) == key:
                return ent
        return None

    def _add(self, relation, src, dst, payload):
        self.groups[relation].observe(
            make_bucket(relation, src, dst), self.frame, payload)

    def _ee_relations(self, state) -> None:
        tcp = _list(state.tcp_pose_world)
        for ent in self.entities:
            key = self.keys[id(ent)]
            force = float(state.ee_object_contact_force(ent))
            grasped = state.is_grasping(ent, max_angle=self.grasp_angle)
            if force <= self.eps_force and not grasped:
                continue
            payload = {
                "force": force,
                "tcp_pose": tcp,
                "obj_pose": _pose(ent, self.env_idx),
                "gripper_width": state.gripper_width,
            }
            if force > self.eps_force:
                self._add("contact", EE_KEY, key, payload)
            if grasped:
                self._add("grasp", EE_KEY, key, dict(payload))

    def _object_relations(self, state) -> None:
        """Every unordered object pair. These scenes hold few objects, so the
        one-hop restriction MS-HAB needs for ReplicaCAD is unnecessary."""
        for a, b in itertools.combinations(self.entities, 2):
            vec = np.asarray(state.pairwise_force_vector(a, b), dtype=float)
            force = float(np.linalg.norm(vec))
            if force <= self.eps_force:
                continue
            ka, kb = self.keys[id(a)], self.keys[id(b)]
            pose_a, pose_b = _pose(a, self.env_idx), _pose(b, self.env_idx)
            payload = {
                "force": force,
                "force_vector": vec.tolist(),
                "key_a": ka, "key_b": kb,
                "pose_a": pose_a,
                "pose_b": pose_b,
            }
            geometry = paired_contact_frame(
                self.scene, a, b, pose_a, pose_b,
                self._row(a), self._row(b))
            if geometry:
                payload.update(geometry)
            self._add("contact", ka, kb, payload)

            # "force on a due to b": fz < 0 means a carries b.
            if abs(float(vec[2])) / force < self.min_vertical_ratio:
                continue
            if float(vec[2]) < 0.0:
                self._add("support", ka, kb, dict(payload))
            else:
                self._add("support", kb, ka, dict(payload))


    def _spatial_stats(self, state) -> None:
        """Relation bin edges come from the range a run actually spans, so
        this samples every pair every step -- not only when a predicate fires,
        which would leave every bin describing contact distance alone."""
        poses = {}
        for ent in self.entities:
            pose = _pose(ent, self.env_idx)
            if pose is not None:
                poses[self.keys[id(ent)]] = pose
        tcp = _list(state.tcp_pose_world)
        if tcp is not None:
            poses[EE_KEY] = tcp
        self.bins.observe(poses, self.frame)

    def _containment(self) -> None:
        """The one relation physics cannot find: a hole is not an actor."""
        if self.capability is None:
            return
        feat = containment_features(self.env, self.env_idx, self.capability)
        if feat is None or not feat.get("holds"):
            return
        payload = dict(feat)
        payload["force"] = 1.0      # peak-mode reducer needs a magnitude
        for role, key in (("container", feat["container_key"]),
                          ("containee", feat["containee_key"])):
            ent = self._entity_for(key)
            if ent is not None:
                payload[f"{role}_pose"] = _pose(ent, self.env_idx)
        self._add("contain", feat["container_key"], feat["containee_key"],
                  payload)


def make_env(env_id: str, recorder_box: list):
    """Env wrapped so every control step reaches the recorder.

    The solutions call ``planner.env.step``, and the planner is constructed
    with whatever ``solve`` was handed, so wrapping here is enough.
    """
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    class _Wrapper(gym.Wrapper):
        def reset(self, **kwargs):
            out = self.env.reset(**kwargs)
            if recorder_box:
                recorder_box[0].on_env_reset()
            return out

        def step(self, action):
            out = self.env.step(action)
            if recorder_box:
                recorder_box[0].observe(out[4])
            return out

    env = gym.make(env_id, obs_mode="none", control_mode="pd_joint_pos",
                   render_mode="rgb_array", sim_backend="cpu")
    return _Wrapper(env)


def get_solver(env_id: str):
    from mani_skill.examples.motionplanning.panda.run import MP_SOLUTIONS

    if env_id not in MP_SOLUTIONS:
        raise SystemExit(
            f"no motion-planning solution for {env_id}; "
            f"have {sorted(MP_SOLUTIONS)}"
        )
    return MP_SOLUTIONS[env_id]


def collect(args):
    box: list = []
    env = make_env(args.env_id, box)
    solve = get_solver(args.env_id)
    env.reset(seed=args.seed)
    set_merged_view_aliasing(env, True)

    recorder = InteractionRecorder(
        env, eps_force=args.eps_force,
        min_vertical_ratio=args.min_vertical_ratio,
        group_gap=args.group_gap, group_window=args.group_window)
    box.append(recorder)

    store = BucketStore(target=args.target)
    window = DiscoveryWindow(patience=args.patience)
    seed = args.seed
    attempts = successes = 0
    started = time.time()

    while attempts < args.max_attempts:
        attempts += 1
        recorder.reset_episode()
        try:
            solve(env, seed=seed, debug=False, vis=False)
        except Exception as exc:                       # noqa: BLE001
            print(f"[warn] seed {seed}: solver failed: {exc}", flush=True)
        seed += 1

        recorder.finalize_episode()
        succeeded = recorder.episode.success_once
        buckets = ({e.bucket for e in recorder.episode.events}
                   if succeeded else set())
        committed = recorder.episode.commit(store)
        if succeeded:
            successes += 1
            if store.frozen is None:
                window.observe(buckets)
                if window.settled:
                    store.freeze(args.min_presence)
                    print(f"[prep] discovery settled after {successes} "
                          f"successes: {len(store.frozen)} buckets kept, "
                          f"{len(store.excluded)} incidental",
                          flush=True)
        if attempts % args.log_every == 0:
            rate = successes / max(attempts, 1)
            print(f"[{attempts}] success={successes} ({rate:.0%}) "
                  f"buckets={len(store.buckets())} "
                  f"committed={committed} "
                  f"{time.time() - started:.0f}s", flush=True)
        if store.frozen is not None and store.is_done():
            print("[prep] every frozen bucket reached target", flush=True)
            break

    # Read before close(): both describe the scene, which close() tears down.
    symmetry = dict(recorder.symmetry or {})
    capability = recorder.capability
    bin_stats = recorder.bins.as_dict()
    env.close()
    print(f"\nattempts={attempts} successes={successes} "
          f"rate={successes / max(attempts, 1):.1%}")
    print(store.report())
    if capability:
        print(f"containment capability: {capability}")
    for key, sym in sorted(symmetry.items()):
        if sym.get("symmetry") != "none":
            print(f"symmetry: {key} -> {sym}")
    incidental = store.incidental(args.min_presence)
    if incidental:
        print()
        print(f"incidental (< {args.min_presence:.0%} of episodes) "
              "-- likely brushes, not task interactions:")
        for b in incidental:
            print(f"  {store.presence(b):.0%}  {b}")
    return store, symmetry, capability, bin_stats


def write_shard(store: BucketStore, args, symmetry=None,
                capability=None, bin_stats=None) -> Path:
    out = Path(args.out) / args.env_id
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"shard_{args.shard:03d}.pkl"
    payload = {
        "_schema_version": SCHEMA_VERSION,
        "env_id": args.env_id,
        "target": args.target,
        "episodes": store.episodes,
        "seen_counts": {str(b): n for b, n in store.seen_counts.items()},
        "samples": {
            str(b): [{"frame": e.frame, "payload": e.payload} for e in evs]
            for b, evs in store.samples.items()
        },
        "incomplete": [str(b) for b in store.incomplete()],
        "presence": {str(b): store.presence(b) for b in store.buckets()},
        "excluded": {str(b): r for b, r in store.excluded.items()},
        "complete": [str(b) for b in store.complete_buckets()],
        "symmetry": symmetry or {},
        "bin_stats": bin_stats or {},
        "capability": capability,
        "late": {str(b): n for b, n in store.late.items()},
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"wrote {path}")
    return path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Collect ManiSkill interactions")
    p.add_argument("--env-id", default="PickCube-v1")
    p.add_argument("--target", type=int, default=300,
                   help="samples per discovered bucket")
    p.add_argument("--patience", type=int, default=25,
                   help="successful episodes with no new bucket before freeze")
    p.add_argument("--max-attempts", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--out", default="data/maniskill_evidence")
    p.add_argument("--eps-force", type=float, default=0.05)
    p.add_argument("--min-vertical-ratio", type=float, default=0.5)
    p.add_argument("--group-gap", type=int, default=GROUP_GAP,
                   help="observed steps without the predicate to end a group")
    p.add_argument("--group-window", type=int, default=GROUP_WINDOW,
                   help="positive frames averaged around the peak")
    p.add_argument("--min-presence", type=float, default=0.2,
                   help="buckets in fewer than this fraction of episodes are reported as incidental, never silently dropped")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--pilot", action="store_true",
                   help="short run: measure rates and buckets, write no shard")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.pilot:
        args.max_attempts = min(args.max_attempts, 25)
        args.patience = 10 ** 6      # never freeze; report what appears
    store, symmetry, capability, bin_stats = collect(args)
    if not args.pilot:
        write_shard(store, args, symmetry, capability, bin_stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
