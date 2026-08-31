"""Collect successful Fetch manipulation rollouts into robot_success_states.

Discovers checkpoints under
``<ckpt-root>/<task>/<subtask>/<target>/policy.pt`` and rolls each policy out
in a vectorised MS-HAB subtask environment wrapped in
the local ``FetchCollectContactDataWrapper``. MS-HAB remains untouched. The
wrapper commits one schema-v4 record per successful environment rollout:
success pose data, every robot-interacted entity, and direct supporters of the
interacted entities. On ``close()`` it writes to::

    $MS_ASSET_DIR/data/robot_success_states/fetch/<task>/pick/<obj_id>.pkl

``build_affordances.py`` consumes the pose arrays and
``build_subtask_whitelists.py`` consumes ``interaction_rollouts``.

Usage::

    export MS_ASSET_DIR=/root/.maniskill
    python -m scenegraph.tools.collect_robot_success_states \\
        --ckpt-root /root/projects/ReLDreamer/mshab_checkpoints \\
        --algo rl --n-success 30 --num-envs 8

Filters::

    --task tidy_house --task set_table     # only these tasks
    --obj 024_bowl --obj 003_cracker_box   # only these YCB ids
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple


# Where the released MS-HAB per-object policies live on this project's server.
# The tree below it is ``<task>/<subtask>/<target>/policy.pt``; a release that
# nests them under an extra level wants that level named here, and discovery
# says which one when it finds none.
DEFAULT_CKPT_ROOT = "/root/projects/ReLDreamer/mshab_checkpoints"
# The release keeps one tree per training algorithm --  bc, dp, rl -- above
# the task level, so the full path is
# ``<root>/<algo>/<task>/<subtask>/<target>/policy.pt``. Which algorithm
# produced a rollout is not cosmetic: the whitelist and the calibration bins
# describe the behaviour of the policy that was rolled out, so mixing two
# would mine one asset from two different behaviours. Named explicitly, and
# printed, rather than merged by globbing every tree at once.
DEFAULT_CKPT_ALGO = "rl"

# What ``--skip-done`` accepts as already collected. Must track the collector
# wrapper's ``_SCHEMA_VERSION`` and the miner's ``MIN_ROLLOUT_SCHEMA``: all
# three describe the same evidence, and a floor left behind would let a stale
# pickle satisfy a recollection it cannot actually satisfy.
REQUIRED_ROLLOUT_SCHEMA = 9


def checkpoint_glob(subtask: str, algo: str = "") -> str:
    """The pattern that finds one policy per (task, target).

    With an algorithm the tree is ``<algo>/<task>/<subtask>/<target>``; the
    flat form without one is kept so a root already pointed at a single
    algorithm still works.
    """
    prefix = f"{algo}/" if algo else ""
    return f"{prefix}*/{subtask}/*/policy.pt"


def _discover_work(
    ckpt_root: Path,
    subtask: str,
    task_filter: List[str],
    obj_filter: List[str],
    algo: str = "",
) -> List[Tuple[str, str, Path]]:
    """Find per-object checkpoints, deduplicated by (task, subtask, obj_id).

    Never by obj_id alone. The same object is a different mining problem in
    each task -- set_table's bowl starts in a counter drawer, prepare_groceries'
    on the counter -- so collapsing them keeps whichever task sorted first and
    silently mines every other task's whitelist against the wrong scene. The
    output tree is task-namespaced to match, so the three collections coexist.

    ``algo`` names the training-algorithm tree above the task level. The
    positions read below are counted from the end, so both layouts index the
    same way.
    """
    seen: dict = {}
    for pt in sorted(ckpt_root.glob(checkpoint_glob(subtask, algo))):
        parts = pt.parts
        if len(parts) < 4:
            continue
        task = parts[-4]
        obj_id = parts[-2]
        # 'all' is a multi-object policy, not a per-object one, so it is only
        # collected when asked for by name. Its rollouts carry a target_key
        # each, and the whitelist miner groups on that rather than on the file.
        if obj_id == "all" and "all" not in obj_filter:
            continue
        if task_filter and task not in task_filter:
            continue
        if obj_filter and obj_id not in obj_filter:
            continue
        seen.setdefault((task, subtask, obj_id), (task, obj_id, pt.parent))
    return [seen[k] for k in sorted(seen)]


def suggest_ckpt_roots(ckpt_root: Path, scan_cap: int = 500) -> List[str]:
    """Roots under ``ckpt_root`` where a ``policy.pt`` sits at the right depth.

    Discovery wants ``<root>/<task>/<subtask>/<target>/policy.pt``, so for any
    policy found, the root it implies is its fourth parent. A release that
    nests everything one level deeper yields nothing and reports the same "no
    checkpoints matched" as an empty directory, which is the least useful
    thing it could say.
    """
    roots: Set[Path] = set()
    for index, policy in enumerate(ckpt_root.glob("**/policy.pt")):
        if index >= scan_cap:
            break
        if len(policy.parents) >= 4:
            roots.add(policy.parents[3])
    return sorted(str(path) for path in roots)


def available_algos(ckpt_root: Path, scan_cap: int = 500) -> List[str]:
    """Algorithm trees that hold policies at the expected depth.

    The released checkpoints are ``<root>/{bc,dp,rl}/<task>/...``, so a root
    pointed at the base finds nothing and the missing piece is an algorithm
    name, not a different root. Derived from where the policies actually are
    rather than from a hardcoded list, so a release that adds a fourth
    algorithm reports it without a code change.
    """
    names: Set[str] = set()
    for root in suggest_ckpt_roots(ckpt_root, scan_cap):
        candidate = Path(root)
        if candidate.parent == ckpt_root:
            names.add(candidate.name)
    return sorted(names)


def _already_done(
    asset_dir: Path, task: str, subtask: str, obj_id: str, min_samples: int,
) -> bool:
    pkl = (
        asset_dir / "robot_success_states" / "fetch" / task / subtask
        / f"{obj_id}.pkl"
    )
    if not pkl.exists():
        return False
    return _is_complete(pkl, min_samples)


def _is_complete(pkl: Path, min_samples: int) -> bool:
    """A current-schema pickle holding at least ``min_samples`` of every array.

    The version floor is not cosmetic. An older pickle is complete by every
    other measure -- it has its successes, its poses, its rollouts -- but it
    was written before the collector recorded collision extents, the
    end-effector rest calibration and the per-rollout build configuration.
    Treating it as done is how a recollection turns into a no-op that reports
    success, and the gap only surfaces later as a miner refusing every shard.
    """
    if not pkl.exists():
        return False
    try:
        import pickle
        with open(pkl, "rb") as f:
            d = pickle.load(f)
        return (
            int(d.get("_schema_version", 0)) >= REQUIRED_ROLLOUT_SCHEMA
            and len(d.get("robot_qpos", [])) >= min_samples
            and len(d.get("tcp_pose_wrt_base", [])) >= min_samples
            and len(d.get("interaction_rollouts", [])) >= min_samples
        )
    except Exception:
        return False


def _staging_root(asset_dir: Path) -> Path:
    """Where a rollout in progress writes, before it has earned its place.

    The wrapper commits on ``close()`` regardless of how many successes it
    gathered, so writing straight to the final path lets a run that stalled at
    1/30 replace a complete 30/30 pickle -- and with ``--no-skip-done`` that is
    the common case, not the rare one. Staging keeps the previous evidence
    intact until a full replacement exists.
    """
    return asset_dir / "robot_success_states.staging"


def _final_path(asset_dir: Path, task: str, subtask: str, obj_id: str) -> Path:
    return (
        asset_dir / "robot_success_states" / "fetch" / task / subtask
        / f"{obj_id}.pkl"
    )


def build_config_names(plans) -> List[str]:
    """Every build configuration a task-plan file can produce, sorted."""
    return sorted({str(getattr(p, "build_config_name", "") or "") for p in plans})


def select_build_config_plans(plans, requested: str, plan_fp) -> List:
    """The plans belonging to one build configuration, or all of them.

    Experiment B trains on one scene and evaluates on the rest, and a task
    plan file holds plans from many. Cycling all of them across the vector
    environments -- which is what happens without this -- puts several scenes
    into one collection, and provenance recorded afterwards can only tell you
    that it happened.

    Unpinned collection stays legal, because a whitelist meant to cover a
    whole split wants exactly that. It says so loudly instead.
    """
    available = build_config_names(plans)
    if not requested:
        if len(available) > 1:
            print(
                f"[env] NO --build-config: this collection spans "
                f"{len(available)} build configurations and its evidence will "
                f"mix them. Scene-pinned experiments must pass one of: "
                f"{', '.join(available)}",
                flush=True,
            )
        return list(plans)

    kept = [p for p in plans
            if str(getattr(p, "build_config_name", "") or "") == requested]
    if not kept:
        raise SystemExit(
            f"[env] --build-config {requested!r} matches no plan in "
            f"{plan_fp}. Available: {', '.join(available) or '<none>'}"
        )
    print(
        f"[env] pinned build config {requested!r}: {len(kept)}/{len(plans)} "
        f"plans, init configs "
        f"{', '.join(sorted({str(getattr(p, 'init_config_name', '') or '?') for p in kept}))}",
        flush=True,
    )
    return kept


def assert_pinned_build_config(venv, requested: str) -> None:
    """After reset, check the built scenes are the one that was asked for.

    The filter selects plans; this checks what the environment actually
    built. They can differ -- a plan attribute renamed upstream would make the
    filter match nothing quietly if it were not for the empty-set check, and
    match the wrong thing if the attribute changed meaning.
    """
    if not requested:
        return
    base = venv.unwrapped
    built = sorted(str(bc) for bc in getattr(base, "bc_to_task_plans", {}))
    if built != [requested]:
        raise SystemExit(
            f"[env] asked for build config {requested!r} but the environment "
            f"built {built}. Refusing to collect: the evidence would not be "
            "from the requested scene."
        )
    print(f"[env] verified after reset: every sub-scene is {requested!r}",
          flush=True)


def _build_env(task: str, obj_id: str, args, ckpt_dir: Optional[Path] = None):
    """Recreate the wrapper stack the released SAC was trained on, plus the
    local collector on the inside (so it sees raw success
    info before policy-side obs transforms)."""
    import gymnasium as gym
    from mani_skill import ASSET_DIR
    from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
    import mshab.envs  # noqa: F401  registers PickSubtaskTrain-v0 etc.
    from mshab.envs.planner import plan_data_from_file
    from mshab.envs.wrappers import (
        FetchActionWrapper,
        FetchDepthObservationWrapper,
        FrameStack,
    )
    from scenegraph.adapters.collect_contact_data import FetchCollectContactDataWrapper

    RD = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
    split = str(getattr(args, "split", "train") or "train")
    plan_fp = RD / "task_plans" / task / args.subtask / split / f"{obj_id}.json"
    if not plan_fp.exists():
        raise FileNotFoundError(f"missing task plan: {plan_fp}")
    spawn_data_fp = RD / "spawn_data" / task / args.subtask / split / "spawn_data.pt"
    plan_data = plan_data_from_file(plan_fp)
    if not plan_data.plans:
        raise RuntimeError(f"{plan_fp} contained no plans")

    requested = str(getattr(args, "build_config", "") or "")
    plans = select_build_config_plans(plan_data.plans, requested, plan_fp)

    # Cycle plans across envs so we get init-config diversity even when a
    # single task plan file is shorter than --num-envs. With a pinned build
    # config the diversity is in the starting arrangement only, which is the
    # point: one scene, many spawns.
    n_plans = len(plans)
    n_envs = max(1, args.num_envs)
    task_plans = [plans[i % n_plans] for i in range(n_envs)]

    env = gym.make(
        f"{args.subtask.capitalize()}SubtaskTrain-v0",
        num_envs=n_envs,
        obs_mode="rgb+depth+segmentation",
        sim_backend="gpu",
        robot_uids="fetch",
        control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense",
        render_mode="all",
        shader_dir="minimal",
        max_episode_steps=args.max_episode_steps,
        task_plans=task_plans,
        scene_builder_cls=plan_data.dataset,
        spawn_data_fp=spawn_data_fp,
        require_build_configs_repeated_equally_across_envs=False,
        add_event_tracker_info=True,
        sensor_configs=dict(width=args.sensor_width, height=args.sensor_height),
    )

    # Collect on the INSIDE (closest to base env) so it reads raw
    # ``info["success"]`` and raw ``agent.robot.qpos`` before any policy-side
    # wrapper has a chance to mutate them.
    collect = FetchCollectContactDataWrapper(
        env,
        out_root=str(_staging_root(Path(args.asset_dir))),
        task_group=task,
        split=split,
        checkpoint_path=str(ckpt_dir) if ckpt_dir else "",
        task_plan_path=str(plan_fp),
        # Index matches the vector-environment index, so each rollout can name
        # the plan that produced it. ``require_build_config`` makes the
        # wrapper refuse a mixed environment outright rather than record that
        # it was mixed after the sim-hours are spent.
        env_plans=task_plans,
        require_build_config=requested,
    )
    env = collect
    env = FetchDepthObservationWrapper(env, cat_state=True, cat_pixels=False)
    env = FrameStack(
        env,
        num_stack=args.frame_stack,
        stacking_keys=["fetch_head_depth", "fetch_hand_depth"],
    )
    env = FetchActionWrapper(
        env,
        stationary_base=False,
        stationary_torso=False,
        stationary_head=True,
    )
    venv = ManiSkillVectorEnv(
        env,
        ignore_terminations=True,
        max_episode_steps=args.max_episode_steps,
    )
    return venv, collect, plan_fp


def _to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    import numpy as np
    return np.asarray(x)


def _commit_successes_at_script_level(venv, collect, info, committed_mask):
    """Detect newly-successful envs and call collect.commit_success.

    Reading pose/qpos here -- AFTER venv.step has fully returned -- is the
    point at which the simulator state matches what MS-HAB's evaluate() saw.
    Doing the read from inside the wrapper's step sees the post-stowage
    state, where the active actor is parked at ~[-1e4,-1e4,-1e4] and
    is_grasping returns False. ``committed_mask`` is a per-env boolean
    array owned by the script that prevents duplicate commits within a
    single episode; it is the caller's responsibility to reset entries on
    episode boundary.
    """
    import numpy as np
    success = info.get("success")
    if success is None:
        return
    n_envs = collect.num_envs
    success_np = _to_np(success).astype(bool).reshape(-1)
    if success_np.size < n_envs:
        success_np = np.pad(success_np, (0, n_envs - success_np.size))
    newly = success_np[:n_envs] & ~committed_mask
    if not newly.any():
        return

    base_env = venv.unwrapped
    agent = base_env.agent
    ptrs = _to_np(base_env.subtask_pointer).astype(int).reshape(-1)
    base_inv = agent.base_link.pose.inv()
    tcp_rel_all = _to_np((base_inv * agent.tcp.pose).raw_pose)
    qpos_all = _to_np(agent.robot.qpos)

    for env_idx in np.where(newly)[0].tolist():
        ptr = int(min(ptrs[env_idx], len(base_env.task_plan) - 1))
        # Close subtasks have no target actor (``subtask_objs[ptr] is None``);
        # the interaction target is the articulation instead. Fall back to
        # ``subtask_articulations[ptr]`` so its pose gets recorded.
        target = None
        if ptr < len(base_env.subtask_objs):
            target = base_env.subtask_objs[ptr]
        if target is None and ptr < len(base_env.subtask_articulations):
            target = base_env.subtask_articulations[ptr]
        if target is None or getattr(target, "pose", None) is None:
            continue
        obj_rel_all = _to_np((base_inv * target.pose).raw_pose)
        obj_pose = (
            obj_rel_all[env_idx] if obj_rel_all.ndim == 2 else obj_rel_all
        )
        tcp_pose = (
            tcp_rel_all[env_idx] if tcp_rel_all.ndim == 2 else tcp_rel_all
        )
        qpos = qpos_all[env_idx] if qpos_all.ndim == 2 else qpos_all
        collect.commit_success(int(env_idx), qpos, obj_pose, tcp_pose)
        committed_mask[env_idx] = True


def _collect_one(task: str, obj_id: str, ckpt_dir: Path, args) -> Tuple[int, Path]:
    from scenegraph.adapters.policy_loader import load_policy

    venv, collect, plan_fp = _build_env(task, obj_id, args, ckpt_dir=ckpt_dir)
    print(f"[env] plan={plan_fp.name} num_envs={venv.unwrapped.num_envs}")

    obs, _ = venv.reset(seed=args.seed, options=dict(reconfigure=True))
    assert_pinned_build_config(venv, str(getattr(args, "build_config", "") or ""))
    policy = load_policy(str(ckpt_dir), venv, obs, device=args.device)
    if policy.kind == "random":
        print(f"[warn] {obj_id}: policy loader fell back to random actions; "
              "success rate will be ~0. Check ckpt-dir and config.yml.")

    import numpy as np
    n_envs = collect.num_envs
    # Per-env latch: True once we've committed in the current episode.
    # Cleared on truncation / termination signal so the next episode can
    # contribute another sample.
    committed_mask = np.zeros(n_envs, dtype=bool)
    n_target = args.n_success
    t0 = time.time()
    step = 0
    last_log = 0
    last_progress_step = 0
    last_progress_n = 0
    while True:
        n_succ = len(collect.success_robot_qpos)
        if n_succ >= n_target:
            print(f"[ok] {obj_id}: reached {n_succ}/{n_target} successes "
                  f"in {step} steps ({time.time() - t0:.1f}s)")
            break
        if step >= args.max_total_steps:
            print(f"[cap] {obj_id}: hit --max-total-steps={args.max_total_steps} "
                  f"with {n_succ}/{n_target} successes")
            break
        # Stall detector: if no new success in --stall-steps env steps, give up.
        if n_succ > last_progress_n:
            last_progress_n = n_succ
            last_progress_step = step
        if step - last_progress_step > args.stall_steps and n_succ > 0:
            print(f"[stall] {obj_id}: no new success in {args.stall_steps} steps; "
                  f"stopping with {n_succ}/{n_target}")
            break

        action = policy.act(obs)
        obs, _rew, term, trunc, info = venv.step(action)
        # Commit successes RIGHT HERE -- this is the same point at which
        # the probe (probe_pose_sources.py) read state and got |tcp-obj|
        # ~5 cm. Doing it inside the wrapper sees the post-stowage state.
        _commit_successes_at_script_level(venv, collect, info, committed_mask)
        # Clear the per-episode latch on done so the next episode can
        # contribute its own commit.
        done = (
            _to_np(term).astype(bool).reshape(-1)
            | _to_np(trunc).astype(bool).reshape(-1)
        )
        if done.size < n_envs:
            done = np.pad(done, (0, n_envs - done.size))
        committed_mask[done[:n_envs]] = False
        step += 1
        if step - last_log >= args.log_every:
            last_log = step
            print(f"  [{obj_id}] step={step} successes={n_succ} "
                  f"({step / max(time.time() - t0, 1e-9):.1f} steps/s)")

    # close() flushes the pickle to disk via the collector wrapper.
    venv.close()
    n_final = len(collect.success_robot_qpos)
    print(f"[staged] {collect.save_path}  ({n_final} samples)")
    return n_final, Path(collect.save_path)


def parse_args(argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--ckpt-root", default=DEFAULT_CKPT_ROOT,
        help="Root containing <task>/<subtask>/<target>/policy.pt subtrees. "
             "If a release nests them under an extra level, name that level "
             "-- discovery reports the roots it found on failure.",
    )
    p.add_argument(
        "--subtask", default="pick", choices=["pick", "open", "close"],
        help="Manipulation subtask checkpoint tree to collect.",
    )
    p.add_argument(
        "--algo", default=DEFAULT_CKPT_ALGO,
        help="Training-algorithm tree under --ckpt-root (bc, dp, rl). The "
             "released checkpoints keep one per algorithm above the task "
             "level. Pass an empty string if the root already names one.",
    )
    p.add_argument(
        "--build-config", default="",
        help="Collect only from this ReplicaCAD build configuration. Required "
             "for a scene-pinned experiment: without it the task plans are "
             "cycled across the vector envs and one collection mixes scenes.",
    )
    p.add_argument(
        "--list-build-configs", action="store_true",
        help="Print the build configurations each selected task plan file "
             "offers and exit. Reads the task plan JSON only -- no "
             "simulator and no rollouts, but it still needs mshab and the "
             "ReplicaCAD assets, so it runs on the collection server.",
    )
    p.add_argument(
        "--n-success", type=int, default=30,
        help="Target successful picks per object before stopping (default 30).",
    )
    p.add_argument(
        "--max-total-steps", type=int, default=20000,
        help="Hard cap on env steps per object (default 20000).",
    )
    p.add_argument(
        "--stall-steps", type=int, default=4000,
        help="Stop if no new success in this many steps (default 4000).",
    )
    p.add_argument(
        "--max-episode-steps", type=int, default=200,
        help="Per-episode step cap (default 200, same as MS-HAB training).",
    )
    p.add_argument(
        "--num-envs", type=int, default=8,
        help="Parallel envs per object on GPU sim (default 8).",
    )
    p.add_argument("--frame-stack", type=int, default=3)
    p.add_argument("--sensor-width", type=int, default=128)
    p.add_argument("--sensor-height", type=int, default=128)
    p.add_argument(
        "--task", action="append", default=[],
        help="Filter to specific task(s); repeatable. Default: all.",
    )
    p.add_argument(
        "--obj", action="append", default=[],
        help="Filter to specific YCB id(s); repeatable. Default: all.",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--asset-dir",
        default=str(
            Path(os.environ.get("MS_ASSET_DIR", os.path.expanduser("~/.maniskill")))
            / "data"
        ),
        help="Data root (default: $MS_ASSET_DIR/data, then ~/.maniskill/data). "
             "This is the directory that contains robot_success_states/, "
             "scene_datasets/, etc -- the same dir mani_skill.ASSET_DIR points at. "
             "Rollouts are written under its robot_success_states/; scene assets "
             "are still read from mani_skill.ASSET_DIR.",
    )
    p.add_argument(
        "--split", default="train",
        help="Task-plan split to roll out (default: train). Recorded in each "
             "pickle's provenance block.",
    )
    p.add_argument(
        "--no-skip-done", action="store_true",
        help="Re-collect even if a .pkl with >= n-success rows already exists.",
    )
    p.add_argument("--log-every", type=int, default=200)
    return p.parse_args(argv)


def _list_build_configs(work, args) -> int:
    """Report which scenes each unit could collect from, without a simulator.

    The audit that has to happen before a scene-pinned collection: which build
    configurations exist, and therefore which one ``--build-config`` should
    name. Reads the task plan JSON only.
    """
    from mani_skill import ASSET_DIR
    from mshab.envs.planner import plan_data_from_file

    rd = ASSET_DIR / "scene_datasets/replica_cad_dataset/rearrange"
    split = str(getattr(args, "split", "train") or "train")
    for task, obj_id, _ckpt in work:
        plan_fp = rd / "task_plans" / task / args.subtask / split / f"{obj_id}.json"
        if not plan_fp.exists():
            print(f"{task}/{obj_id}: MISSING {plan_fp}")
            continue
        plans = plan_data_from_file(plan_fp).plans
        names = build_config_names(plans)
        print(f"{task}/{obj_id}: {len(plans)} plans over "
              f"{len(names)} build config(s)")
        for name in names:
            n = sum(1 for p in plans
                    if str(getattr(p, "build_config_name", "") or "") == name)
            print(f"    {name}  ({n} plan(s))")
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    ckpt_root = Path(args.ckpt_root).resolve()
    asset_dir = Path(args.asset_dir).resolve()

    if not ckpt_root.is_dir():
        print(f"ERROR: --ckpt-root not found: {ckpt_root}", file=sys.stderr)
        if os.name == "nt" and str(args.ckpt_root).startswith("/"):
            # Windows reads a leading slash as drive-relative, so a POSIX
            # server path silently resolves against the current drive. The
            # resolved path above looks nothing like what was typed, which is
            # a confusing way to learn the command is on the wrong machine.
            print(f"       {args.ckpt_root!r} is a POSIX path and this is "
                  "Windows, so it resolved against the current drive. "
                  "Collection runs on the server.", file=sys.stderr)
        return 2

    algo = str(getattr(args, "algo", "") or "")
    work = _discover_work(ckpt_root, args.subtask, args.task, args.obj, algo)
    if not work:
        print("ERROR: no per-object checkpoints matched the filters under "
              f"{ckpt_root}", file=sys.stderr)
        shape = f"{algo}/" if algo else ""
        print(f"       expected {ckpt_root}/{shape}<task>/{args.subtask}"
              "/<target>/policy.pt", file=sys.stderr)
        algos = sorted({name for name in available_algos(ckpt_root)
                        if name != algo})
        if algos:
            print("       policies do exist under: "
                  + ", ".join(f"--algo {name}" for name in algos),
                  file=sys.stderr)
        elif not any(ckpt_root.glob("**/policy.pt")):
            print(f"       no policy.pt anywhere under {ckpt_root}",
                  file=sys.stderr)
        return 2

    if args.list_build_configs:
        return _list_build_configs(work, args)

    print(f"[plan] algo={algo or '<flat>'} under {ckpt_root}")
    print(f"[plan] {len(work)} (task, obj) units; n_success target={args.n_success}")
    print(
        f"[plan] writing under "
        f"{asset_dir}/robot_success_states/fetch/<task>/{args.subtask}/"
    )
    ok = 0
    failed: List[str] = []
    for task, obj_id, ckpt_dir in work:
        print(f"\n=== {task}/{obj_id}   ckpt={ckpt_dir.name} ===")
        if not args.no_skip_done and _already_done(
            asset_dir, task, args.subtask, obj_id, args.n_success,
        ):
            print(f"[skip] {obj_id}: existing schema-v4 .pkl already has "
                  f">= {args.n_success} samples (use --no-skip-done to redo)")
            ok += 1
            continue
        try:
            n, staged = _collect_one(task, obj_id, ckpt_dir, args)
            final = _final_path(asset_dir, task, args.subtask, obj_id)
            if n >= args.n_success:
                final.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged, final)
                print(f"[wrote] {final}  ({n} samples)")
                ok += 1
            else:
                # Short of target: the staged file is real evidence but not the
                # complete set the miners assume, so it is discarded rather
                # than promoted over whatever is already there.
                print(f"[incomplete] {obj_id}: {n}/{args.n_success} successes; "
                      f"discarding {staged} and keeping any existing "
                      f"{final.name}")
                staged.unlink(missing_ok=True)
                failed.append(obj_id)
        except Exception:
            print(f"[error] {task}/{obj_id}:")
            traceback.print_exc()
            failed.append(obj_id)

    print(f"\n[summary] {ok}/{len(work)} units produced .pkl files")
    if failed:
        print(f"[summary] failures: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
