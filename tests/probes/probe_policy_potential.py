"""Score the graph of a real successful MS-HAB rollout.

Everything else validates the schedule against frames someone constructed.
This drives the released policy to an actual success, builds the graph on
every step, and runs the same replay potential the world model is supervised
against -- so it answers the one question constructed frames cannot: does a
trajectory the *environment* calls a success end at 1.0.

Why it lives here rather than in ``render_paper_frames``: the released SAC
consumes MS-HAB's own observation stack, while the graph builder reads raw
segmentation. Both are available inside the collector's environment, which
already drives the policy to success, so this borrows that stack and adds a
passive observation cache. Nothing about collection changes, and no
production path is made to capture graphs.

**The frames are packed the way training packs them.** Scoring reads row
indices out of ``pack_graph``, not entity ids off the graph: the schedule
names the target by the ``$active_target`` sentinel, which is not any real
entity id, so comparing the two directly matched nothing and silently dropped
every fact about the target -- the facts the schedule is almost entirely made
of. ``pack_graph`` also runs ``verify_protected_rows``, so a frame whose row 0
is not the end effector, or whose row 1 is not the flagged target, refuses to
be scored rather than being scored wrongly.

Asset paths are explicit so the same probe runs against a provisional tree now
and the final server assets later::

    python tests/probes/probe_policy_potential.py \\
        --whitelist-dir scenegraph/configs/subtask_whitelists/tidy_house \\
        --affordance   scenegraph/configs/affordances/tidy_house.json \\
        --thresholds   scenegraph/configs/thresholds.yaml \\
        --task tidy_house --obj 004_sugar_box --algo rl \\
        --build-config v3_sc0_staging_00.scene_instance.json

LIVE EXECUTION IS PENDING: it needs the simulator and the released
checkpoints. Everything that can be exercised without them -- the packing
contract, the scoring, the trace report -- is unit-tested in
``tests/test_potential_probe.py``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class Report:
    def __init__(self):
        self.rows: List[Any] = []

    def check(self, name, ok, detail=""):
        self.rows.append((bool(ok), name, detail))

    def note(self, name, detail=""):
        self.rows.append((None, name, detail))

    def render(self) -> int:
        width = max(len(n) for _, n, _ in self.rows)
        failed = 0
        for ok, name, detail in self.rows:
            mark = "    " if ok is None else (" ok " if ok else "FAIL")
            failed += 1 if ok is False else 0
            print(f"  [{mark}] {name:<{width}}  {detail}")
        return failed


def capacity_for(graphs, n_max: int = 0,
                 e_max: int = 0) -> Tuple[int, int, Dict]:
    """``(n_max, e_max, occupancy)`` for packing this trace.

    When the run's configured capacities are given they are used, so an
    overflow here is the overflow training would hit. When they are not, the
    trace is packed at its own peak -- which lets the frames be scored but
    proves nothing about capacity, and is reported as such. One episode in one
    configuration is a floor: retention only adds nodes, another scene holds
    other furniture, and a canonical key can have two instances at once.
    """
    from scenegraph.tools.audit_graph_capacity import (
        frames_from_graphs, occupancy_report,
    )

    occupancy = occupancy_report(frames_from_graphs(graphs), n_max or None,
                                 e_max or None)
    # Three rows are reserved by contract -- end effector, target, site --
    # so a trace that never showed three nodes still needs room for them.
    return (int(n_max or max(occupancy["peak_nodes"], 3)),
            int(e_max or max(occupancy["peak_edges"], 1)),
            occupancy)


def score_frames(graphs, schedule, vocab, *, n_max: int, e_max: int,
                 n_cams: int = 2):
    """``(potentials, valids)`` for recorded graphs, packed as training packs.

    Separated from the rollout so the arithmetic can be exercised without a
    simulator. Every tensor here is read straight out of ``pack_graph``: node
    rows, not entity ids, and edge endpoints as row indices, which is the only
    form the scorer's ``resolve_rows`` is written against.
    """
    import torch

    from progress import TaskScheduleReplayPotential
    from scenegraph.adapters.graph_pack import pack_graph

    scorer = TaskScheduleReplayPotential(
        schedule, len(vocab.absolute.token_to_id))
    potentials, valids = [], []
    for graph in graphs:
        packed = pack_graph(graph, vocab, n_max=n_max, e_max=e_max,
                            n_cams=n_cams, use_target_flag=True)
        node_ent = torch.as_tensor(
            packed["graph_node_ent"], dtype=torch.long)[None]
        rel = torch.as_tensor(packed["graph_edge_rel"], dtype=torch.long)
        # A packed edge slot is real exactly when its relation is not the pad
        # id -- the same test the encoder and the world model apply.
        real = rel.ne(vocab.relation.pad_id)
        value, valid = scorer(
            node_ent,
            rel[real],
            torch.as_tensor(packed["graph_edge_abs"], dtype=torch.long)[real],
            torch.as_tensor(packed["graph_edge_src"], dtype=torch.long)[real],
            torch.as_tensor(packed["graph_edge_dst"], dtype=torch.long)[real],
            torch.zeros(int(real.sum()), dtype=torch.long), 1,
        )
        potentials.append(float(value.item()))
        valids.append(bool(valid.item()))
    return potentials, valids


def report_trace(rep, graphs, potentials, valids, discount: float,
                 occupancy=None, n_max=None, e_max=None,
                 configured: bool = False) -> None:
    """Every acceptance criterion the trace can answer."""
    import torch

    from progress import potential_shaping

    rep.note("frames", f"{len(graphs)}")
    invalid = [i for i, v in enumerate(valids) if not v]
    rep.check("every frame is schedule-readable", not invalid,
              "all valid" if not invalid else f"{len(invalid)} invalid: "
                                              f"{invalid[:8]}")
    if not potentials:
        rep.check("a trajectory was scored", False, "no frames")
        return
    rep.check("the successful frame scores 1.0",
              abs(potentials[-1] - 1.0) < 1e-4, f"{potentials[-1]:.6f}")
    rep.check("the potential starts below 1", potentials[0] < 1.0,
              f"{potentials[0]:.6f}")
    rep.check("the potential never exceeds 1", max(potentials) <= 1.0 + 1e-6,
              f"peak {max(potentials):.6f}")

    phi = torch.tensor(potentials, dtype=torch.float32).reshape(1, -1, 1)
    shaped = potential_shaping(phi, torch.ones_like(phi), discount=discount)
    total, expected = float(shaped.sum().item()), potentials[-1] - potentials[0]
    if abs(discount - 1.0) < 1e-9:
        # Undiscounted, a telescoping sum is exactly the endpoint difference.
        rep.check("shaping telescopes", abs(total - expected) < 1e-3,
                  f"sum {total:+.6f} vs phi_T - phi_0 {expected:+.6f}")
    else:
        rep.note("shaping sum",
                 f"{total:+.6f} at discount {discount:g} (telescoping is only "
                 "the endpoint difference undiscounted)")

    if occupancy is None:
        return
    rep.note("peak occupancy (LOWER bound: one episode, one scene)",
             f"{occupancy['peak_nodes']} node(s), "
             f"{occupancy['peak_edges']} edge(s) over "
             f"{occupancy['n_frames']} frame(s)")
    rep.note("most simultaneous instances of one key",
             f"{occupancy['peak_instances_per_key']} "
             f"({occupancy['peak_duplicate_key'] or 'none duplicated'})")
    rep.note("distinct entity keys in the trace",
             f"{occupancy['distinct_entity_keys']}")
    over_n = occupancy["frames_over_n_max"]
    over_e = occupancy["frames_over_e_max"]
    if configured:
        rep.check("the configured capacities hold for this episode",
                  not over_n and not over_e,
                  f"{len(over_n)} frame(s) over n_max, "
                  f"{len(over_e)} over e_max")
        for frame in over_n[:3]:
            rep.note("  over n_max", f"frame {frame['frame']}: "
                                     f"{frame['entity_keys']}")
        for frame in over_e[:3]:
            rep.note("  over e_max",
                     f"frame {frame['frame']}: {frame['n_edges']} edges")
    else:
        rep.note("packed at", f"n_max={n_max}, e_max={e_max} -- fitted to "
                              "this trace, so it proves nothing about "
                              "capacity. Pass --n-max/--e-max to test the "
                              "run's own settings.")


def compile_schedule_from(args, entity):
    from scenegraph.core.schedule import compile_from_source, mshab_schedule_source

    source = mshab_schedule_source(
        args.task, args.subtask, args.configs, args.schedule_dir,
        args.whitelist_dir, affordance_path=args.affordance)
    return compile_from_source(source, entity), source


def load_vocabs(args):
    """The run's own vocabulary, from the explicitly named asset tree.

    Built through ``build_graph_vocab`` rather than assembled here, so the
    label-validity masks the packer checks against are the ones the runtime
    uses.
    """
    from scenegraph.adapters.graph_vocab import build_graph_vocab

    return build_graph_vocab(args.whitelist_dir)


def build_config(args, rep):
    """``cfg`` for the graph builder, from the paths this run was given.

    ``load_config`` resolves both mined assets from the thresholds file's own
    directory and the task group. That is right for a training run and wrong
    for a probe pointed at a tree somewhere else, so both are overridden here
    and the override is reported -- a probe that silently scored the packaged
    assets while claiming to score the server's would be worse than one that
    failed.
    """
    from scenegraph.configs.loader import load_config
    from scenegraph.core.affordance import load_affordance_set

    cfg = load_config(args.thresholds, task_group=args.task,
                      require_assets=False)
    affordance = pathlib.Path(args.affordance)
    if not affordance.is_file():
        rep.check("the affordance asset exists", False, str(affordance))
        return None
    aff = load_affordance_set(str(affordance))
    if aff.is_empty():
        rep.check("the affordance asset carries objects", False,
                  str(affordance))
        return None
    cfg["affordance_set"] = aff
    cfg["affordances"] = dict(cfg.get("affordances") or {},
                              asset_path_abs=str(affordance))
    cfg["whitelist_dir"] = args.whitelist_dir
    cfg["use_target_flag"] = True
    # The MS-HAB run's own graph settings, not this module's defaults. A probe
    # that measured a keep-everything graph would report an occupancy the
    # training run never sees, and one that emitted object-object spatial
    # edges would report facts the schedule cannot name.
    cfg["object_object_spatial"] = bool(args.object_object_spatial)
    cfg["cameras"] = list(args.cameras)
    cfg["visibility_policy"] = args.visibility_policy
    rep.note("affordance asset", str(affordance))
    rep.note("whitelist dir", args.whitelist_dir)
    rep.note("graph settings",
             f"visibility={args.visibility_policy}, "
             f"cameras={list(args.cameras)}, "
             f"object_object_spatial={bool(args.object_object_spatial)}")
    rep.check("the configured assets are the ones named on the command line",
              cfg["whitelist_dir"] == args.whitelist_dir
              and cfg["affordances"]["asset_path_abs"] == str(affordance))
    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--whitelist-dir", required=True)
    p.add_argument("--affordance", required=True)
    p.add_argument("--schedule-dir",
                   default=str(ROOT / "scenegraph" / "configs" / "schedules"))
    p.add_argument("--configs", default=str(ROOT / "scenegraph" / "configs"))
    p.add_argument("--thresholds", required=True)
    p.add_argument("--task", default="tidy_house")
    p.add_argument("--subtask", default="pick")
    p.add_argument("--obj", default="004_sugar_box")
    p.add_argument("--algo", default="rl")
    p.add_argument("--build-config", default="")
    p.add_argument("--ckpt-root", default="")
    p.add_argument("--asset-dir", default="",
                   help="ManiSkill data root. The collector wrapper stages "
                        "under it; pointing it at the config tree would write "
                        "rollout files into the repository.")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--max-episode-steps", type=int, default=200)
    p.add_argument("--max-total-steps", type=int, default=4000)
    p.add_argument("--discount", type=float, default=1.0)
    p.add_argument("--n-max", type=int, default=0)
    p.add_argument("--e-max", type=int, default=0)
    p.add_argument("--cameras", nargs="+",
                   default=["fetch_head", "fetch_hand"],
                   help="Camera order, which fixes the bbox axis order. Must "
                        "match env.cameras for the run being probed.")
    p.add_argument("--visibility-policy", default="projected_camera",
                   help="MS-HAB runs projected_camera; the builder's own "
                        "default is keep, which admits more nodes.")
    p.add_argument("--object-object-spatial", action="store_true",
                   help="Off for MS-HAB Pick, matching env.graph.")
    p.add_argument("--out", default="", help="Write the trace as JSON.")
    args = p.parse_args(argv)

    rep = Report()
    vocab = load_vocabs(args)
    schedule, source = compile_schedule_from(args, vocab.entity)
    rep.note("entity vocabulary", f"{len(vocab.entity.token_to_id)} tokens")
    rep.note("schedule", f"{len(schedule.phases)} phases, "
                         f"{len(schedule.slots)} distinct facts")
    # One asset tree for both halves. Compiling the schedule against the
    # packaged affordances while building graphs against a mined tree is a
    # mismatch nothing downstream would report.
    rep.check("the schedule compiled against the named affordance asset",
              source.affordance_path == args.affordance,
              f"{source.affordance_path} vs {args.affordance}")

    cfg = build_config(args, rep)
    if cfg is None:
        rep.render()
        return 1

    episode, success_at = rollout_graphs(args, cfg, rep)
    if not episode or success_at is None:
        rep.render()
        return 1

    # Scored to the success frame; audited over the whole episode. Retention
    # only adds nodes, so the frames after success are where occupancy peaks,
    # and stopping the audit at success would understate it.
    scored = episode[:success_at + 1]
    configured = bool(args.n_max and args.e_max)
    n_max, e_max, occupancy = capacity_for(episode, args.n_max, args.e_max)
    try:
        potentials, valids = score_frames(
            scored, schedule, vocab, n_max=n_max, e_max=e_max,
            n_cams=len(args.cameras))
    except (ValueError, RuntimeError) as exc:
        # ``pack_graph`` raises on a broken protected row and on edge overflow.
        # Both are results, not crashes: report and stop.
        rep.check("the frames pack the way training packs them", False,
                  f"{type(exc).__name__}: {exc}")
        rep.render()
        return 1
    rep.check("the frames pack the way training packs them", True,
              f"n_max={n_max}, e_max={e_max}, protected rows verified")
    report_trace(rep, scored, potentials, valids, args.discount,
                 occupancy, n_max, e_max, configured)

    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"potentials": potentials, "valid": valids,
                       "success_frame": success_at,
                       "episode_frames": len(episode)}, handle)
        rep.note("trace written", args.out)

    failed = rep.render()
    print(f"\n{'all checks passed' if not failed else str(failed) + ' CHECK(S) FAILED'}")
    return 1 if failed else 0


def rollout_graphs(args, cfg, rep):
    """``(frames, success_index)`` for one episode the environment calls a
    success, or ``([], None)`` with a reason already reported.

    Four things this has to get right, none of which the frames themselves
    would reveal:

    * **The reset frame.** The graph at reset is a real frame of the episode
      and the one every temporal delta is differenced from. Building the first
      graph after the first action skipped it and dated every later frame by
      one step.
    * **The raw observation.** The builder reads segmentation, which the
      policy-side wrappers flatten away. It comes from the cache installed
      inside the stack, and a probe that read ``None`` there would build
      frames from nothing and report them as valid.
    * **Episode boundaries.** ``ignore_terminations=True``, so success does
      not end the episode and the successful state is observable after the
      step returns. Truncation still auto-resets, and the observation that
      comes back with it already belongs to the next episode -- so a success
      landing on the same step as a truncation must not be read off it, and
      the trace restarts rather than splicing two episodes into one curve.
    * **The whole episode.** Occupancy is audited past the success frame,
      because retention only adds nodes and the peak is usually at the end.
    """
    import numpy as np

    from scenegraph.core.graph_builder import GraphBuilder
    from scenegraph.adapters.policy_loader import load_policy
    from scenegraph.tools.collect_robot_success_states import (
        DEFAULT_CKPT_ALGO, DEFAULT_CKPT_ROOT, _build_env,
    )

    ckpt_root = pathlib.Path(args.ckpt_root or DEFAULT_CKPT_ROOT)
    ckpt_dir = (ckpt_root / (args.algo or DEFAULT_CKPT_ALGO) / args.task
                / args.subtask / args.obj)
    if not ckpt_dir.is_dir():
        rep.check("the policy checkpoint exists", False, str(ckpt_dir))
        return [], None
    if not args.asset_dir:
        rep.check("--asset-dir was given", False,
                  "the collector wrapper stages under it; without it the "
                  "probe would write rollout files beside the configs")
        return [], None

    env_args = argparse.Namespace(
        subtask=args.subtask, num_envs=args.num_envs, split="train",
        build_config=args.build_config,
        max_episode_steps=args.max_episode_steps,
        sensor_width=128, sensor_height=128, frame_stack=3,
        asset_dir=args.asset_dir)
    venv, _collect, _plan = _build_env(
        args.task, args.obj, env_args, ckpt_dir=ckpt_dir,
        capture_raw_obs=True)
    try:
        obs, _ = venv.reset(seed=0, options=dict(reconfigure=True))
        cache = getattr(venv.unwrapped, "raw_obs_cache", None)
        if cache is None or cache.raw_obs is None:
            rep.check("the raw observation is reachable", False,
                      "env.unwrapped.raw_obs_cache is unset, so every graph "
                      "would be built from no observation")
            return [], None
        rep.check("the raw observation is reachable", True,
                  f"{len(cache.raw_obs)} top-level key(s)")

        policy = load_policy(str(ckpt_dir), venv, obs, device="cuda")
        # Constructed the way ``graph_obs`` constructs it for training, so the
        # nodes admitted here are the nodes the run admits.
        builder = GraphBuilder(venv.unwrapped, cfg, env_idx=0,
                               env_id=f"{args.task}/{args.subtask}",
                               camera_order=list(args.cameras),
                               visibility_policy=args.visibility_policy,
                               use_target_flag=True)

        def flag(info, key):
            value = info.get(key) if isinstance(info, dict) else None
            if value is None:
                return False
            if hasattr(value, "detach"):
                value = value.detach().cpu()
            return bool(np.asarray(value).reshape(-1)[0])

        frames, success_at, restarts = [], None, 0
        boundary, succeeded = True, False
        for step in range(args.max_total_steps):
            graph, _m, _c, _r = builder.step(
                cache.raw_obs, step, episode_boundary=boundary,
                need_masks=False)
            frames.append(graph)
            if succeeded and success_at is None:
                success_at = len(frames) - 1
                rep.note("success at frame",
                         f"{success_at} of the episode, step {step}")
                rep.note("episode restarts before it", f"{restarts}")
            obs, _reward, terminated, truncated, info = venv.step(
                policy.act(obs))
            cut = flag(info, "TimeLimit.truncated") or _reshaped(truncated)
            done = cut or _reshaped(terminated)
            if success_at is not None and done:
                return frames, success_at        # audited to the episode end
            # A success that coincides with the reset is not observable: the
            # returned observation is already the next episode's, so reading
            # the flag off it would score an auto-reset frame as the success.
            succeeded = flag(info, "success") and not done
            boundary = done
            if done and success_at is None:
                frames, restarts = [], restarts + 1
        if success_at is not None:
            return frames, success_at
        rep.check("the policy reached a success", False,
                  f"none in {args.max_total_steps} steps, "
                  f"{restarts} episode(s)")
        return [], None
    finally:
        venv.close()


def _reshaped(value) -> bool:
    """First environment's flag, whatever tensor library produced it."""
    import numpy as np

    if value is None:
        return False
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return bool(np.asarray(value).reshape(-1)[0])
