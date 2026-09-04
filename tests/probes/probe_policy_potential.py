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

Asset paths are explicit so the same probe runs against a provisional pilot
tree now and the final server assets later::

    python tests/probes/probe_policy_potential.py \\
        --whitelist-dir /tmp/mshab_pick_pilot/subtask_whitelists/tidy_house \\
        --affordance   /tmp/mshab_pick_pilot/affordances/tidy_house.json \\
        --schedule-dir scenegraph/configs/schedules \\
        --thresholds   /tmp/mshab_pick_pilot/thresholds.yaml \\
        --task tidy_house --obj 004_sugar_box --algo rl \\
        --build-config v3_sc0_staging_00.scene_instance.json

LIVE EXECUTION IS PENDING: it needs the simulator, the released checkpoints
and a re-mined asset. The plumbing below is unit-tested; the rollout is not.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class Report:
    def __init__(self):
        self.rows: List[Any] = []

    def check(self, name, ok, detail=""):
        self.rows.append((bool(ok), name, detail))

    def note(self, name, detail):
        self.rows.append((None, name, detail))

    def render(self) -> int:
        width = max(len(n) for _, n, _ in self.rows)
        failed = 0
        for ok, name, detail in self.rows:
            mark = "    " if ok is None else (" ok " if ok else "FAIL")
            failed += 1 if ok is False else 0
            print(f"  [{mark}] {name:<{width}}  {detail}")
        return failed


def score_frames(graphs, schedule, n_abs, entity, relation, absolute):
    """``(potentials, valids)`` for recorded graphs, packed as training does.

    Separated from the rollout so the arithmetic can be exercised without a
    simulator: it turns graph objects into the tensors the scorer reads and
    returns one potential per frame.
    """
    import torch

    from progress import TaskScheduleReplayPotential

    scorer = TaskScheduleReplayPotential(schedule, n_abs)
    entities = list(schedule.entity_ids)
    row = {ent: i for i, ent in enumerate(entities)}
    slots = set(schedule.slots)

    potentials, valids = [], []
    for graph in graphs:
        node_ent, position = [], {}
        for node in graph.nodes:
            key = ("<ee>" if node.node_type == "ee"
                   else (node.attributes or {}).get("whitelist_key"))
            position[node.node_id] = len(node_ent)
            node_ent.append(entity.encode(key) if key else entity.pad_id)

        rel, abs_, src, dst = [], [], [], []
        for edge in graph.edges:
            if edge.src not in position or edge.dst not in position:
                continue
            slot = (relation.encode(edge.relation),
                    node_ent[position[edge.src]],
                    node_ent[position[edge.dst]])
            if slot not in slots:
                continue
            rel.append(slot[0])
            abs_.append(absolute.encode(edge.label))
            src.append(row[slot[1]])
            dst.append(row[slot[2]])

        value, valid = scorer(
            torch.tensor([entities]),
            torch.tensor(rel, dtype=torch.long),
            torch.tensor(abs_, dtype=torch.long),
            torch.tensor(src, dtype=torch.long),
            torch.tensor(dst, dtype=torch.long),
            torch.zeros(len(rel), dtype=torch.long), 1,
        )
        potentials.append(float(value.item()))
        valids.append(bool(valid.item()))
    return potentials, valids


def report_trace(rep, graphs, potentials, valids, discount: float,
                 n_max=None, e_max=None) -> None:
    """Every acceptance criterion the trace can answer."""
    import torch

    from progress import potential_shaping
    from scenegraph.tools.audit_graph_capacity import (
        frames_from_graphs, occupancy_report,
    )

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

    occupancy = occupancy_report(frames_from_graphs(graphs), n_max, e_max)
    rep.note("peak occupancy (LOWER bound)",
             f"{occupancy['peak_nodes']} node(s), "
             f"{occupancy['peak_edges']} edge(s)")
    rep.note("entity keys seen", f"{occupancy['distinct_entity_keys']}")
    for frame in occupancy["frames_over_n_max"][:3]:
        rep.check("within the node budget", False,
                  f"frame {frame['frame']}: {frame['entity_keys']}")
    for frame in occupancy["frames_over_e_max"][:3]:
        rep.check("within the edge budget", False,
                  f"frame {frame['frame']}: {frame['n_edges']} edges")
    if not occupancy["frames_over_n_max"] and not occupancy["frames_over_e_max"]:
        rep.check("within the budgets given", True,
                  "" if n_max else "(no budget supplied)")


def compile_schedule_from(args, entity):
    from scenegraph.core.schedule import compile_from_source, mshab_schedule_source

    source = mshab_schedule_source(
        args.task, args.subtask, args.configs, args.schedule_dir,
        args.whitelist_dir)
    return compile_from_source(source, entity)


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
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--max-total-steps", type=int, default=4000)
    p.add_argument("--discount", type=float, default=1.0)
    p.add_argument("--n-max", type=int, default=0)
    p.add_argument("--e-max", type=int, default=0)
    p.add_argument("--out", default="", help="Write the trace as JSON.")
    args = p.parse_args(argv)

    from scenegraph.adapters.graph_vocab import (
        build_absolute_vocab, build_entity_vocab, build_relation_vocab,
    )

    entity = build_entity_vocab(args.whitelist_dir)
    schedule = compile_schedule_from(args, entity)
    rep = Report()
    rep.note("entity vocabulary", f"{len(entity.token_to_id)} tokens")
    rep.note("schedule", f"{len(schedule.phases)} phases, "
                         f"{len(schedule.slots)} distinct facts")

    graphs = rollout_graphs(args, rep)
    if graphs is None:
        rep.render()
        return 1
    potentials, valids = score_frames(
        graphs, schedule, len(build_absolute_vocab()), entity,
        build_relation_vocab(), build_absolute_vocab())
    report_trace(rep, graphs, potentials, valids, args.discount,
                 args.n_max or None, args.e_max or None)

    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"potentials": potentials, "valid": valids}, handle)
        rep.note("trace written", args.out)

    failed = rep.render()
    print(f"\n{'all checks passed' if not failed else str(failed) + ' CHECK(S) FAILED'}")
    return 1 if failed else 0


def rollout_graphs(args, rep) -> Optional[List[Any]]:
    """Drive the released policy to one success, recording a graph per step.

    Returns the frames of the first environment that succeeded, or None with a
    reason already reported.
    """
    import numpy as np

    from scenegraph.adapters.policy_loader import load_policy
    from scenegraph.configs.loader import load_config
    from scenegraph.core.graph_builder import GraphBuilder
    from scenegraph.tools.collect_robot_success_states import (
        DEFAULT_CKPT_ALGO, DEFAULT_CKPT_ROOT, _build_env,
    )

    ckpt_root = pathlib.Path(args.ckpt_root or DEFAULT_CKPT_ROOT)
    ckpt_dir = (ckpt_root / (args.algo or DEFAULT_CKPT_ALGO) / args.task
                / args.subtask / args.obj)
    if not ckpt_dir.is_dir():
        rep.check("the policy checkpoint exists", False, str(ckpt_dir))
        return None

    env_args = argparse.Namespace(
        subtask=args.subtask, num_envs=args.num_envs, split="train",
        build_config=args.build_config, max_episode_steps=200,
        sensor_width=128, sensor_height=128, frame_stack=3,
        asset_dir=str(pathlib.Path(args.thresholds).parent))
    venv, collect, _plan = _build_env(
        args.task, args.obj, env_args, ckpt_dir=ckpt_dir,
        capture_raw_obs=True)
    try:
        obs, _ = venv.reset(seed=0, options=dict(reconfigure=True))
        policy = load_policy(str(ckpt_dir), venv, obs, device="cuda")
        cfg = load_config(args.thresholds, task_group=args.task,
                          require_assets=True)
        cfg["whitelist_dir"] = args.whitelist_dir
        cfg["use_target_flag"] = True
        builder = GraphBuilder(venv.unwrapped, cfg, env_idx=0,
                               env_id=f"{args.task}/{args.subtask}",
                               use_target_flag=True)
        frames: List[Any] = []
        for step in range(args.max_total_steps):
            action = policy.act(obs)
            obs, _reward, _term, _trunc, info = venv.step(action)
            raw = getattr(type(venv), "raw_obs", None)
            graph, _m, _c, _r = builder.step(
                raw, step, episode_boundary=(step == 0), need_masks=False)
            frames.append(graph)
            success = info.get("success")
            if success is not None and bool(np.asarray(
                    success.detach().cpu() if hasattr(success, "detach")
                    else success).reshape(-1)[0]):
                rep.note("success at step", str(step))
                return frames
        rep.check("the policy reached a success", False,
                  f"none in {args.max_total_steps} steps")
        return None
    finally:
        venv.close()


if __name__ == "__main__":
    sys.exit(main())
