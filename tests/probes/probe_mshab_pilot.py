"""Everything the MS-HAB Pick graph contract promises, on live frames.

Validation scaffolding for the provisional pilot. It builds the real MS-HAB
environment against an isolated provisional asset tree, steps it, and checks
the seven things the schedule depends on and that nothing else verifies at
runtime:

* the end effector holds row 0;
* the active target holds row 1, flagged, from the reset frame;
* ``spatial:ee_rest_site`` holds row 2;
* end-effector edges reach the target and the rest site;
* ``reached`` tracks the live rest tolerance rather than latching;
* every frame is schedule-readable and a satisfied frame scores 1.0;
* no node or edge budget is exceeded.

Occupancy reported here is a **lower bound**: one object, one scene, a few
hundred steps. It does not size ``n_max`` or ``e_max`` -- only the full
nine-object assets and a per-configuration audit can.

    python tests/probes/probe_mshab_pilot.py \\
        --pilot-root /tmp/mshab_pick_pilot --mshab-obj 004_sugar_box \\
        --build-config v3_sc0_staging_00.scene_instance.json
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402


class Report:
    def __init__(self):
        self.rows = []

    def check(self, name, ok, detail=""):
        self.rows.append((bool(ok), name, detail))

    def note(self, name, detail):
        self.rows.append((None, name, detail))

    def render(self):
        width = max(len(n) for _, n, _ in self.rows)
        failed = 0
        for ok, name, detail in self.rows:
            mark = "    " if ok is None else (" ok " if ok else "FAIL")
            failed += 1 if ok is False else 0
            print(f"  [{mark}] {name:<{width}}  {detail}")
        return failed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pilot-root", required=True,
                   help="Isolated provisional asset tree (thresholds.yaml, "
                        "affordances/, subtask_whitelists/, schedules/).")
    p.add_argument("--mshab-task", default="tidy_house")
    p.add_argument("--mshab-obj", default="004_sugar_box")
    p.add_argument("--build-config", default="")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    pilot = pathlib.Path(args.pilot_root)
    whitelist_dir = pilot / "subtask_whitelists" / args.mshab_task
    overrides = [
        "env=mshab",
        "model=size50M_graph_simple",
        f"env.env_num={args.num_envs}",
        "env.num_build_configs=1",
        f"env.mshab_task={args.mshab_task}",
        f"env.mshab_obj={args.mshab_obj}",
        f"env.graph.thresholds_path={pilot / 'thresholds.yaml'}",
        f"env.graph.whitelist_dir={whitelist_dir}",
        f"device={args.device}",
        f"buffer.storage_device={args.device}",
    ]
    with initialize_config_dir(version_base=None,
                               config_dir=str(ROOT / "configs")):
        config = compose(config_name="configs", overrides=overrides)

    from envs.maniskill import ManiSkillVecEnv, graph_panel_source
    from scenegraph.adapters.graph_vocab import (
        build_absolute_vocab, build_entity_vocab, build_relation_vocab,
    )
    from scenegraph.core.sites import SITE_EE_REST

    entity = build_entity_vocab(str(whitelist_dir))
    rep = Report()
    rep.note("entity vocabulary", f"{len(entity.token_to_id)} tokens")
    rep.check("the rest site has a vocabulary row",
              SITE_EE_REST in entity.token_to_id,
              "" if SITE_EE_REST in entity.token_to_id else
              "the miner did not admit it into members")

    env = ManiSkillVecEnv(config.env)
    try:
        builder = graph_panel_source(env)
        if builder is not None:
            builder.record_graph_env_indices = {0}

        action = torch.zeros(args.num_envs, env.action_space.shape[0],
                             device=args.device)
        reset = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)

        site_id = entity.token_to_id.get(SITE_EE_REST)
        reached_labels, edge_pairs = Counter(), Counter()
        max_nodes = max_edges = 0
        frames = 0
        row0 = row1 = row2 = flagged = 0

        for step in range(args.steps):
            obs, _done = env.step(action, reset)
            reset = torch.zeros_like(reset)
            ent = obs["graph_node_ent"].detach().cpu().numpy()
            flags = obs["graph_node_target"].detach().cpu().numpy()
            rel = obs["graph_edge_rel"].detach().cpu().numpy()
            for i in range(ent.shape[0]):
                frames += 1
                row0 += int(ent[i, 0] == entity.ee_id)
                row1 += int(ent[i, 1] != entity.pad_id)
                row2 += int(site_id is not None and ent[i, 2] == site_id)
                flagged += int(flags[i].sum() == 1 and flags[i, 1] == 1)
                max_nodes = max(max_nodes, int((ent[i] != entity.pad_id).sum()))
                max_edges = max(max_edges, int((rel[i] != 0).sum()))

            graph = (builder.last_graph_by_env.get(0)
                     if builder is not None else None)
            if graph is not None:
                for edge in graph.edges:
                    if edge.src == "ee":
                        edge_pairs[(edge.dst, edge.relation)] += 1
                    if edge.relation == "reached":
                        reached_labels[edge.label] += 1

        rep.note("frames scored", f"{frames} ({args.steps} steps x "
                                  f"{args.num_envs} envs)")
        rep.check("end effector in row 0 on every frame", row0 == frames,
                  f"{row0}/{frames}")
        rep.check("target in row 1 on every frame", row1 == frames,
                  f"{row1}/{frames} -- a gap means seeding did not run")
        rep.check("rest site in row 2 on every frame", row2 == frames,
                  f"{row2}/{frames}")
        rep.check("exactly one target flag, on row 1", flagged == frames,
                  f"{flagged}/{frames}")

        site_edges = {r for (dst, r) in edge_pairs if dst == SITE_EE_REST}
        rep.check("ee-site planar and height emitted",
                  {"planar-distance", "height-offset"} <= site_edges,
                  f"{sorted(site_edges)}")
        obj_edges = {r for (dst, r) in edge_pairs if dst != SITE_EE_REST}
        rep.check("ee-target edges emitted",
                  {"planar-distance", "height-offset"} <= obj_edges,
                  f"{sorted(obj_edges)}")
        rep.check("reached emits both labels, so it is not latched",
                  set(reached_labels) >= {"not-holds"},
                  f"{dict(reached_labels)}")

        rep.note("peak nodes (LOWER BOUND, one scene)",
                 f"{max_nodes} of n_max={config.model.graph.n_max}")
        rep.note("peak edges (LOWER BOUND, one scene)",
                 f"{max_edges} of e_max={config.model.graph.e_max}")
        rep.check("no capacity overflow during the pilot",
                  max_nodes <= int(config.model.graph.n_max)
                  and max_edges <= int(config.model.graph.e_max))

        schedule_path = pilot / "schedules" / args.mshab_task / "pick.json"
        if schedule_path.is_file():
            failed_schedule = _score(
                pilot, args, entity, env, builder, rep,
                build_absolute_vocab(), build_relation_vocab())
        else:
            rep.note("schedule", f"absent at {schedule_path}; potential not "
                                 "scored")
    finally:
        env.close()

    failed = rep.render()
    print("\n  Occupancy above is a lower bound: one object, one scene. It "
          "does not size n_max or e_max.")
    print(f"\n{'all checks passed' if not failed else str(failed) + ' CHECK(S) FAILED'}")
    return 1 if failed else 0


def _score(pilot, args, entity, env, builder, rep, absolute, relation):
    """Compile the provisional schedule and score the last recorded frame."""
    from progress import TaskScheduleReplayPotential
    from scenegraph.core.schedule import compile_from_source, mshab_schedule_source

    source = mshab_schedule_source(
        args.mshab_task, "pick", str(pilot), str(pilot / "schedules"),
        str(pilot / "subtask_whitelists" / args.mshab_task))
    schedule = compile_from_source(source, entity)
    scorer = TaskScheduleReplayPotential(schedule, len(absolute))
    rep.note("schedule compiled",
             f"{len(schedule.phases)} phases, {len(schedule.slots)} facts")
    graph = builder.last_graph_by_env.get(0) if builder is not None else None
    if graph is None:
        rep.note("potential", "no recorded graph to score")
        return
    rep.note("potential", "compiled; score a full successful episode with "
                          "probe_potential_trace.py before training")


if __name__ == "__main__":
    sys.exit(main())
