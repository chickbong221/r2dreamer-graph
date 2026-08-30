"""Score a recorded episode's graphs with the schedule the runtime would use.

The acceptance criterion nothing else checks: a successful demonstration ends
at potential 1.0, no frame is unreadable, and the shaping term telescopes.

It reads the ``graph_json`` a ``render_paper_frames`` run wrote, packs each
frame exactly as the training path does, and runs the same
``TaskScheduleReplayPotential`` the world model is supervised against. So it
exercises role resolution, edge emission, the phase gates and the weight
arithmetic together, against a real trajectory rather than a constructed one.

    python tests/probes/probe_potential_trace.py data/verify/PickCube-v1_seed0000

Needs torch, because the scorer is a torch module.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from progress import TaskScheduleReplayPotential, potential_shaping  # noqa: E402
from scenegraph.adapters.graph_vocab import (  # noqa: E402
    build_absolute_vocab,
    build_entity_vocab,
    build_relation_vocab,
)
from scenegraph.core.schedule import compile_from_files  # noqa: E402


def load_frames(root):
    paths = sorted(glob.glob(os.path.join(root, "graph_json", "*.json")))
    if not paths:
        sys.exit(f"no graph json under {root}")
    return [json.load(open(p)) for p in paths]


def score(root, configs, schedules, verbose=False):
    frames = load_frames(root)
    env_id = frames[0]["env_id"]
    entity = build_entity_vocab(
        os.path.join(configs, "subtask_whitelists", env_id))
    absolute, relation = build_absolute_vocab(), build_relation_vocab()
    schedule = compile_from_files(env_id, schedules, configs, entity)
    scorer = TaskScheduleReplayPotential(schedule, len(absolute))

    entities = list(schedule.entity_ids)
    row = {e: i for i, e in enumerate(entities)}
    slots = set(schedule.slots)

    potentials, valids = [], []
    for frame in frames:
        # Pack the frame the way graph_pack does: entity ids per node row,
        # then one row per fact whose endpoints both resolved.
        node_ent, position = [], {}
        for node in frame["nodes"]:
            key = ("<ee>" if node["node_type"] == "ee"
                   else (node.get("attributes") or {}).get("whitelist_key"))
            position[node["node_id"]] = len(node_ent)
            node_ent.append(entity.encode(key) if key else entity.pad_id)

        rel, abs_, src, dst = [], [], [], []
        for edge in frame["edges"]:
            if edge["src"] not in position or edge["dst"] not in position:
                continue
            slot = (relation.encode(edge["relation"]),
                    node_ent[position[edge["src"]]],
                    node_ent[position[edge["dst"]]])
            if slot not in slots:
                continue          # a fact the schedule does not read
            rel.append(slot[0])
            abs_.append(absolute.encode(edge["label"]))
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

    return env_id, frames, potentials, valids


def report(env_id, frames, potentials, valids, verbose):
    print(f"\n=== {env_id}   {len(frames)} frames")
    invalid = [i for i, v in enumerate(valids) if not v]
    peak = max(potentials)
    final = potentials[-1]

    if verbose:
        for i, (p, v) in enumerate(zip(potentials, valids)):
            if i % 10 == 0 or i == len(potentials) - 1:
                print(f"    frame {i:4d}  phi={p:.4f}  valid={v}")

    # The shaping term the actor actually receives, with no discounting so a
    # telescoping sum is exactly the endpoint difference.
    phi = torch.tensor(potentials, dtype=torch.float32).reshape(1, -1, 1)
    cont = torch.ones_like(phi)
    shaped = potential_shaping(phi, cont, discount=1.0)
    telescoped = float(shaped.sum().item())
    expected = final - potentials[0]

    rows = [
        ("every frame readable", not invalid,
         "all valid" if not invalid else f"{len(invalid)} invalid: {invalid[:8]}"),
        ("terminal potential is 1.0", abs(final - 1.0) < 1e-4, f"{final:.6f}"),
        ("potential never exceeds 1", peak <= 1.0 + 1e-6, f"peak {peak:.6f}"),
        ("potential starts below 1", potentials[0] < 1.0, f"{potentials[0]:.6f}"),
        ("shaping telescopes", abs(telescoped - expected) < 1e-3,
         f"sum {telescoped:+.6f} vs phi_T - phi_0 {expected:+.6f}"),
    ]
    width = max(len(n) for n, _, _ in rows)
    failed = 0
    for name, ok, detail in rows:
        failed += 0 if ok else 1
        print(f"  [{' ok ' if ok else 'FAIL'}] {name:<{width}}  {detail}")

    # Where the potential actually moves, which is what shaping rewards.
    steps = np.diff(potentials)
    rises = int((steps > 1e-6).sum())
    falls = int((steps < -1e-6).sum())
    print(f"         phi range [{min(potentials):.4f}, {peak:.4f}]  "
          f"rises {rises}  falls {falls}")
    return failed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("roots", nargs="+")
    p.add_argument("--configs", default=str(ROOT / "scenegraph" / "configs"))
    p.add_argument("--schedules",
                   default=str(ROOT / "scenegraph" / "configs" / "schedules"))
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    failed = 0
    for root in args.roots:
        failed += report(*score(root, args.configs, args.schedules), args.verbose)
    print(f"\n{'all checks passed' if not failed else str(failed) + ' CHECK(S) FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
