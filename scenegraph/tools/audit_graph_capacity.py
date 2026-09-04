"""How large the graph vocabulary and the row/edge budgets have to be.

Three questions the model configuration cannot be written without, and none of
them answerable by guessing:

* **Vocabulary.** ``model.graph.entity_vocab`` sizes an embedding table, and
  the runtime refuses to start when it is smaller than the mined asset needs.
  The number comes from the asset, never from a remembered value.
* **A static upper bound.** Every member the whitelist admits could stand in
  the graph at once, plus the end effector, plus whichever rows the protected
  contract reserves. That bound is safe and usually loose.
* **What a run actually held.** Retention is unconditional -- once a camera
  has seen a whitelisted object it stays a vertex until reset -- so occupancy
  only grows within an episode, and the honest measurement is over full-length
  episodes rather than a short probe. When a frame exceeds a budget this
  reports the node ids and entity keys that were in it, because "eight nodes"
  does not say which one to look at.

Nothing here writes a capacity. It reports what the evidence supports and
leaves ``n_max``/``e_max`` to a decision made against the final assets.

    python -m scenegraph.tools.audit_graph_capacity \\
        --whitelist-dir scenegraph/configs/subtask_whitelists/tidy_house \\
        --subtask pick --reserved-rows 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence

from scenegraph.core.sites import SITE_PREFIX

# The rows the MS-HAB protected contract pins by meaning: end effector, active
# target, rest site. A task pinning fewer passes its own number.
DEFAULT_RESERVED_ROWS = 3


def vocabulary_report(whitelist_dir: str) -> Dict[str, Any]:
    """Sizes and membership of the entity vocabulary an asset tree implies.

    Built the way the runtime builds it: every ``.json`` in the directory
    contributes its member keys, so a per-target file for an object the policy
    never picks still occupies a row.
    """
    from scenegraph.adapters.graph_vocab import build_entity_vocab

    vocab = build_entity_vocab(whitelist_dir)
    tokens = sorted(vocab.token_to_id, key=lambda t: vocab.token_to_id[t])
    sites = [t for t in tokens if t.startswith(SITE_PREFIX)]
    return {
        "whitelist_dir": whitelist_dir,
        # ``len(vocab)`` counts the pad slot the ids are offset by.
        "entity_vocab_required": len(vocab),
        "tokens": tokens,
        "site_tokens": sites,
        "n_files": sum(1 for n in os.listdir(whitelist_dir)
                       if n.endswith(".json")),
    }


def static_bound(union_path: str,
                 reserved_rows: int = DEFAULT_RESERVED_ROWS) -> Dict[str, Any]:
    """One row per canonical member, plus the end effector.

    Loose in one direction and **not a bound at all** in another. Loose,
    because it assumes every member coexists, which no single build
    configuration need do. Not a bound, because a row is an *instance* and a
    member is a *key*: a scene holding two bowls packs two rows against one
    whitelist key, and ``_row_assignment`` is written for exactly that case.

    So a capacity at or above this number can still overflow, and the only
    thing that settles it is counting simultaneous instances in real frames --
    see ``occupancy_report``'s ``peak_instances_per_key``.
    """
    with open(union_path) as handle:
        payload = json.load(handle)
    members = payload.get("members") or {}
    physical = sorted(k for k in members if not k.startswith(SITE_PREFIX))
    sites = sorted(k for k in members if k.startswith(SITE_PREFIX))
    return {
        "union": union_path,
        "physical_members": physical,
        "site_members": sites,
        # ee + every member. The reserved rows are already among the members
        # (target and site), so they are not added again -- but a reserved row
        # standing empty still costs a slot, which is what ``reserved_rows``
        # accounts for.
        "nodes_one_row_per_key": 1 + len(members),
        "reserved_rows": int(reserved_rows),
        "ordinary_rows_needed": max(0, len(physical) - 1),
        # Stated rather than assumed. Nothing in the assets says how many
        # instances of one key a scene spawns, so nothing here can bound it.
        "assumes_one_instance_per_key": True,
    }


def occupancy_report(frames: Iterable[Dict[str, Any]],
                     n_max: Optional[int] = None,
                     e_max: Optional[int] = None) -> Dict[str, Any]:
    """What a recorded run actually held, and which nodes were in it.

    ``frames`` are dicts with ``node_ids``, ``entity_keys`` and ``n_edges``.
    Peaks are a **lower** bound on what the task needs: retention only grows
    occupancy within an episode, and a run that never looked at a corner of
    the scene never admitted what is in it.
    """
    peak_nodes = peak_edges = 0
    # A row is an instance and an entity key is a category: two bowls in one
    # scene are two rows sharing one key. Nothing in the assets says how many
    # a scene spawns, so this is the only place the question is answerable.
    peak_instances = 1
    peak_duplicate = ""
    peak_frame: Dict[str, Any] = {}
    over_node: List[Dict[str, Any]] = []
    over_edge: List[Dict[str, Any]] = []
    seen: Counter = Counter()
    n_frames = 0

    for frame in frames:
        n_frames += 1
        node_ids = list(frame.get("node_ids") or ())
        keys = list(frame.get("entity_keys") or ())
        edges = int(frame.get("n_edges") or 0)
        seen.update(keys)
        within = Counter(keys)
        if within:
            key, count = within.most_common(1)[0]
            if count > peak_instances:
                peak_instances, peak_duplicate = count, key
        if len(node_ids) > peak_nodes:
            peak_nodes = len(node_ids)
            peak_frame = {"frame": frame.get("frame"),
                          "node_ids": node_ids, "entity_keys": keys}
        peak_edges = max(peak_edges, edges)
        # Named, not counted. "nine nodes against eight" does not say which
        # one to look at, and the answer is usually a sibling instance or a
        # piece of furniture nobody expected to be admitted.
        if n_max is not None and len(node_ids) > int(n_max):
            over_node.append({"frame": frame.get("frame"),
                              "node_ids": node_ids, "entity_keys": keys,
                              "excess": len(node_ids) - int(n_max)})
        if e_max is not None and edges > int(e_max):
            over_edge.append({"frame": frame.get("frame"), "n_edges": edges,
                              "excess": edges - int(e_max)})
    return {
        "n_frames": n_frames,
        "peak_nodes": peak_nodes,
        "peak_edges": peak_edges,
        "peak_instances_per_key": peak_instances,
        "peak_duplicate_key": peak_duplicate,
        "peak_frame": peak_frame,
        "distinct_entity_keys": sorted(seen),
        "entity_key_frequency": dict(seen.most_common()),
        "frames_over_n_max": over_node,
        "frames_over_e_max": over_edge,
        "bound_kind": "lower",
    }


def frames_from_graphs(graphs: Iterable[Any]) -> List[Dict[str, Any]]:
    """Adapt recorded ``Graph`` objects into what ``occupancy_report`` reads."""
    out = []
    for graph in graphs:
        nodes = list(graph.nodes)
        out.append({
            "frame": getattr(graph, "frame", None),
            "node_ids": [n.node_id for n in nodes],
            "entity_keys": [
                "<ee>" if n.node_type == "ee"
                else (n.attributes or {}).get("whitelist_key") or n.node_id
                for n in nodes
            ],
            "n_edges": len(graph.edges),
        })
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--whitelist-dir", required=True)
    parser.add_argument("--subtask", default="pick")
    parser.add_argument("--reserved-rows", type=int,
                        default=DEFAULT_RESERVED_ROWS)
    parser.add_argument("--occupancy-json", default="",
                        help="Frames recorded by a live probe, to fold in.")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.whitelist_dir):
        print(f"[capacity] no such directory: {args.whitelist_dir}",
              file=sys.stderr)
        return 2

    vocab = vocabulary_report(args.whitelist_dir)
    print(f"\n=== {args.whitelist_dir}")
    print(f"  entity_vocab required: {vocab['entity_vocab_required']} "
          f"(from {vocab['n_files']} file(s))")
    print(f"  site tokens: {vocab['site_tokens'] or '<none>'}")

    union = os.path.join(args.whitelist_dir, f"{args.subtask}_all.json")
    if os.path.isfile(union):
        bound = static_bound(union, args.reserved_rows)
        print(f"  rows at one instance per key: "
              f"{bound['nodes_one_row_per_key']} "
              f"(ee + {len(bound['physical_members'])} physical + "
              f"{len(bound['site_members'])} site)")
        print(f"  physical members: {bound['physical_members']}")
    else:
        print(f"  no union at {union}; static bound not computed")

    if args.occupancy_json:
        with open(args.occupancy_json) as handle:
            frames = json.load(handle)
        report = occupancy_report(frames)
        print(f"  observed peak (LOWER bound): {report['peak_nodes']} node(s), "
              f"{report['peak_edges']} edge(s) over {report['n_frames']} frame(s)")
        print(f"  most simultaneous instances of one key: "
              f"{report['peak_instances_per_key']} "
              f"({report['peak_duplicate_key'] or 'none duplicated'})")
        print(f"  distinct entity keys seen: {report['distinct_entity_keys']}")
    else:
        print("  no --occupancy-json: simultaneous instances unmeasured, so "
              "no row count here is an upper bound")

    print("\n  Neither number is a capacity. A key is not a row -- a scene "
          "holding two bowls packs two rows against one key -- so the static "
          "count bounds nothing until live frames say how many instances "
          "coexist. The observed peak is a floor. Setting n_max and e_max is "
          "a decision against the final assets and a per-configuration audit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
