"""Show what EntityRegistry does with staleness_enabled=false.

With history off, GraphBuilder only ever hands `assign()` the objects that have
at least one segmentation pixel this frame, but the registry still holds an
index for every object it has ever admitted. Nothing releases those indices:
`selector.commit` is skipped, so `evict_expired` has nothing to expire.

Run from the repo root:

    python runs/ghost_slots_demo.py

Needs only numpy (via scenegraph.core.schema). No simulator, no torch.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scenegraph.core.schema import Node
from scenegraph.core.selector import EntityRegistry


def obj(nid, kind):
    return Node(
        node_id=nid, node_type="object", name=nid, visible=True,
        attributes={"whitelist_roles": ["interacted"], "whitelist_key": kind},
    )


def ee():
    return Node(node_id="ee", node_type="ee", name="end_effector")


# The shipped config: n_max=8 -> ee + 7 object slots.
reg = EntityRegistry(n_max=8)

# A plausible camera trace. The robot pans along a counter, turns away, then
# faces the fridge area, then swings back to where it started.
TRACE = [
    (0, [("apple-1", "apple"), ("can-1", "can"), ("can-2", "can")]),
    (1, [("apple-1", "apple"), ("can-1", "can"), ("can-2", "can"),
         ("box-1", "box")]),
    (2, [("box-1", "box"), ("bowl-1", "bowl"), ("bowl-2", "bowl")]),
    (3, [("bowl-1", "bowl"), ("bowl-2", "bowl"), ("mug-1", "mug")]),
    (4, [("mug-1", "mug")]),
    (5, [("shelf-1", "shelf"), ("jar-1", "jar")]),
    (6, [("shelf-1", "shelf"), ("jar-1", "jar"), ("jar-2", "jar")]),
    (7, [("apple-1", "apple"), ("jar-1", "jar")]),
    (8, [("apple-1", "apple"), ("can-1", "can"), ("can-2", "can")]),
]

print("capacity = 7 object slots (n_max=8 minus the ee)\n")
for frame, visible in TRACE:
    nodes = {"ee": ee()}
    for nid, kind in visible:
        nodes[nid] = obj(nid, kind)

    admitted = reg.assign(nodes)          # no target flagged in this scene

    seen = {nid for nid, _ in visible}
    emitted = sorted(k for k in admitted if k != "ee")
    lost = sorted(seen - set(admitted))
    ghosts = sorted(k for k in reg._index if k not in seen)

    print(f"frame {frame}  camera sees: {', '.join(sorted(seen))}")
    print(f"          emitted vertices ({len(emitted)}): {emitted}")
    if lost:
        print(f"          !! VISIBLE BUT NOT A VERTEX: {lost}")
    print(f"          slots wasted on invisible objects ({len(ghosts)}): {ghosts}")
    print()

print(f"total overflow_drops over 9 frames: {reg.overflow_drops}")
print("(this counter is no longer logged: log_graph_overflow_drops was removed"
      " from envs/maniskill.py in 92dddad)")
