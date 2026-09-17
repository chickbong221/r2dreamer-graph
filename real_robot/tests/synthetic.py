"""A scripted kitchen episode with known milestones, for tests.

Frames, for the default length of 170:

====================  =========================================================
0-29                  gripper approaches the banana, open
30                    grasp (label and gripper)
30-59                 banana carried towards the pot entry
60-69                 banana over the pot opening, lowered
70                    the pot contains the banana
75                    release: gripper opens, grasp label clears
85-109                gripper approaches the lid handle
110                   lid grasped
110-139               lid carried onto the pot
140                   the pot supports the lid
145                   lid released
====================  =========================================================

With an 8-frame settling window the lid has been still for eight frames, the
release included, at frame 152: completion verifies there.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..common import load_config
from ..graphs.schema import GraphSpec
from ..graphs.validate import build_annotation, holder_label
from ..rewards.kitchen import RewardInputs

N = 170
GRASP_BANANA, IN_POT, RELEASE_BANANA = 30, 70, 75
GRASP_LID, LID_ON_POT, RELEASE_LID = 110, 140, 145
SETTLE_FRAMES = 8
EXPECTED_COMPLETION = RELEASE_LID + SETTLE_FRAMES - 1


def graph_spec() -> GraphSpec:
    return GraphSpec.from_config(load_config("graph"))


def reward_config(**changes) -> dict:
    cfg = load_config("reward")
    for key, value in changes.items():
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return cfg


def geometry(n: int = N) -> Dict[str, np.ndarray]:
    """Built at full length and cut, so a truncated episode is a true prefix."""
    total = max(n, N)
    g = {
        "d_gripper_banana": np.concatenate([np.linspace(0.30, 0.02, GRASP_BANANA),
                                            np.full(total - GRASP_BANANA, 0.02)]),
        "d_banana_pot_entry": np.concatenate([np.full(GRASP_BANANA, 0.30), np.linspace(0.30, 0.05, 30),
                                              np.linspace(0.05, 0.0, total - 60)]),
        "banana_placement_error": np.concatenate([np.full(60, 0.20), np.linspace(0.05, 0.01, 10),
                                                  np.full(total - 70, 0.01)]),
        "d_gripper_lid_handle": np.concatenate([np.full(85, 0.40), np.linspace(0.30, 0.02, GRASP_LID - 85),
                                                np.full(total - GRASP_LID, 0.02)]),
        "lid_lateral_error": np.concatenate([np.full(GRASP_LID, 0.25),
                                             np.linspace(0.15, 0.01, LID_ON_POT - GRASP_LID),
                                             np.full(total - LID_ON_POT, 0.01)]),
        "lid_height_above_rim": np.concatenate([np.full(GRASP_LID, -0.05),
                                                np.linspace(0.10, 0.0, LID_ON_POT - GRASP_LID),
                                                np.full(total - LID_ON_POT, 0.0)]),
        "banana_speed": np.concatenate([np.zeros(GRASP_BANANA), np.full(RELEASE_BANANA - GRASP_BANANA, 0.2),
                                        np.zeros(total - RELEASE_BANANA)]),
        "lid_speed": np.concatenate([np.zeros(GRASP_LID), np.full(RELEASE_LID - GRASP_LID, 0.2),
                                     np.zeros(total - RELEASE_LID)]),
    }
    return {k: v[:n].astype(np.float64) for k, v in g.items()}


def closure(n: int = N) -> np.ndarray:
    c = np.zeros(max(n, N))
    c[GRASP_BANANA:RELEASE_BANANA] = 0.6
    c[GRASP_LID:RELEASE_LID] = 0.6
    return c[:n]


def reward_inputs(n: int = N, **overrides) -> RewardInputs:
    frames = np.arange(n)
    values = dict(
        **geometry(n),
        banana_grasp_label=(frames >= GRASP_BANANA) & (frames < RELEASE_BANANA),
        lid_grasp_label=(frames >= GRASP_LID) & (frames < RELEASE_LID),
        banana_in_pot_label=frames >= IN_POT,
        lid_on_pot_label=frames >= LID_ON_POT,
        gripper_closure=closure(n),
        observed_completion_frame=EXPECTED_COMPLETION if n > EXPECTED_COMPLETION else -1,
        observed_success=n > EXPECTED_COMPLETION,
    )
    values.update(overrides)
    return RewardInputs(**values)


def annotation(spec: Optional[GraphSpec] = None, n: int = N, target_switch: int = RELEASE_BANANA + 1,
               raw: bool = False):
    """A valid annotation whose milestone labels match :func:`reward_inputs`.

    ``raw=True`` returns the answers instead: ``(events, absolute, temporal)``.
    """
    spec = spec or graph_spec()
    K = spec.temporal_window
    frames = np.arange(n)
    absolute, temporal = [], []

    def add(relation, src, dst, labels):
        for start, end, value in _runs(labels):
            absolute.append({"relation": relation, "src": src, "dst": dst, "label": value,
                             "start_frame": start, "end_frame": end})

    grasp_banana = np.where((frames >= GRASP_BANANA) & (frames < RELEASE_BANANA), "holds", "not-holds")
    grasp_lid = np.where((frames >= GRASP_LID) & (frames < RELEASE_LID), "holds", "not-holds")
    contain = np.where(frames >= IN_POT, holder_label(spec, "pot", "banana"), "not-holds")
    support = np.where(frames >= LID_ON_POT, holder_label(spec, "pot", "lid"), "not-holds")
    for fact in spec.facts:
        if (fact.relation, fact.src, fact.dst) == ("grasp", "ee", "banana"):
            add(fact.relation, fact.src, fact.dst, grasp_banana)
        elif (fact.relation, fact.src, fact.dst) == ("grasp", "ee", "lid"):
            add(fact.relation, fact.src, fact.dst, grasp_lid)
        elif fact.relation == "contain":
            add(fact.relation, fact.src, fact.dst, contain)
        elif fact.relation == "support" and {fact.src, fact.dst} == {"lid", "pot"}:
            add(fact.relation, fact.src, fact.dst, support)
        else:
            add(fact.relation, fact.src, fact.dst, np.full(n, spec.legal_labels(fact.relation)[0]))
        if fact.temporal and n > K:
            temporal.append({"relation": fact.relation, "src": fact.src, "dst": fact.dst, "label": "stable",
                             "start_frame": K, "end_frame": n - 1})
    success = n > EXPECTED_COMPLETION
    events_list = [e for e in (
        {"type": "grasp", "object": "banana", "frame": GRASP_BANANA},
        {"type": "place", "object": "banana", "frame": IN_POT},
        {"type": "release", "object": "banana", "frame": RELEASE_BANANA},
        {"type": "grasp", "object": "lid", "frame": GRASP_LID},
        {"type": "lid_seated", "object": "lid", "frame": LID_ON_POT},
        {"type": "release", "object": "lid", "frame": RELEASE_LID},
        {"type": "task_complete", "object": "lid", "frame": EXPECTED_COMPLETION},
    ) if e["frame"] < n and (e["type"] != "task_complete" or success)]
    events = {
        "events": events_list,
        "active_target": [{"object": "banana", "start_frame": 0, "end_frame": min(target_switch, n) - 1}]
        + ([{"object": "lid", "start_frame": target_switch, "end_frame": n - 1}] if target_switch < n else []),
        "keyframes": anchors(spec, n, [e["frame"] for e in events_list]),
        "outcome": {"success": success, "banana_in_pot_at_end": n > IN_POT, "lid_closed_at_end": n > LID_ON_POT,
                    "completion_frame": EXPECTED_COMPLETION if success else -1,
                    "failure_reason": "" if success else "recording ends first"},
    }
    if raw:
        return events, absolute, temporal
    return build_annotation(spec, episode_index=0, n_frames=n, fps=15.0, mode="full_episode",
                            events_raw=events, absolute_raw={"intervals": absolute},
                            temporal_raw={"intervals": temporal}, provenance={"model": "synthetic"},
                            spec_identity=spec.identity())


# Bare-table points [y, x] spread across the image, so they define a plane.
TABLE_POINTS = ([700, 150], [700, 850], [950, 500], [900, 200])


def anchors(spec: GraphSpec, n: int, event_frames, every: int = 15, cameras=None):
    """Keyframes for every entity in every camera: at frame 0, every ``every`` frames, the last frame and each event."""
    frames = sorted(set(range(0, n, every)) | {n - 1} | {int(f) for f in event_frames})

    def point(entity: str, index: int):
        return list(TABLE_POINTS[index % len(TABLE_POINTS)]) if entity == "table" else [200, 200 + 10 * index]

    return [{"frame": f, "camera": camera, "object": e.id, "visible": True,
             "box_2d": [600, 100, 1000, 900] if e.id == "table" else [100, 100, 300, 300],
             "points": [{"name": p, "visible": True, "point": point(e.id, i)}
                        for i, p in enumerate(spec.points.get(e.id, ()))]}
            for f in frames for e in spec.entities for camera in (cameras or spec.cameras)]


def relations_answer(spec: GraphSpec, absolute, temporal, disagreements=()):
    """Named interval lists -> a relations answer keyed by fact id, as Gemini returns it."""
    from ..graphs.validate import fact_ids, _interval_key

    by_key = {tuple(f.key): fid for fid, f in zip(fact_ids(spec), spec.facts)}
    facts = {fid: {"fact": fid, "absolute": [], "temporal": []} for fid in fact_ids(spec)}
    for kind, items in (("absolute", absolute), ("temporal", temporal)):
        for item in items:
            fid = by_key[_interval_key(spec, item)]
            facts[fid][kind].append({"start": item["start_frame"], "end": item["end_frame"], "label": item["label"]})
    return {"facts": list(facts.values()), "pass1_disagreements": list(disagreements)}


def _runs(values):
    start = 0
    values = list(values)
    for t in range(1, len(values) + 1):
        if t == len(values) or values[t] != values[start]:
            yield start, t - 1, str(values[start])
            start = t
