"""Annotate the two-cube cup task from RGB and recorded gripper actions.

Sustained gripper transitions identify every grasp attempt, including retries.
Current-frame colour measurements assign the moving cube and measure spatial
and compatibility relations with past-only smoothing. Containment requires a
release plus visual cup alignment. The output is the standard validated
annotation consumed by ``pack_graphs``.

    python -m real_robot.preprocessing.annotate_cups_rgb --episodes cubes_in_cup
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..common import add_config_arguments, load_configs, stable_hash, utc_now
from ..graphs.validate import ValidationSettings, build_annotation, fact_ids
from .annotate_episode import EpisodeAnnotator
from .annotate_stack_rgb import (_box1000, _components, _pick, _runs,
                                 _temporal, causal_smooth, compatibility, decimate_tracks,
                                 distance_label, episode_actions,
                                 measured_distance_cm, track_geometry)
from .prepare_videos import prepared_status


def _cup_components(rgb: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # The cup is pale cyan. The saturation floor excludes the white table;
    # the ceiling separates it from the turquoise robot body.
    mask = cv2.inRange(hsv, (78, 18, 75), (112, 155, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    parts = [tuple(int(v) for v in row) for row in stats[1:]
             if 500 <= int(row[4]) <= 30000]
    return sorted(parts, key=lambda box: box[4], reverse=True)


def _pick_cup(parts, camera: str, previous=None):
    candidates = []
    for x, y, w, h, area in parts:
        ratio = w / max(h, 1)
        if not (20 <= w <= 300 and 20 <= h <= 300 and .3 <= ratio <= 3.2):
            continue
        cx, cy = x + w / 2, y + h / 2
        if camera == "top":
            # The pale cup is the largest low-saturation cyan component in the
            # fixed view. Nearest-neighbour tracking can drift to a small pale
            # robot highlight during occlusion and never recover.
            score = -float(area)
        elif previous is not None:
            score = float(np.hypot(cx - previous[0], cy - previous[1]))
        else:
            score = -.02 * area
        candidates.append((score, (x, y, x + w, y + h), (cx, cy)))
    return min(candidates, default=(None, None, None), key=lambda item: item[0])[1:]


def track(source, episode: int, every: int = 15):
    n = source.lengths()[episode]
    frames = sorted(set(range(0, n, every)) | {n - 1})
    tracks: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    centers: Dict[Tuple[str, str], Dict[int, Tuple[float, float]]] = {}
    for camera in source.cameras():
        wanted = set(frames)
        previous = {name: None for name in ("red_cube", "blue_cube", "ee", "cup")}
        for index, rgb in source.frames(episode, camera):
            if index not in wanted:
                continue
            for entity in ("red_cube", "blue_cube", "ee", "cup"):
                if entity == "cup":
                    box, center = _pick_cup(_cup_components(rgb), camera, previous[entity])
                else:
                    box, center = _pick(_components(rgb, entity), entity, camera,
                                        previous[entity])
                if center is not None:
                    previous[entity] = center
                    centers.setdefault((entity, camera), {})[index] = center
                tracks.setdefault((entity, camera), []).append({
                    "frame": index, "visible": box is not None,
                    "box_2d": _box1000(box, rgb.shape)})
            tracks.setdefault(("table", camera), []).append({
                "frame": index, "visible": True, "box_2d": [0, 0, 1000, 1000]})
    return tracks, centers


def _persistent(mask: np.ndarray, start: int, width: int = 5) -> Optional[int]:
    for index in range(max(0, start), len(mask) - width + 1):
        if bool(np.all(mask[index:index + width])):
            return index
    return None


def grasp_cycles(gripper: np.ndarray) -> List[Tuple[int, int]]:
    lo, hi = float(np.quantile(gripper, .05)), float(np.quantile(gripper, .95))
    span = max(hi - lo, 1e-6)
    closed = gripper <= lo + .35 * span
    # A cube only needs a partial jaw opening to be released. Several valid
    # demonstrations finish around 20 rather than returning to the initial
    # 38--45 command, so an episode-global 55% threshold misses the release.
    opened = gripper >= lo + .30 * span
    cycles: List[Tuple[int, int]] = []
    # A cycle starts on a sustained open->closed transition. Starting inside
    # an already-closed run (episode 108 begins that way) is not a grasp.
    cursor = 1
    while cursor < len(gripper) - 5:
        grasp = None
        for index in range(cursor, len(gripper) - 4):
            if not closed[index - 1] and bool(np.all(closed[index:index + 5])):
                grasp = index
                break
        if grasp is None:
            break
        release = _persistent(opened, grasp + 8)
        if release is None:
            release = len(gripper) - 1
        cycles.append((grasp, release))
        cursor = release + 8
    if len(cycles) < 2:
        raise ValueError(f"expected at least two grasp/release cycles, found {cycles}")
    return cycles


def assign_cycles(geometry, cycles) -> Tuple[List[str], Tuple[str, str]]:
    """Assign every attempt to the cube that visibly moves during it.

    Some demonstrations contain a failed grasp and retry. Movement over the
    current close/open segment distinguishes that from the stationary cube;
    the decision is available at release and uses no later episode frames.
    """
    assigned: List[str] = []
    for grasp, release in cycles:
        movement = {}
        for cube in ("red_cube", "blue_cube"):
            first = geometry[(cube, "top")][grasp]
            last = geometry[(cube, "top")][min(release, len(geometry[(cube, "top")]) - 1)]
            if np.isfinite(first).all() and np.isfinite(last).all():
                movement[cube] = float(np.linalg.norm(_centre_xy(first) - _centre_xy(last)))
            else:
                movement[cube] = -1.0
        assigned.append(max(movement, key=movement.get))
    if len(cycles) == 2 and assigned[0] == assigned[1]:
        assigned[1] = "red_cube" if assigned[0] == "blue_cube" else "blue_cube"
    order = []
    for cube in assigned:
        if cube not in order:
            order.append(cube)
    for cube in ("red_cube", "blue_cube"):
        if cube not in order:
            order.append(cube)
    return assigned, (order[0], order[1])


def _centre_xy(box: np.ndarray) -> np.ndarray:
    return np.asarray([(box[1] + box[3]) / 2, (box[0] + box[2]) / 2])


def _direction(label: str, fact, positive_src: str) -> str:
    if fact.src == positive_src:
        return label
    inverse = {"above": "below", "far-above": "far-below", "below": "above",
               "far-below": "far-above", "level": "level"}
    return inverse[label]


def make_answer(spec, n: int, cycles, assignments, order, tracks,
                box_every: int = 15) -> Dict[str, Any]:
    geometry = track_geometry(tracks, n)
    attempts = {cube: [] for cube in ("red_cube", "blue_cube")}
    for cycle, cube in zip(cycles, assignments):
        attempts[cube].append(cycle)
    distances = {
        fact.key: causal_smooth(np.asarray([
            measured_distance_cm(geometry, fact.src, fact.dst, t)
            for t in range(n)]))
        for fact in spec.facts if fact.relation == "planar-distance"
    }
    cube_cup = {
        cube: causal_smooth(np.asarray([
            measured_distance_cm(geometry, cube, "cup", t)
            for t in range(n)]))
        for cube in ("red_cube", "blue_cube")
    }
    cube_cube = causal_smooth(np.asarray([
        measured_distance_cm(geometry, "red_cube", "blue_cube", t)
        for t in range(n)]))
    contained_series: Dict[str, np.ndarray] = {}
    holding_series: Dict[str, np.ndarray] = {}
    for cube in ("red_cube", "blue_cube"):
        holding = np.zeros(n, dtype=bool)
        for grasp, release in attempts[cube]:
            holding[grasp:release] = True
        holding_series[cube] = holding
        contained = np.zeros(n, dtype=bool)
        latched = False
        releases = {release for _, release in attempts[cube]}
        grasps = {grasp for grasp, _ in attempts[cube]}
        has_released = False
        for t in range(n):
            if t in grasps:
                latched = False
            if t in releases:
                has_released = True
            # Below-rim depth is not available without depth, so release plus
            # current top-view alignment is the observable containment test.
            # The pale cup is often partially cropped or occluded. The loose
            # 15 cm image-scale gate is applied only after a causal release;
            # a later retry grasp clears the latch again.
            if has_released and not holding[t] and cube_cup[cube][t] < 15.0:
                latched = True
            contained[t] = latched
        contained_series[cube] = contained

    series: Dict[Tuple[str, str, str], List[str]] = {}
    for fact in spec.facts:
        pair, relation = {fact.src, fact.dst}, fact.relation
        values = []
        for t in range(n):
            cube = next((name for name in ("red_cube", "blue_cube") if name in pair), None)
            if cube:
                holding = bool(holding_series[cube][t])
                contained = bool(contained_series[cube][t])
            else:
                holding = contained = False
            distance = float(distances.get(fact.key, np.full(n, 1000.0))[t])

            if relation == "grasp":
                label = "holds" if holding else "not-holds"
            elif relation == "contact":
                if "ee" in pair and cube:
                    label = "holds" if holding else "not-holds"
                elif "cup" in pair and cube:
                    label = "holds" if contained else "not-holds"
                elif pair == {"red_cube", "blue_cube"}:
                    together = (contained_series["red_cube"][t] and
                                contained_series["blue_cube"][t] and
                                cube_cube[t] < 3.0)
                    label = "holds" if together else "not-holds"
                else:
                    label = "not-holds"
            elif relation == "support":
                if pair == {"cup", "table"}:
                    holder = "table"
                    label = "src-holds" if fact.src == holder else "dst-holds"
                else:
                    supported = not holding and not contained
                    holder = "table"
                    label = (("src-holds" if fact.src == holder else "dst-holds")
                             if supported else "not-holds")
            elif relation == "contain":
                holder = "cup"
                label = (("src-holds" if fact.src == holder else "dst-holds")
                         if contained else "not-holds")
            elif relation == "planar-distance":
                known_contact = (("ee" in pair and cube is not None and holding) or
                                 ("cup" in pair and cube is not None and contained))
                label = ("very-near" if known_contact else
                         distance_label(distance, ee_object="ee" in pair))
            elif relation == "height-offset":
                if "ee" in pair and cube:
                    raw = "level" if holding else "above"
                    label = _direction(raw, fact, "ee")
                elif "cup" in pair and cube:
                    raw = "far-above" if holding else "below"
                    label = _direction(raw, fact, cube)
                elif pair == {"ee", "cup"}:
                    label = _direction("above", fact, "ee")
                else:
                    label = "level"
            elif relation == "grasp-compatibility":
                label = "match" if holding else compatibility(distance, ee_object=True)
            elif relation == "contact-compatibility":
                label = "match" if holding or contained else compatibility(
                    distance, ee_object="ee" in pair)
            elif relation == "contain-compatibility":
                label = "match" if contained else compatibility(
                    float(cube_cup[cube][t]), ee_object=False)
            else:
                label = "unobserved"
            values.append(label)
        series[fact.key] = values

    facts = []
    for fid, fact in zip(fact_ids(spec), spec.facts):
        absolute = series[fact.key]
        temporal = _temporal(absolute, spec.temporal_window, fact.relation) if fact.temporal else []
        entry = {"fact": fid, "absolute": _runs(absolute), "temporal": []}
        if temporal:
            entry["temporal"] = _runs(temporal[spec.temporal_window:])
            for interval in entry["temporal"]:
                interval["start"] += spec.temporal_window
                interval["end"] += spec.temporal_window
        facts.append(entry)
    target_values = []
    attempt = 0
    for t in range(n):
        while attempt + 1 < len(cycles) and t >= cycles[attempt][1]:
            attempt += 1
        target_values.append(assignments[attempt])
    targets = [{"start": row["start"], "end": row["end"], "object": row["label"]}
               for row in _runs(target_values)]
    return {
        "active_target": targets,
        "facts": facts,
        "boxes": [{"entity": entity, "camera": camera, "keyframes": keyframes}
                  for (entity, camera), keyframes in sorted(
                      decimate_tracks(tracks, box_every, n).items())],
        "notes": (f"Causal RGB/action-derived annotation; image-measured distances; "
                  f"order={order[0]},{order[1]}; cycles={cycles}; "
                  f"assignments={assignments}."),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    from ..data.source import LeRobotSource

    parser = argparse.ArgumentParser(description="Annotate two-cube cup episodes from RGB/actions.")
    parser.add_argument("--episodes", default="cubes_in_cup")
    parser.add_argument("--force", action="store_true")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    configs = load_configs(["dataset", "annotation", "graph", "labels"], args.overrides)
    source = LeRobotSource(configs)
    annotator = EpisodeAnnotator(configs, source)
    settings = ValidationSettings.from_config(configs["annotation"], annotator.stride)
    for episode in source.select(args.episodes):
        spec = source.spec(episode)
        if spec.task != "cubes_in_cup":
            print(f"[cups-rgb] episode {episode}: skipped ({spec.task})", flush=True)
            continue
        path = annotator.path(episode)
        if os.path.isfile(path) and not args.force:
            print(f"[cups-rgb] episode {episode}: exists, use --force to replace", flush=True)
            continue
        prepared, reason = prepared_status(source, episode, configs["annotation"]["videos"])
        if prepared is None:
            print(f"[cups-rgb] episode {episode}: skipped, videos {reason}", flush=True)
            continue
        actions = episode_actions(source, episode)
        cycles = grasp_cycles(actions[:, -1])
        tracks, _ = track(source, episode, 1)
        geometry = track_geometry(tracks, len(actions))
        assignments, order = assign_cycles(geometry, cycles)
        answer = make_answer(spec, len(actions), cycles, assignments, order, tracks,
                             settings.box_every)
        annotation = build_annotation(spec, episode_index=episode, n_frames=len(actions),
                                      fps=source.fps(), answer=answer, settings=settings)
        annotation.input_identity = annotator.input_identity(spec, prepared)
        annotation.provenance = {
            "created": utc_now(), "producer": "annotate_cups_rgb",
            "method": "causal HSV measurements plus current gripper-action state",
            "answer_hash": stable_hash(answer), "cycles": cycles,
            "cycle_assignments": assignments, "cube_order": order,
        }
        annotator.save(annotation)
        print(f"[cups-rgb] episode {episode}: {'valid' if annotation.valid else 'INVALID'} "
              f"order={order} assignments={assignments} cycles={cycles} "
              f"issues={len(annotation.issues)}", flush=True)


if __name__ == "__main__":
    main()
