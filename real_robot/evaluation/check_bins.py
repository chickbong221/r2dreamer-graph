"""Ground the frozen centimetre bins against measured geometry.

    python -m real_robot.evaluation.check_bins --episodes pilot

The bin specification's sizes and thresholds are estimates Gemini read off RGB
video. Once geometry exists, this compares them with measurements: for every
planar-distance and height-offset fact, on every frame where both reference
points are known, the measured value in centimetres is grouped by the label
Gemini gave that frame. Each label reports how many frames carry it, the
median and 10th-90th percentile of what was measured, and the fraction that
falls inside the label's declared range. A label whose measured median lies
outside its declared range is flagged.

The reference points are the ones the prompts define: the gripper's closing
point, the banana's centre, the lid's centre and the centre of the pot's
opening at rim height. Heights are taken along the scene frame's z axis --
the table normal in the table-aligned camera frame, and the robot base's
vertical in the robot frame.

This is a report, not a gate: disagreement means either the thresholds or the
geometry need a look, and the overlays are how to tell which.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..common import add_config_arguments, load_configs, repo_path, utc_now, write_json

REFERENCES = {"ee": "ee", "banana": "banana:center", "lid": "lid:center", "pot": "pot:rim_center"}
RELATIONS = ("planar-distance", "height-offset")


def measured_cm(relation: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if relation == "planar-distance":
        return np.linalg.norm(a[..., :2] - b[..., :2], axis=-1) * 100.0
    return (a[..., 2] - b[..., 2]) * 100.0


def declared_ranges(bins: Mapping[str, Any]) -> Dict[tuple, tuple]:
    return {(item["relation"], item["scope"], item["label"]): (float(item["lower_cm"]), float(item["upper_cm"]))
            for item in bins["spatial_bins"]}


def grounding(spec, annotations: Sequence[Any], geometries: Sequence[Mapping[str, Any]],
              bins: Mapping[str, Any]) -> Dict[str, Any]:
    """Per fact and label: counts, measured percentiles and agreement with the declared range."""
    ranges = declared_ranges(bins)
    pooled: Dict[tuple, List[float]] = {}
    for annotation, geometry in zip(annotations, geometries):
        names = [str(v) for v in geometry["point_names"]]
        for index, fact in enumerate(spec.facts):
            if fact.relation not in RELATIONS:
                continue
            ref_a, ref_b = REFERENCES.get(fact.src), REFERENCES.get(fact.dst)
            if ref_a not in names or ref_b not in names:
                continue
            a, b = names.index(ref_a), names.index(ref_b)
            known = geometry["point_known"][:, a] & geometry["point_known"][:, b]
            values = measured_cm(fact.relation, geometry["points"][:, a], geometry["points"][:, b])
            scope = spec.scope(fact.src, fact.dst)
            for t, label in enumerate(annotation.absolute[index]):
                if label is None or not known[t] or not np.isfinite(values[t]):
                    continue
                pooled.setdefault((fact.relation, fact.src, fact.dst, scope, label), []).append(float(values[t]))
    rows = []
    for (relation, src, dst, scope, label), values in sorted(pooled.items()):
        lower, upper = ranges.get((relation, scope, label), (float("nan"), float("nan")))
        array = np.asarray(values)
        inside = float(np.mean((array >= lower) & (array < upper))) if np.isfinite([lower, upper]).all() else None
        median = float(np.median(array))
        rows.append({
            "fact": f"{relation}({src}, {dst})", "scope": scope, "label": label, "frames": int(array.size),
            "measured_cm": {"p10": float(np.percentile(array, 10)), "median": median,
                            "p90": float(np.percentile(array, 90))},
            "declared_cm": [lower, upper], "inside_fraction": inside,
            "median_outside_declared": bool(inside is not None and not (lower <= median < upper)),
        })
    return {"rows": rows, "flagged": [r for r in rows if r["median_outside_declared"]]}


def main(argv: Optional[Sequence[str]] = None) -> None:
    from ..data.episode_dataset import RawEpisodeSource
    from ..preprocessing.define_bins import load_frozen_bins
    from ..preprocessing.freshness import ArtifactChain, warn_stale

    parser = argparse.ArgumentParser(description="Compare the frozen centimetre bins with measured geometry.")
    parser.add_argument("--episodes", default="pilot")
    parser.add_argument("--mode", choices=("full_episode", "past_only"), default=None)
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    configs = load_configs(["dataset", "annotation", "graph"], args.overrides)
    source = RawEpisodeSource(configs, mode=args.mode)
    spec = source.spec
    bins = load_frozen_bins(configs["dataset"], spec)
    episodes = source.select(args.episodes)
    stale = warn_stale(ArtifactChain(configs, source), episodes, "geometry", "[bins]")
    annotations, geometries, frames = [], [], set()
    for episode in episodes:
        annotations.append(source.annotation(episode))
        geometry = source.geometry(episode)
        geometries.append(geometry)
        frames.add(geometry["meta"]["scene_frame"])
    report = grounding(spec, annotations, geometries, bins["bins"])
    report.update({"created": utc_now(), "episodes": [int(e) for e in episodes], "scene_frames": sorted(frames),
                   "bins_hash": bins["bins_hash"], "stale": {str(e): p for e, p in stale.items()}})
    out = os.path.join(repo_path(configs["dataset"]["paths"]["renders"]), "bins", f"grounding_{source.mode}.json")
    write_json(out, report)
    print(f"[bins] {len(episodes)} episode(s), scene frame(s) {sorted(frames)}")
    for row in report["rows"]:
        flag = "  <- median outside the declared range" if row["median_outside_declared"] else ""
        inside = "-" if row["inside_fraction"] is None else f"{row['inside_fraction']:.0%}"
        m = row["measured_cm"]
        print(f"  {row['fact']:34s} {row['label']:11s} n={row['frames']:5d} measured {m['p10']:6.1f} "
              f"/ {m['median']:6.1f} / {m['p90']:6.1f} cm, declared [{row['declared_cm'][0]:g}, "
              f"{row['declared_cm'][1]:g}), inside {inside}{flag}")
    print(f"[bins] {len(report['flagged'])} label(s) flagged -> {out}")


if __name__ == "__main__":
    main()
