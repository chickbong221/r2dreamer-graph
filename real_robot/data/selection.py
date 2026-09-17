"""Which episodes train, and which of those are watched.

    python -m real_robot.data.selection create
    python -m real_robot.data.selection show

Every episode of the pinned dataset trains. There is no validation or test
allocation: policy performance is measured on the robot, not on recorded
episodes.

A fixed subset of those same episodes -- the *diagnostic* episodes -- is where
fitting and pipeline behaviour are watched: world-model diagnostics, the
diagnostic rows of the latent cache, policy and progress-head diagnostics. They
stay in training. Nothing measured on them is evidence of generalisation, and
every report that uses them says so.

The subset is chosen once and saved as episode ids in
``selections/selection_<version>.json``: spread across episode lengths and,
when valid full-episode annotations exist for every episode, covering each
outcome those annotations record. A saved version is never rewritten with
different contents. A different subset is a new version, and every dataset,
checkpoint and cache records the version it was built on.
"""

from __future__ import annotations

import argparse
import collections
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..common import add_config_arguments, load_configs, read_json, repo_path, stable_hash, utc_now, write_json

SELECTION_FORMAT = "real_robot/episode-selection-v1"
DIAGNOSTIC_NOTE = ("Diagnostic episodes are training episodes. What is measured on them describes fitting and "
                   "pipeline behaviour, not generalisation.")


# --------------------------------------------------------------------------- #
# Choosing
# --------------------------------------------------------------------------- #
def _by_length(episodes: Sequence[int], lengths: Mapping[int, int]) -> List[int]:
    return sorted((int(e) for e in episodes), key=lambda e: (int(lengths[e]), e))


def _nearest_unused(ordered: Sequence[int], position: int, used: set) -> int:
    for delta in range(len(ordered)):
        for candidate in (position - delta, position + delta):
            if 0 <= candidate < len(ordered) and ordered[candidate] not in used:
                return ordered[candidate]
    raise ValueError("every episode is already chosen")


def spread_by_length(episodes: Sequence[int], lengths: Mapping[int, int], count: int) -> List[int]:
    """``count`` episodes at evenly spaced positions in length order, shortest and longest included."""
    ordered = _by_length(episodes, lengths)
    count = min(int(count), len(ordered))
    if count <= 0:
        return []
    if count == 1:
        return [ordered[len(ordered) // 2]]
    chosen: List[int] = []
    used: set = set()
    for position in np.linspace(0, len(ordered) - 1, count):
        episode = _nearest_unused(ordered, int(round(float(position))), used)
        used.add(episode)
        chosen.append(episode)
    return chosen


def cover_outcomes(chosen: Sequence[int], episodes: Sequence[int], lengths: Mapping[int, int],
                   outcomes: Mapping[int, str]) -> Tuple[List[int], List[Dict[str, Any]]]:
    """Swap picks until every outcome appears, keeping the shortest and longest episode.

    Rarer outcomes are placed first. Each missing outcome contributes its
    median-length episode, replacing the pick closest to it in length among
    picks whose own outcome would still be represented.
    """
    chosen = list(chosen)
    replacements: List[Dict[str, Any]] = []
    sizes = collections.Counter(outcomes[e] for e in episodes)
    for outcome in sorted(sizes, key=lambda o: (sizes[o], o)):
        if any(outcomes[e] == outcome for e in chosen):
            continue
        members = _by_length([e for e in episodes if outcomes[e] == outcome], lengths)
        candidate = members[len(members) // 2]
        counts = collections.Counter(outcomes[e] for e in chosen)
        ordered = _by_length(chosen, lengths)
        protected = {ordered[0], ordered[-1]} if len(ordered) > 2 else set()
        swappable = [e for e in chosen if counts[outcomes[e]] > 1 and e not in protected]
        if not swappable:
            swappable = [e for e in chosen if counts[outcomes[e]] > 1]
        if not swappable:
            break      # more outcomes than diagnostic slots
        removed = min(swappable, key=lambda e: (abs(int(lengths[e]) - int(lengths[candidate])), e))
        chosen[chosen.index(removed)] = candidate
        replacements.append({"removed": int(removed), "added": int(candidate), "outcome": outcome})
    return chosen, replacements


def outcome_label(outcome: Mapping[str, Any]) -> str:
    return "success" if bool(outcome.get("success", False)) else "failure"


def make_selection(episodes: Sequence[int], lengths: Mapping[int, int], count: int, version: str,
                   outcomes: Optional[Mapping[int, str]] = None, outcome_source: Optional[str] = None,
                   outcome_note: Optional[str] = None, source_revision: Optional[str] = None) -> Dict[str, Any]:
    training = sorted(int(e) for e in episodes)
    chosen = spread_by_length(training, lengths, count)
    replacements: List[Dict[str, Any]] = []
    if outcomes is not None:
        missing = [e for e in training if e not in outcomes]
        if missing:
            raise ValueError(f"no outcome for episodes {missing[:10]}")
        chosen, replacements = cover_outcomes(chosen, training, lengths, outcomes)
    diagnostic = sorted(chosen)
    data = {
        "format": SELECTION_FORMAT,
        "version": str(version),
        "created": utc_now(),
        "source_revision": source_revision,
        "training": training,
        "diagnostic": diagnostic,
        "diagnostic_in_training": True,
        "rule": {
            "count": int(count),
            "lengths": "evenly spaced positions in length order; shortest and longest included",
            "outcomes_used": outcomes is not None,
            "outcome_source": outcome_source,
            "outcome_note": outcome_note,
            "outcome_counts": dict(collections.Counter(outcomes.values())) if outcomes is not None else None,
            "diagnostic_outcomes": ({str(e): outcomes[e] for e in diagnostic} if outcomes is not None else None),
            "replacements": replacements,
        },
        "diagnostic_lengths": {str(e): int(lengths[e]) for e in diagnostic},
        "note": DIAGNOSTIC_NOTE,
    }
    problems = selection_problems(data, training)
    if problems:
        raise AssertionError("; ".join(problems))
    return data


# --------------------------------------------------------------------------- #
# Checking and loading
# --------------------------------------------------------------------------- #
def selection_problems(data: Mapping[str, Any], available: Optional[Sequence[int]] = None) -> List[str]:
    problems = []
    if data.get("format") != SELECTION_FORMAT:
        problems.append(f"format is {data.get('format')!r}, expected {SELECTION_FORMAT!r}")
    training = [int(e) for e in data.get("training", [])]
    diagnostic = [int(e) for e in data.get("diagnostic", [])]
    if not training:
        problems.append("no training episodes")
    repeated = sorted(e for e, n in collections.Counter(training).items() if n > 1)
    if repeated:
        problems.append(f"episodes listed more than once in training: {repeated}")
    if not diagnostic:
        problems.append("no diagnostic episodes")
    repeated = sorted(e for e, n in collections.Counter(diagnostic).items() if n > 1)
    if repeated:
        problems.append(f"episodes listed more than once in diagnostic: {repeated}")
    outside = sorted(set(diagnostic) - set(training))
    if outside:
        problems.append(f"diagnostic episodes that are not training episodes: {outside}")
    if available is not None:
        available = set(int(e) for e in available)
        missing = sorted(available - set(training))
        extra = sorted(set(training) - available)
        if missing:
            problems.append(f"episodes of the dataset missing from training: {missing[:20]}")
        if extra:
            problems.append(f"training lists episodes the dataset does not have: {extra[:20]}")
    return problems


def selection_path(dataset_cfg: Mapping[str, Any], version: Optional[str] = None) -> str:
    version = version or dataset_cfg["selection"]["version"]
    return os.path.join(repo_path(dataset_cfg["paths"]["selections"]), f"selection_{version}.json")


def load_selection(dataset_cfg: Mapping[str, Any], available: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    path = selection_path(dataset_cfg)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no episode selection at {path}; run `python -m real_robot.data.selection create`"
        )
    data = read_json(path)
    problems = selection_problems(data, available)
    if problems:
        raise ValueError(f"{path}:\n  " + "\n  ".join(problems))
    return data


def selection_identity(selection: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "version": str(selection["version"]),
        "training": stable_hash([int(e) for e in selection["training"]]),
        "training_count": len(selection["training"]),
        "diagnostic": [int(e) for e in selection["diagnostic"]],
        "diagnostic_in_training": True,
    }


def extremes_and_median(episodes: Sequence[int], lengths: Mapping[int, int]) -> List[int]:
    ordered = _by_length(episodes, lengths)
    if len(ordered) < 3:
        return list(ordered)
    picks = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    return list(dict.fromkeys(picks))


def pilot_episodes(dataset_cfg: Mapping[str, Any], lengths: Mapping[int, int]) -> List[int]:
    """The first milestone's episodes: shortest, median and longest (or the configured list)."""
    chosen = dataset_cfg.get("pilot_episodes", "auto")
    if chosen != "auto":
        return [int(i) for i in chosen]
    return extremes_and_median(sorted(lengths), lengths)


def annotated_outcomes(source) -> Tuple[Optional[Dict[int, str]], str]:
    """Outcome per episode from valid full-episode annotations, or why there are none."""
    outcomes: Dict[int, str] = {}
    unusable = []
    for episode in source.available():
        try:
            annotation = source.annotation(episode, require_valid=True)
        except (FileNotFoundError, ValueError):
            unusable.append(episode)
            continue
        outcomes[episode] = outcome_label(annotation.outcome)
    if unusable:
        return None, (f"{len(unusable)} of {len(source.available())} episodes have no valid full-episode "
                      "annotation, so outcomes were not used")
    return outcomes, f"valid full-episode annotations for all {len(outcomes)} episodes"


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> None:
    from .episode_dataset import RawEpisodeSource

    parser = argparse.ArgumentParser(description="Create or show the episode selection.")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--lengths-only", action="store_true",
                        help="ignore annotated outcomes even when every episode has one")
    add_config_arguments(create)
    show = sub.add_parser("show")
    add_config_arguments(show)
    args = parser.parse_args(argv)

    configs = load_configs(["dataset", "annotation", "graph"], args.overrides)
    dataset_cfg = configs["dataset"]
    source = RawEpisodeSource(configs, mode="full_episode")
    path = selection_path(dataset_cfg)

    if args.command == "show":
        data = load_selection(dataset_cfg, source.available())
        print(f"[selection] {data['version']}: {len(data['training'])} training episodes, all of them trained on")
        print(f"  diagnostic (also training): {data['diagnostic']}")
        print(f"  lengths: {data['diagnostic_lengths']}")
        if data["rule"]["outcomes_used"]:
            print(f"  outcomes: {data['rule']['diagnostic_outcomes']}")
        else:
            print(f"  outcomes not used: {data['rule']['outcome_note']}")
        return

    cfg = dataset_cfg["selection"]
    if args.lengths_only:
        outcomes, note = None, "--lengths-only was given"
    elif not bool(cfg["diagnostic"]["use_annotated_outcomes"]):
        outcomes, note = None, "dataset.selection.diagnostic.use_annotated_outcomes is false"
    else:
        outcomes, note = annotated_outcomes(source)
    data = make_selection(source.available(), source.lengths(), int(cfg["diagnostic"]["count"]), cfg["version"],
                          outcomes=outcomes, outcome_source="annotations/full_episode" if outcomes else None,
                          outcome_note=note, source_revision=source.source_record()["resolved_revision"])
    if os.path.isfile(path):
        existing = read_json(path)
        if existing.get("training") == data["training"] and existing.get("diagnostic") == data["diagnostic"]:
            print(f"[selection] {cfg['version']} unchanged ({path})")
            return
        raise SystemExit(
            f"{path} already holds selection {cfg['version']!r} with different episodes "
            f"({existing.get('diagnostic')} vs {data['diagnostic']}). Set dataset.selection.version to a new "
            "version; datasets, checkpoints and caches built on the old one keep referring to it."
        )
    write_json(path, data)
    print(f"[selection] {cfg['version']}: all {len(data['training'])} episodes train; "
          f"diagnostic subset {data['diagnostic']} -> {path}")
    print(f"  outcomes: {note}")


if __name__ == "__main__":
    main()
