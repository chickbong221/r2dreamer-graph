"""The frozen scene split Experiment B trains and generalises across.

Naming the scenes in the env YAML would work for one of them and stops
working at forty-seven: the list has to be read by the launcher's validation,
by the panel that scores each half separately, and by the tests, and three
copies of forty-seven strings is three chances for them to disagree. So there
is one file -- ``configs/scenes/mshab_pick_b.json`` -- and everything reads
it.

What the manifest is *not* is a count. ``num_build_configs`` takes a sorted
prefix, and which apartment "the first five" means moves the moment the
installed build list does. Every scene here is named outright, and
``select_named_build_configs`` raises rather than falling back if one is not
in the installed task plan, so a manifest that has drifted from the dataset
fails at construction instead of quietly training somewhere else.

The split rule itself lives in ``split_scenes`` and is applied by
``scenegraph/tools/freeze_scene_split.py``: training takes the pinned scene
and its macro-group neighbours, held-out takes every scene outside that
group. Different arrangements of one apartment against every arrangement of
the others -- which is the generalisation the experiment claims to measure.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

# ``v3_sc0_staging_00.scene_instance.json`` -> group ``v3_sc0_staging``,
# index 0. A name that does not parse has no group, which collapses the rule
# below to plain sorted order rather than guessing.
_SCENE_NAME = re.compile(r"^(?P<group>.+?)_(?P<index>\d+)\.scene_instance\.json$")


def scene_group(name: str) -> str:
    """The apartment a build configuration is an arrangement of, or ``''``."""
    match = _SCENE_NAME.match(str(name))
    return match.group("group") if match else ""


@dataclass(frozen=True)
class SceneSplit:
    """Which scenes are trained in, which are held out, which carries light."""

    train: List[str]
    held_out: List[str]
    lighting_scene: str
    source: str = ""

    @property
    def evaluation(self) -> List[str]:
        """Every scene the evaluation simulator has to build.

        Both halves, because the panel scores them separately: the held-out
        scenes are the generalisation number and the training scenes are what
        the checkpoint is selected on.
        """
        return list(self.train) + list(self.held_out)

    def validate(self) -> "SceneSplit":
        for label, names in (("train", self.train), ("held_out", self.held_out)):
            if not names:
                raise ValueError(f"scene manifest has no {label} scenes")
            if len(set(names)) != len(names):
                raise ValueError(f"scene manifest repeats a {label} scene")
            for name in names:
                if not str(name).endswith(".scene_instance.json"):
                    raise ValueError(f"not a build configuration name: {name!r}")
        overlap = sorted(set(self.train) & set(self.held_out))
        if overlap:
            raise ValueError(
                f"scene manifest trains and holds out the same scene(s): "
                f"{overlap}. An unseen-scene score measured on a trained "
                "scene is not a generalisation number."
            )
        if self.lighting_scene not in self.train:
            raise ValueError(
                f"lighting scene {self.lighting_scene!r} is not one of the "
                "training scenes; the lighting comparison is a controlled "
                "change to a scene the policy trained in, not a second "
                "generalisation test."
            )
        return self


def _spread(groups: Dict[str, List[str]], count: int) -> List[str]:
    """Take ``count`` names one apartment at a time.

    A plain prefix would fill the held-out set from whichever apartment sorts
    first and only reach the next one once that ran out -- 21 arrangements of
    one room and 9 of another, which is not the even coverage a
    scene-generalisation number wants. Round-robin instead, then sort, so the
    manifest reads in order and the panel stays deterministic.

    When ``count`` is every scene available this is exactly the sorted list,
    so a full split is unaffected by the rule.
    """
    ordered = [list(groups[key]) for key in sorted(groups)]
    out: List[str] = []
    for index in range(max((len(group) for group in ordered), default=0)):
        for group in ordered:
            if index < len(group):
                out.append(group[index])
                if len(out) == count:
                    return sorted(out)
    return sorted(out)


def split_scenes(available: Sequence[str], pinned: str, n_train: int,
                 n_held_out: int) -> SceneSplit:
    """Training scenes from the pinned scene's apartment, held-out from the rest.

    Deliberately not a sorted prefix over everything: that would put the
    held-out set in whichever apartments happen to sort last and leave the
    count to chance. Grouping by apartment makes the two halves mean something
    -- arrangements of one apartment against arrangements of the others -- and
    a held-out set drawn evenly from every remaining apartment rather than
    filling one before starting the next.
    """
    names = sorted({str(n) for n in available})
    if pinned not in names:
        raise ValueError(f"pinned scene {pinned!r} is not in the {len(names)} "
                         f"available build configurations")
    group = scene_group(pinned)
    inside = [n for n in names if scene_group(n) == group] if group else names
    outside = [n for n in names if n not in inside]
    if len(inside) < n_train:
        raise ValueError(
            f"apartment {group!r} has {len(inside)} arrangement(s), need "
            f"{n_train} training scenes")
    train = [pinned] + [n for n in inside if n != pinned]
    train = sorted(train[:n_train])
    by_group: Dict[str, List[str]] = {}
    for name in outside:
        by_group.setdefault(scene_group(name), []).append(name)
    held_out = _spread(by_group, n_held_out)
    if len(held_out) != n_held_out:
        raise ValueError(
            f"{len(outside)} scene(s) outside apartment {group!r}, need "
            f"{n_held_out} held-out scenes")
    return SceneSplit(train=train, held_out=held_out,
                      lighting_scene=pinned).validate()


def load_manifest(path: str) -> SceneSplit:
    """Read a frozen split, or say exactly what is wrong with it."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"scene manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    missing = [key for key in ("train", "held_out", "lighting_scene")
               if key not in raw]
    if missing:
        raise ValueError(f"scene manifest {path} is missing {missing}")
    return SceneSplit(
        train=[str(n) for n in raw["train"]],
        held_out=[str(n) for n in raw["held_out"]],
        lighting_scene=str(raw["lighting_scene"]),
        source=path,
    ).validate()


def apply_scene_manifest(env_config) -> "SceneSplit | None":
    """Write a frozen split into the env config, before anything is built.

    Resolved here rather than inside the environment adapter so that every
    later reader -- the training/evaluation selectors, ``training_scenes``,
    the checkpoint identity, the transfer stage's copy of the config -- sees
    one explicit list of names and no code has to know a manifest exists.

    Refuses to overwrite. A config that names both a manifest and its own
    scenes has two sources for one decision, and quietly preferring either is
    how the two drift apart.
    """
    raw = str(getattr(env_config, "scene_manifest", "") or "")
    if not raw:
        return None
    path = raw
    if not os.path.isabs(path):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
    split = load_manifest(path)
    for field_name, wanted in (("train_build_config_ids", split.train),
                               ("eval_build_config_ids", split.evaluation)):
        existing = [str(n) for n in (getattr(env_config, field_name, None) or [])]
        if existing and existing != list(wanted):
            raise ValueError(
                f"env.{field_name} is set to {existing} and scene_manifest "
                f"{raw} says {list(wanted)}. Name the scenes in one place.")
        setattr(env_config, field_name, list(wanted))
    lighting = getattr(env_config, "eval_lighting", None)
    if lighting is not None and getattr(lighting, "enabled", False):
        named = str(getattr(lighting, "scene", "") or "")
        if named and named != split.lighting_scene:
            raise ValueError(
                f"env.eval_lighting.scene is {named!r} and the manifest pins "
                f"{split.lighting_scene!r}")
        lighting.scene = split.lighting_scene
    print(f"[scenes] {raw}: {counts(split)}; lighting on "
          f"{split.lighting_scene}", flush=True)
    return split


def counts(split: SceneSplit) -> Dict[str, int]:
    return {"train": len(split.train), "held_out": len(split.held_out),
            "evaluation": len(split.evaluation)}
