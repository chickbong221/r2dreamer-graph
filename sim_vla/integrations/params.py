"""Counting the parameters that were actually built, not the ones intended.

The budget is a property of instantiated modules. A width in a config file is
not a parameter count: TD-MPC2's convolutional encoder is sized by the image
resolution and the camera count as much as by ``latent_dim``, and SOLD's
predictors are sized by ``num_slots`` through the attention mask. So this
module measures objects, and the sizing decisions are taken afterwards.

Double counting
---------------

Components overlap. TD-MPC2's ``_Qs`` and ``_target_Qs`` are distinct tensors
and both count; its ``_encoder`` is a sub-module of the same ``WorldModel``
whose ``parameters()`` also yields the dynamics, so summing per-component
totals over-reports. SmolVLA's adapter is an attribute of the actor.

Every parameter is therefore attributed to the **first** component that claims
it, in the order the components are declared. Each component reports:

``total`` / ``trainable``   everything it reaches, overlap included
``unique`` / ``unique_trainable``  what was attributed to it
``shared_with``             the earlier component that already claimed the rest

and the report's ``total`` is the sum of the ``unique`` figures, which is the
number of distinct tensors in memory.

A parameter is identified by ``data_ptr`` as well as ``id``: a module that was
deep-copied has different Python objects over different storage and must count
twice (target networks), while a tied weight shares storage and must not.
``id`` alone is the right test for tying and ``data_ptr`` alone misfires on
empty tensors, so both are used and either match means "already seen".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

MILLION = 1_000_000


def _parameters(component: Any) -> List[Any]:
    """Parameters of a module, an iterable of parameters, or nothing."""
    if component is None:
        return []
    if hasattr(component, "parameters"):
        return list(component.parameters())
    return list(component)


def _key(parameter) -> Tuple[int, int, int]:
    """Identity for de-duplication: object, storage, offset."""
    try:
        pointer = int(parameter.data_ptr())
    except Exception:                                      # noqa: BLE001
        pointer = 0
    return (id(parameter), pointer, int(parameter.numel()))


@dataclass
class ComponentReport:
    name: str
    total: int = 0
    trainable: int = 0
    unique: int = 0
    unique_trainable: int = 0
    tensors: int = 0
    shared_with: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "total": self.total,
                "trainable": self.trainable, "unique": self.unique,
                "unique_trainable": self.unique_trainable,
                "tensors": self.tensors,
                "shared_with": sorted(set(self.shared_with))}


def report(components: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    """Per-component counts plus a de-duplicated total.

    ``components`` is an ordered sequence of ``(name, module_or_parameters)``.
    Order decides attribution, so the declaration puts the owner first: the
    encoder before the world model that contains it.
    """
    seen: Dict[Tuple[int, int, int], str] = {}
    reports: List[ComponentReport] = []
    for name, component in components:
        entry = ComponentReport(name=str(name))
        for parameter in _parameters(component):
            numel = int(parameter.numel())
            grad = bool(getattr(parameter, "requires_grad", False))
            entry.total += numel
            entry.trainable += numel if grad else 0
            entry.tensors += 1
            key = _key(parameter)
            owner = seen.get(key)
            if owner is None:
                # data_ptr is not unique across devices or for views, so the
                # id-only fallback is checked too.
                owner = seen.get((key[0], 0, key[2]))
            if owner is not None:
                if owner != entry.name:
                    entry.shared_with.append(owner)
                continue
            seen[key] = entry.name
            entry.unique += numel
            entry.unique_trainable += numel if grad else 0
        reports.append(entry)

    total = sum(entry.unique for entry in reports)
    trainable = sum(entry.unique_trainable for entry in reports)
    return {
        "components": [entry.as_dict() for entry in reports],
        "total": total,
        "trainable": trainable,
        "total_millions": round(total / MILLION, 3),
        "trainable_millions": round(trainable / MILLION, 3),
    }


def split(full: Mapping[str, Any], *, exclude: Sequence[str]) -> Dict[str, Any]:
    """The same report with some components left out of the headline total.

    The budget is about the world-model side; SmolVLA's own weights are
    accepted and counted separately rather than hidden. ``exclude`` names the
    components that are reported but not added to ``budget_total``.
    """
    excluded = set(exclude)
    inside = [c for c in full["components"] if c["name"] not in excluded]
    outside = [c for c in full["components"] if c["name"] in excluded]
    budget = sum(c["unique"] for c in inside)
    budget_trainable = sum(c["unique_trainable"] for c in inside)
    other = sum(c["unique"] for c in outside)
    return {**full,
            "excluded": sorted(excluded),
            "budget_total": budget,
            "budget_trainable": budget_trainable,
            "budget_total_millions": round(budget / MILLION, 3),
            "budget_trainable_millions": round(budget_trainable / MILLION, 3),
            "excluded_total": other,
            "excluded_total_millions": round(other / MILLION, 3)}


def render(full: Mapping[str, Any], *, title: str = "parameters") -> str:
    """A table, because a dict of eleven components is not readable."""
    lines = [f"=== {title}",
             f"{'component':<28}{'total':>14}{'trainable':>14}"
             f"{'unique':>14}  shared with"]
    for entry in full["components"]:
        shared = ",".join(entry["shared_with"]) or "-"
        lines.append(
            f"{entry['name']:<28}{entry['total']:>14,}"
            f"{entry['trainable']:>14,}{entry['unique']:>14,}  {shared}")
    lines.append("-" * 72)
    lines.append(f"{'distinct total':<28}{full['total']:>14,}"
                 f"{full['trainable']:>14,}"
                 f"  ({full['total_millions']}M / "
                 f"{full['trainable_millions']}M trainable)")
    if "budget_total" in full:
        lines.append(
            f"{'world-model side':<28}{full['budget_total']:>14,}"
            f"{full['budget_trainable']:>14,}"
            f"  ({full['budget_total_millions']}M), excluding "
            f"{full['excluded']}")
        lines.append(
            f"{'excluded':<28}{full['excluded_total']:>14,}"
            f"{'':>14}  ({full['excluded_total_millions']}M)")
    return "\n".join(lines)
