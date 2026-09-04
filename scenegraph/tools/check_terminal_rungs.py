"""Can a successful episode actually pay every rung its final phase weights?

A terminal phase completes on an *exact* environment condition -- MS-HAB Pick
on ``norm(tcp - rest) <= pick_cfg.ee_rest_thresh``, PickCube on
``norm(cube - goal) <= goal_thresh`` -- while its spatial rungs are labelled
from *mined quantiles*. The two are calibrated independently, so a rung can end
up stricter than the condition it sits beside. When that happens a genuinely
successful trajectory leaves a weighted clause unpaid and the potential tops
out below 1.0, which is not a failure the runtime can report: every frame is
readable and every number is in range.

This decides that question from the actual asset and the actual tolerance
rather than from a label written by hand. A clause labelled ``very-near`` in a
test proves nothing about geometry; the bin edges do.

**Fails closed.** A missing terminal phase, an unresolved site, geometry it
cannot reason about, or a phase where nothing turned out to be checkable are
all reported as findings rather than silence. "Nothing to check" and "checked
and fine" must not share an exit status.

**Planar and height are independent.** They are mined from separate
statistics, so a wide excursion on one says nothing about the other, and this
never infers one from the other.

    python -m scenegraph.tools.check_terminal_rungs \\
        --asset  scenegraph/configs/subtask_whitelists/tidy_house/pick_all.json \\
        --schedule scenegraph/configs/schedules/tidy_house/pick.json \\
        --sites  scenegraph/configs/sites/tidy_house/pick.json \\
        --tolerance 0.05

Exit status is non-zero when anything is unpayable or unverifiable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from scenegraph.core.relation_rules import SPATIAL_LABELS
from scenegraph.core.sites import (
    METRIC_EUCLIDEAN,
    SITE_POINT,
    SITE_PREFIX,
)
from scenegraph.core.spatial_metrics import (
    EE_SITE_HEIGHT_KEY,
    EE_SITE_PLANAR_KEY,
    OBJECT_SITE_HEIGHT_KEY,
    OBJECT_SITE_PLANAR_KEY,
)

EE_KEY = "ee"
TERMINAL_RELATION = "reached"
SPATIAL_RELATIONS = ("planar-distance", "height-offset")

# Finding kinds. ``unpayable`` is a rung a successful episode cannot earn;
# ``unverifiable`` is anything this tool declines to reason about. Both stop
# the run, because both mean the question was not answered.
UNPAYABLE = "unpayable"
UNVERIFIABLE = "unverifiable"


class Finding:
    """One reason the terminal phase is not known to be payable."""

    __slots__ = ("kind", "phase", "relation", "labels", "weight", "value",
                 "label", "detail")

    def __init__(self, kind, phase="", relation="", labels=(), weight=0.0,
                 value=None, label="", detail=""):
        self.kind = kind
        self.phase = str(phase)
        self.relation = str(relation)
        self.labels = tuple(labels)
        self.weight = float(weight)
        self.value = value
        self.label = str(label)
        self.detail = str(detail)

    def __repr__(self):
        return f"Finding({self.kind!r}, {self.relation!r}, {self.detail!r})"

    def __str__(self):
        if self.kind == UNPAYABLE:
            where = f" at {self.value:+.4f} m" if self.value is not None else ""
            return (f"[unpayable] {self.phase}/{self.relation} "
                    f"weight={self.weight:g} wants {list(self.labels)} but"
                    f"{where} the label is {self.label!r}")
        return f"[unverifiable] {self.detail}"


def reachable_labels(
    lo: float, hi: float, edges: Sequence[float], names: Sequence[str],
) -> List[str]:
    """Every label a value in ``[lo, hi]`` can take.

    Not the endpoints. Testing only those assumes a clause's label set is
    "everything at least this good" -- a prefix of the planar ladder, or a
    band of the height one containing whatever the endpoints hit. A
    ``medium``-only planar rung, or a height rung accepting ``below`` and
    ``above`` but not ``level``, breaks that assumption and endpoint testing
    passes them while a successful episode cannot pay them.

    The bin index is a monotone step function of the value, so every label the
    interval can take lies in the contiguous index range between its ends --
    which is what makes enumerating them exact rather than a sample.
    """
    lo_idx = int(np.searchsorted(edges, lo, side="right"))
    hi_idx = int(np.searchsorted(edges, hi, side="right"))
    last = len(names) - 1
    return [names[min(i, last)] for i in range(min(lo_idx, hi_idx),
                                               min(hi_idx, last) + 1)]


def allowed_interval(relation: str, tolerance: float
                     ) -> Optional[Tuple[float, float]]:
    """The range a value takes inside the reached ball, or None.

    ``planar^2 + height^2 <= tolerance^2``, so planar runs over
    ``[0, tolerance]`` and signed height over ``[-tolerance, +tolerance]``.
    The two are treated independently on purpose: they are labelled from
    separately mined statistics, and one being wide says nothing about the
    other.
    """
    tolerance = abs(float(tolerance))
    if relation == "planar-distance":
        return 0.0, tolerance
    if relation == "height-offset":
        return -tolerance, tolerance
    return None


def clause_findings(
    bin_edges: Dict[str, Sequence[float]],
    clauses: Sequence[Dict[str, Any]],
    tolerance: float,
    keys: Dict[str, str],
    pair: Tuple[str, str],
    phase_name: str = "",
) -> Tuple[List[Finding], int, List[str]]:
    """``(findings, n_checked, skipped)`` for one phase's clauses.

    ``pair`` is the ``(src, dst)`` the terminal ``reached`` names. Only
    clauses over that same pair are bounded by the tolerance: a clause against
    any other pair is measured on a different scale over a different geometry,
    and checking it here would answer about a distance the condition does not
    constrain. A weighted one is reported as unverifiable rather than skipped,
    because the phase still has to pay it at success and nothing here can say
    whether it can. ``skipped`` carries only the clauses that are genuinely
    not this check's business: zero-weight gates.
    """
    findings: List[Finding] = []
    skipped: List[str] = []
    checked = 0
    for clause in clauses:
        relation = str(clause.get("relation") or "")
        weight = float(clause.get("weight", 0.0) or 0.0)
        labels = list(clause.get("labels") or ())
        if relation not in SPATIAL_RELATIONS:
            continue
        if weight <= 0.0:
            # A gate carries no reward, so a region that cannot satisfy it is
            # a phase that never opens -- a different problem, and one the
            # schedule compiler already refuses.
            skipped.append(f"{relation} {labels}: weight is zero (a gate)")
            continue
        clause_pair = (str(clause.get("src") or ""), str(clause.get("dst") or ""))
        if clause_pair != pair:
            # Not silence. The phase completes on an exact condition, so a
            # successful frame has to pay this clause too -- but its value is
            # measured on another scale over a geometry the tolerance does not
            # bound, so nothing here can say whether it is payable. That is an
            # unanswered question about the terminal phase, not a detail.
            findings.append(Finding(
                UNVERIFIABLE, phase_name, relation, labels, weight,
                detail=f"{phase_name}/{relation} {labels} weight={weight:g} is "
                       f"over {clause_pair}, not the pair {pair} that "
                       "'reached' names. The tolerance does not bound it, so "
                       "its payability at success is unverified here"))
            continue
        key = keys.get(relation)
        edges = list(bin_edges.get(key) or ()) if key else []
        names = list(SPATIAL_LABELS.get(relation) or ())
        if not edges or len(names) != len(edges) + 1:
            findings.append(Finding(
                UNVERIFIABLE, phase_name, relation, labels, weight,
                detail=f"{phase_name}/{relation} {labels} weight={weight:g}: "
                       f"the asset carries no usable {key!r} calibration "
                       f"({len(edges)} edge(s))"))
            continue
        interval = allowed_interval(relation, tolerance)
        if interval is None:
            findings.append(Finding(
                UNVERIFIABLE, phase_name, relation, labels, weight,
                detail=f"{phase_name}/{relation}: no geometric interval is "
                       "defined for this relation"))
            continue
        checked += 1
        for got in reachable_labels(interval[0], interval[1], edges, names):
            if got not in labels:
                findings.append(Finding(
                    UNPAYABLE, phase_name, relation, labels, weight,
                    value=_witness(got, edges, names, interval), label=got))
    return findings, checked, skipped


def _witness(label: str, edges, names, interval) -> Optional[float]:
    """A value inside the interval that takes ``label``, for the message."""
    lo, hi = interval
    index = list(names).index(label)
    candidates = [lo, hi] + [float(e) for e in edges] + \
                 [float(e) - 1e-9 for e in edges]
    for value in sorted(c for c in candidates if lo <= c <= hi):
        if int(np.searchsorted(edges, value, side="right")) == index:
            return value
    return None


def _terminal_phases(schedule: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Phases whose completion names ``reached``.

    That relation is the exact environment condition, so those are the phases
    whose rungs have to be reachable inside it. A phase completing on a grasp
    or a contact is under no such constraint -- it does not complete on a
    distance at all.
    """
    out = []
    for phase in schedule.get("phases") or ():
        completion = phase.get("completion") or {}
        clauses = (completion.get("all_of")
                   if "all_of" in completion else [completion])
        if any(str(c.get("relation")) == TERMINAL_RELATION for c in clauses):
            out.append(phase)
    return out


def _reached_clause(phase) -> Optional[Dict[str, Any]]:
    completion = phase.get("completion") or {}
    clauses = (completion.get("all_of")
               if "all_of" in completion else [completion])
    for clause in clauses:
        if str(clause.get("relation")) == TERMINAL_RELATION:
            return clause
    return None


def resolve_pair(phase, roles, sites) -> Tuple[Optional[Tuple[str, str]],
                                               Dict[str, str], List[Finding]]:
    """``(pair, calibration keys, findings)`` for one terminal phase.

    Geometry this tool cannot reason about is refused by name rather than
    guessed at. A surface site is entered through an oriented tube and a
    region is a disc, so neither describes the ball this check assumes; a
    planar metric is a cylinder, not a ball.
    """
    clause = _reached_clause(phase)
    name = str(phase.get("name") or "?")
    if clause is None:
        return None, {}, [Finding(
            UNVERIFIABLE, name,
            detail=f"{name}: completes on 'reached' but no clause names it")]
    pair = (str(clause.get("src") or ""), str(clause.get("dst") or ""))

    site_key = subject = None
    for token in pair:
        key = roles.get(token, token)
        if str(key).startswith(SITE_PREFIX):
            site_key, subject = key, (sites.get(key) or {}).get("subject")
    if site_key is None:
        return pair, {}, [Finding(
            UNVERIFIABLE, name,
            detail=f"{name}: 'reached' over {pair} names no declared site, so "
                   "the scale labelling it cannot be identified")]

    declaration = sites.get(site_key) or {}
    site_type = str(declaration.get("site_type") or "")
    metric = str(declaration.get("metric") or "")
    if site_type != SITE_POINT or metric != METRIC_EUCLIDEAN:
        return pair, {}, [Finding(
            UNVERIFIABLE, name,
            detail=f"{name}: site {site_key!r} is {site_type or '<none>'}/"
                   f"{metric or '<none>'}; this check assumes a "
                   f"{SITE_POINT}/{METRIC_EUCLIDEAN} ball. A surface is an "
                   "oriented entry tube and a region is a disc, and neither "
                   "is bounded the way this reasons")]

    if subject == EE_KEY:
        return pair, {"planar-distance": EE_SITE_PLANAR_KEY,
                      "height-offset": EE_SITE_HEIGHT_KEY}, []
    return pair, {"planar-distance": OBJECT_SITE_PLANAR_KEY,
                  "height-offset": OBJECT_SITE_HEIGHT_KEY}, []


def check(asset: Dict[str, Any], schedule: Dict[str, Any],
          sites: Dict[str, Any], tolerance: float
          ) -> Tuple[List[Finding], List[str]]:
    """``(findings, notes)``. A non-empty findings list means do not proceed."""
    notes: List[str] = []
    findings: List[Finding] = []
    phases = _terminal_phases(schedule)
    if not phases:
        return [Finding(
            UNVERIFIABLE,
            detail="no phase completes on 'reached', so the rule this checks "
                   "has nothing to apply to. If the schedule has a terminal "
                   "condition, it is expressed some other way and needs its "
                   "own check")], notes

    notes.append(f"terminal phase(s): {[p.get('name') for p in phases]}")
    notes.append(f"tolerance: {tolerance:g} m")
    roles = schedule.get("roles") or {}
    bin_edges = asset.get("bin_edges") or {}
    total_checked = 0

    for phase in phases:
        name = str(phase.get("name") or "?")
        pair, keys, problems = resolve_pair(phase, roles, sites)
        findings.extend(problems)
        if problems or pair is None:
            continue
        notes.append(f"{name}: pair {pair}, scale "
                     f"{keys['planar-distance']} / {keys['height-offset']}")
        phase_findings, checked, skipped = clause_findings(
            bin_edges, phase.get("clauses") or (), tolerance, keys, pair, name)
        findings.extend(phase_findings)
        total_checked += checked
        for line in skipped:
            notes.append(f"{name}: not checked -- {line}")
        notes.append(f"{name}: {checked} weighted spatial clause(s) checked")
        if checked == 0:
            # Per phase, not across the run: one phase carrying a verified
            # rung must not vouch for another that carried none.
            findings.append(Finding(
                UNVERIFIABLE, name,
                detail=f"{name}: no weighted spatial clause was checked, so "
                       "nothing about this phase was verified. An empty check "
                       "is not a pass"))

    notes.append(f"{total_checked} weighted spatial clause(s) checked in total")
    return findings, notes


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--asset", required=True,
                        help="Union whitelist carrying bin_edges.")
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--sites", default="",
                        help="Reviewed site declarations. Defaults to the "
                             "asset's own 'sites' block when omitted.")
    parser.add_argument(
        "--tolerance", type=float, required=True,
        help="The environment's own success distance, in metres. Read it "
             "from the running task -- pick_cfg.ee_rest_thresh for MS-HAB "
             "Pick -- never from a remembered value.")
    args = parser.parse_args(argv)

    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        print(f"[rungs] tolerance must be a positive distance, got "
              f"{args.tolerance}", file=sys.stderr)
        return 2
    try:
        with open(args.asset) as handle:
            asset = json.load(handle)
        with open(args.schedule) as handle:
            schedule = json.load(handle)
        if args.sites:
            with open(args.sites) as handle:
                sites = json.load(handle).get("sites") or {}
        else:
            sites = asset.get("sites") or {}
    except (OSError, ValueError) as exc:
        print(f"[rungs] could not read the inputs: {exc}", file=sys.stderr)
        return 2

    findings, notes = check(asset, schedule, sites, args.tolerance)
    print(f"\n=== {args.schedule}")
    for note in notes:
        print(f"  {note}")
    if not findings:
        print("  [ ok ] every weighted terminal rung is payable throughout "
              "the reached region")
        return 0
    unpayable = [f for f in findings if f.kind == UNPAYABLE]
    print(f"  [FAIL] {len(findings)} finding(s): {len(unpayable)} unpayable, "
          f"{len(findings) - len(unpayable)} unverifiable")
    for finding in findings:
        print(f"      {finding}")
    if unpayable:
        print("  A successful episode would leave those rungs unpaid and the "
              "potential would top out below 1.0.")
    print("  Reporting only: this tool proposes no change to the schedule, "
          "the bins or the environment threshold. Review required.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
