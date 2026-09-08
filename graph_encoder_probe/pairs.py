"""Fixed graph pairs that differ in exactly one thing.

Four controlled edits, each holding everything else fixed:

* ``absolute``   one edge's sigma becomes another label that relation may take
* ``temporal``   one non-padding delta becomes another change label
* ``geometry``   one node's centroid moves 1-5 cm along one world axis
* ``assignment`` two edges of one relation exchange their labels

``assignment`` is the one that cannot be answered by counting: both graphs carry
the identical label multiset over identical topology, and only the pairing
differs. The others each move one number.

Some edited graphs describe states physics would not produce -- a box that did
not follow its centroid, a distance label that disagrees with the geometry. That
is deliberate. The question is whether the encoder's pooled token responds to the
field, so every other field is pinned, including the ones that would co-vary in a
real frame.

Pure numpy: nothing here imports torch or the simulator, so pair construction and
its verification run anywhere the cache does.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from scenegraph.adapters.graph_pack import GRAPH_KEYS
from scenegraph.adapters.graph_vocab import (
    build_absolute_vocab,
    build_relation_vocab,
    build_temporal_vocab,
)
from scenegraph.core.relation_rules import RELATION_TYPES, abs_labels_for

from . import EDIT_GROUPS
from .dataset import FIELD_DTYPES, GraphFrames

PAIRS_ARRAYS = "pairs.npz"
PAIRS_SPECS = "pairs.json"

CONTROL_GROUP = "control"


class PairError(RuntimeError):
    """A pair is not what its spec says it is."""


def legal_absolute_mask() -> np.ndarray:
    """``[n_rel, n_abs]`` bool: which sigma each relation may carry.

    Derived from the shared tables, the same way ``graph._relation_masks``
    derives the decoder's; the tests assert the two agree. A hand-written mask
    here would be a second source of truth for what "a legal label" means, and
    an edit to an illegal label is measured by a head that masks it to -1e9.
    """
    absolute, relation = build_absolute_vocab(), build_relation_vocab()
    labels = abs_labels_for()
    mask = np.zeros((len(relation), len(absolute)), dtype=bool)
    for name in RELATION_TYPES:
        rid = relation.encode(name)
        for label in labels[name]:
            mask[rid, absolute.encode(label)] = True
    return mask


def label_names() -> dict[str, dict[int, str]]:
    """``id -> token`` per vocabulary, so a spec reads as words not numbers."""
    out: dict[str, dict[int, str]] = {}
    for name, vocab in (
        ("relation", build_relation_vocab()),
        ("absolute", build_absolute_vocab()),
        ("temporal", build_temporal_vocab()),
    ):
        out[name] = {0: "<pad>"} | {i: tok for tok, i in vocab.token_to_id.items()}
    return out


@dataclass
class EditContext:
    """Everything an edit needs that is not the frame itself."""

    abs_valid: np.ndarray
    n_temp: int
    names: dict[str, dict[int, str]]
    geometry_delta_m: tuple[float, float]

    @classmethod
    def build(cls, geometry_delta_m: Sequence[float] = (0.01, 0.05)) -> "EditContext":
        low, high = (float(v) for v in geometry_delta_m)
        if not 0 < low <= high:
            raise ValueError(f"geometry_delta_m must be 0 < low <= high, got {geometry_delta_m}")
        return cls(
            abs_valid=legal_absolute_mask(),
            n_temp=len(build_temporal_vocab()),
            names=label_names(),
            geometry_delta_m=(low, high),
        )


@dataclass(frozen=True)
class PairSpec:
    """One pair: which frame, which field, which cells, and what they became."""

    name: str
    group: str
    source: int
    edited_field: Optional[str]
    positions: tuple[tuple[int, ...], ...]
    before: tuple[float, ...]
    after: tuple[float, ...]
    note: str
    detail: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["positions"] = [list(pos) for pos in self.positions]
        payload["before"] = list(self.before)
        payload["after"] = list(self.after)
        return payload

    @classmethod
    def from_json(cls, payload: Mapping) -> "PairSpec":
        return cls(
            name=str(payload["name"]),
            group=str(payload["group"]),
            source=int(payload["source"]),
            edited_field=payload["edited_field"],
            positions=tuple(tuple(int(v) for v in pos) for pos in payload["positions"]),
            before=tuple(float(v) for v in payload["before"]),
            after=tuple(float(v) for v in payload["after"]),
            note=str(payload["note"]),
            detail=dict(payload.get("detail") or {}),
        )


@dataclass
class PairSet:
    """The probe set: specs plus the edited member of every pair.

    The A member is never stored -- it is ``dataset.frames[spec.source]``, and
    keeping one copy is what makes "identical apart from the edit" checkable
    rather than asserted.
    """

    specs: list[PairSpec]
    edited: GraphFrames
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.specs)

    @property
    def sources(self) -> np.ndarray:
        return np.asarray([spec.source for spec in self.specs], dtype=np.int64)

    def groups(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for i, spec in enumerate(self.specs):
            out.setdefault(spec.group, []).append(i)
        return out

    def verify_against(self, frames: GraphFrames) -> None:
        """Re-check every pair against the frames it was built from."""
        for i, spec in enumerate(self.specs):
            verify_pair(frames.frame(spec.source), self.edited.frame(i), spec)

    def save(self, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        payload = dict(self.edited.fields)
        payload["source"] = self.sources
        np.savez_compressed(os.path.join(out_dir, PAIRS_ARRAYS), **payload)
        specs_path = os.path.join(out_dir, PAIRS_SPECS)
        with open(specs_path, "w") as handle:
            json.dump(
                {"meta": self.meta, "specs": [spec.to_json() for spec in self.specs]},
                handle,
                indent=2,
                sort_keys=True,
            )
        return specs_path

    @classmethod
    def load(cls, out_dir: str) -> "PairSet":
        specs_path = os.path.join(out_dir, PAIRS_SPECS)
        if not os.path.isfile(specs_path):
            raise FileNotFoundError(f"no probe set at {out_dir!r}; run the pairs stage")
        with open(specs_path) as handle:
            payload = json.load(handle)
        with np.load(os.path.join(out_dir, PAIRS_ARRAYS)) as data:
            fields = {key: data[key] for key in GRAPH_KEYS}
        specs = [PairSpec.from_json(item) for item in payload["specs"]]
        edited = GraphFrames(fields)
        if len(edited) != len(specs):
            raise ValueError(
                f"probe set holds {len(edited)} edited frames for {len(specs)} specs"
            )
        return cls(specs, edited, dict(payload.get("meta") or {}))


# --------------------------------------------------------------------------- #
# The four edits
# --------------------------------------------------------------------------- #
def _valid_edges(frame: Mapping[str, np.ndarray]) -> np.ndarray:
    """Rows the encoder actually consumes: relation id non-zero."""
    return np.flatnonzero(np.asarray(frame["graph_edge_rel"]) != 0)


def _valid_nodes(frame: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.flatnonzero(np.asarray(frame["graph_node_ent"]) != 0)


def _edit_absolute(frame, rng, ctx) -> Optional[dict]:
    rows = _valid_edges(frame)
    rng.shuffle(rows)
    for row in rows:
        rel = int(frame["graph_edge_rel"][row])
        current = int(frame["graph_edge_abs"][row])
        # Padding is excluded by construction: ``abs_valid`` never marks column
        # zero, because no relation lists the pad token among its labels.
        choices = np.flatnonzero(ctx.abs_valid[rel])
        choices = choices[choices != current]
        if choices.size == 0:
            continue
        new = int(rng.choice(choices))
        return {
            "field": "graph_edge_abs",
            "positions": ((int(row),),),
            "before": (float(current),),
            "after": (float(new),),
            "note": "one absolute label changed",
            "detail": {
                "edge_row": int(row),
                "relation": ctx.names["relation"][rel],
                "from": ctx.names["absolute"][current],
                "to": ctx.names["absolute"][new],
                "src_row": int(frame["graph_edge_src"][row]),
                "dst_row": int(frame["graph_edge_dst"][row]),
            },
        }
    return None


def _edit_temporal(frame, rng, ctx) -> Optional[dict]:
    rows = _valid_edges(frame)
    # Only a relation that carries a change label has a non-padding delta, so
    # filtering on the stored value is also filtering on ``temp_valid``.
    rows = rows[np.asarray(frame["graph_edge_temp"])[rows] != 0]
    rng.shuffle(rows)
    for row in rows:
        current = int(frame["graph_edge_temp"][row])
        choices = np.arange(1, ctx.n_temp)
        choices = choices[choices != current]
        if choices.size == 0:
            continue
        new = int(rng.choice(choices))
        rel = int(frame["graph_edge_rel"][row])
        return {
            "field": "graph_edge_temp",
            "positions": ((int(row),),),
            "before": (float(current),),
            "after": (float(new),),
            "note": "one temporal label changed",
            "detail": {
                "edge_row": int(row),
                "relation": ctx.names["relation"][rel],
                "from": ctx.names["temporal"][current],
                "to": ctx.names["temporal"][new],
            },
        }
    return None


def _edit_geometry(frame, rng, ctx) -> Optional[dict]:
    rows = _valid_nodes(frame)
    rng.shuffle(rows)
    low, high = ctx.geometry_delta_m
    for row in rows:
        axis = int(rng.integers(3))
        delta = float(rng.uniform(low, high)) * float(rng.choice([-1.0, 1.0]))
        current = np.float32(frame["graph_node_centroid"][row, axis])
        new = np.float32(current + np.float32(delta))
        if new == current:
            # Only reachable if the centroid is far enough from the origin that
            # a centimetre falls under a float32 ulp, which a table-scale scene
            # never is. Skipping beats writing a pair whose edit did nothing.
            continue
        return {
            "field": "graph_node_centroid",
            "positions": ((int(row), axis),),
            "before": (float(current),),
            "after": (float(new),),
            "note": "one centroid moved",
            "detail": {
                "node_row": int(row),
                "axis": "xyz"[axis],
                "delta_m": round(float(new) - float(current), 6),
                "entity_id": int(frame["graph_node_ent"][row]),
            },
        }
    return None


def _edit_assignment(frame, rng, ctx) -> Optional[dict]:
    """Exchange two labels between edges of one relation.

    Both endpoints pairs must differ, or the swap would move a label between two
    facts about the same two nodes and the group would stop being about pairing.
    """
    rows = _valid_edges(frame)
    rel = np.asarray(frame["graph_edge_rel"])
    by_relation: dict[int, list[int]] = {}
    for row in rows:
        by_relation.setdefault(int(rel[row]), []).append(int(row))
    relations = [r for r, group in by_relation.items() if len(group) >= 2]
    rng.shuffle(relations)
    for relation in relations:
        group = list(by_relation[relation])
        rng.shuffle(group)
        for i, first in enumerate(group):
            for second in group[i + 1:]:
                a_abs = int(frame["graph_edge_abs"][first])
                b_abs = int(frame["graph_edge_abs"][second])
                if a_abs == b_abs:
                    continue
                a_pair = (int(frame["graph_edge_src"][first]), int(frame["graph_edge_dst"][first]))
                b_pair = (int(frame["graph_edge_src"][second]), int(frame["graph_edge_dst"][second]))
                if a_pair == b_pair:
                    continue
                return {
                    "field": "graph_edge_abs",
                    "positions": ((first,), (second,)),
                    "before": (float(a_abs), float(b_abs)),
                    "after": (float(b_abs), float(a_abs)),
                    "note": "labels exchanged between pairs",
                    "detail": {
                        "edge_rows": [first, second],
                        "relation": ctx.names["relation"][relation],
                        "labels": [
                            ctx.names["absolute"][a_abs],
                            ctx.names["absolute"][b_abs],
                        ],
                        "node_pairs": [list(a_pair), list(b_pair)],
                    },
                }
    return None


EDITS: dict[str, Callable] = {
    "absolute": _edit_absolute,
    "temporal": _edit_temporal,
    "geometry": _edit_geometry,
    "assignment": _edit_assignment,
}


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify_pair(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray], spec: PairSpec) -> None:
    """Confirm the packed tensors differ in exactly the intended way.

    Three things at once, all on the arrays that reach the encoder rather than
    on the code that produced them: the edited row is one the encoder consumes,
    the written value is legal and not padding, and no other cell moved.
    """
    for key in GRAPH_KEYS:
        left, right = np.asarray(a[key]), np.asarray(b[key])
        if left.shape != right.shape:
            raise PairError(f"{spec.name}: {key} shape {left.shape} vs {right.shape}")
        if left.dtype != FIELD_DTYPES[key] or right.dtype != FIELD_DTYPES[key]:
            raise PairError(
                f"{spec.name}: {key} dtypes {left.dtype}/{right.dtype}, "
                f"the packer emits {FIELD_DTYPES[key]}"
            )
        if key == spec.edited_field:
            continue
        if not np.array_equal(left, right):
            raise PairError(f"{spec.name}: {key} changed but only {spec.edited_field} should")

    if spec.edited_field is None:
        if spec.positions or spec.before or spec.after:
            raise PairError(f"{spec.name}: a control names an edit")
        return

    left = np.asarray(a[spec.edited_field])
    right = np.asarray(b[spec.edited_field])
    moved = {tuple(int(v) for v in pos) for pos in zip(*np.nonzero(left != right))}
    want = {tuple(int(v) for v in pos) for pos in spec.positions}
    if moved != want:
        raise PairError(
            f"{spec.name}: {spec.edited_field} differs at {sorted(moved)}, "
            f"the spec claims {sorted(want)}"
        )
    for pos, before, after in zip(spec.positions, spec.before, spec.after):
        idx = tuple(int(v) for v in pos)
        if float(left[idx]) != float(before) or float(right[idx]) != float(after):
            raise PairError(
                f"{spec.name}: {spec.edited_field}{list(idx)} is "
                f"{float(left[idx])}->{float(right[idx])}, the spec claims "
                f"{before}->{after}"
            )
    _check_consumed(a, b, spec)


def _check_consumed(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray], spec: PairSpec) -> None:
    """The edited row is a real node or a real fact, in both members.

    An edit to a padded row is invisible to the encoder -- it strips those rows
    before the first message pass -- so a pair built on one would sit down at the
    controls' distance while claiming to be an edit.
    """
    ctx_mask = legal_absolute_mask()
    for pos in spec.positions:
        row = int(pos[0])
        if spec.edited_field == "graph_node_centroid":
            for side, frame in (("A", a), ("B", b)):
                if int(frame["graph_node_ent"][row]) == 0:
                    raise PairError(f"{spec.name}: node row {row} is padding in {side}")
            continue
        for side, frame in (("A", a), ("B", b)):
            if int(frame["graph_edge_rel"][row]) == 0:
                raise PairError(f"{spec.name}: edge row {row} is padding in {side}")
        rel = int(a["graph_edge_rel"][row])
        if spec.edited_field == "graph_edge_abs":
            new = int(b["graph_edge_abs"][row])
            if new == 0:
                raise PairError(f"{spec.name}: wrote the pad absolute label at row {row}")
            if not bool(ctx_mask[rel, new]):
                raise PairError(
                    f"{spec.name}: absolute id {new} is not legal for relation id {rel}"
                )
        elif spec.edited_field == "graph_edge_temp":
            if int(b["graph_edge_temp"][row]) == 0:
                raise PairError(f"{spec.name}: wrote the pad temporal label at row {row}")
            if int(a["graph_edge_temp"][row]) == 0:
                raise PairError(f"{spec.name}: edge row {row} carried no temporal label")


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def build_pairs(
    frames: GraphFrames,
    *,
    per_group: int = 25,
    controls: int = 8,
    seed: int = 0,
    geometry_delta_m: Sequence[float] = (0.01, 0.05),
    groups: Sequence[str] = EDIT_GROUPS,
) -> PairSet:
    """Draw the fixed probe set. Built once, then never redrawn."""
    if len(frames) == 0:
        raise ValueError("cannot build pairs from an empty dataset")
    ctx = EditContext.build(geometry_delta_m)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(frames))

    specs: list[PairSpec] = []
    edited_frames: list[dict[str, np.ndarray]] = []

    for group in groups:
        if group not in EDITS:
            raise KeyError(f"unknown edit group {group!r}; have {sorted(EDITS)}")
        made, tried = 0, 0
        for source in order:
            if made >= per_group:
                break
            tried += 1
            frame = frames.frame(int(source))
            edit = EDITS[group](frame, rng, ctx)
            if edit is None:
                continue
            other = {key: np.array(arr, copy=True) for key, arr in frame.items()}
            target = other[edit["field"]]
            for pos, value in zip(edit["positions"], edit["after"]):
                target[tuple(int(v) for v in pos)] = value
            spec = PairSpec(
                name=f"{group}-{made + 1:02d}",
                group=group,
                source=int(source),
                edited_field=edit["field"],
                positions=tuple(tuple(int(v) for v in pos) for pos in edit["positions"]),
                before=tuple(float(v) for v in edit["before"]),
                after=tuple(float(v) for v in edit["after"]),
                note=edit["note"],
                detail=edit["detail"],
            )
            verify_pair(frame, other, spec)
            specs.append(spec)
            edited_frames.append(other)
            made += 1
        if made < per_group:
            raise RuntimeError(
                f"only {made} of {per_group} {group!r} pairs could be built from "
                f"{tried} frames. The dataset does not carry enough variety for "
                "this edit -- extend collection for the relation it needs "
                "(see the collect summary's per-relation label counts)."
            )

    # Controls are the same frame twice. They are the numerical floor: whatever
    # distance they show is what "no change at all" costs on this hardware, in
    # this precision, through this code path.
    control_sources = order[:controls] if controls <= order.size else order
    for i, source in enumerate(control_sources):
        frame = frames.frame(int(source))
        spec = PairSpec(
            name=f"{CONTROL_GROUP}-{i + 1:02d}",
            group=CONTROL_GROUP,
            source=int(source),
            edited_field=None,
            positions=(),
            before=(),
            after=(),
            note="none",
            detail={},
        )
        verify_pair(frame, frame, spec)
        specs.append(spec)
        edited_frames.append(frame)

    edited = GraphFrames(
        {
            key: np.stack([frame[key] for frame in edited_frames]).astype(
                FIELD_DTYPES[key], copy=False
            )
            for key in GRAPH_KEYS
        }
    )
    meta = {
        "per_group": int(per_group),
        "controls": int(len(control_sources)),
        "groups": list(groups),
        "seed": int(seed),
        "geometry_delta_m": [float(v) for v in geometry_delta_m],
        "dataset_frames": int(len(frames)),
        "dataset_fingerprint": frames.fingerprint(),
    }
    return PairSet(specs, edited, meta)
