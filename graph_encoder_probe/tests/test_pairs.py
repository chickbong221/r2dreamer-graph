"""The probe set is only worth measuring if each pair differs in one way.

Every claim in ``PairSpec`` is checked against the packed arrays themselves, and
the verifier is checked against edits it must reject -- a verifier that passes
everything would let a silently broken edit be reported as "no difference".
"""

from __future__ import annotations

import tempfile
import unittest

import numpy as np

from scenegraph.adapters.graph_pack import GRAPH_KEYS

from ..pairs import (
    CONTROL_GROUP,
    PairError,
    PairSet,
    PairSpec,
    build_pairs,
    legal_absolute_mask,
    verify_pair,
)
from .synthetic import make_frames

try:                                                       # torch is the model half
    import torch
except ImportError:                                        # pragma: no cover
    torch = None


class PairConstruction(unittest.TestCase):
    def setUp(self):
        self.frames = make_frames(48, seed=11)
        self.pairs = build_pairs(self.frames, per_group=6, controls=4, seed=5)

    def test_every_group_is_filled(self):
        counts = {name: len(rows) for name, rows in self.pairs.groups().items()}
        self.assertEqual(
            counts,
            {"absolute": 6, "temporal": 6, "geometry": 6, "assignment": 6, CONTROL_GROUP: 4},
        )

    def test_only_the_named_field_moves(self):
        for i, spec in enumerate(self.pairs.specs):
            a = self.frames.frame(spec.source)
            b = self.pairs.edited.frame(i)
            for key in GRAPH_KEYS:
                if key == spec.edited_field:
                    continue
                self.assertTrue(
                    np.array_equal(a[key], b[key]), f"{spec.name} changed {key}"
                )

    def test_edited_rows_are_rows_the_encoder_reads(self):
        """An edit to a padded row is stripped before the first message pass."""
        for i, spec in enumerate(self.pairs.specs):
            a = self.frames.frame(spec.source)
            for pos in spec.positions:
                row = int(pos[0])
                if spec.edited_field == "graph_node_centroid":
                    self.assertNotEqual(int(a["graph_node_ent"][row]), 0, spec.name)
                else:
                    self.assertNotEqual(int(a["graph_edge_rel"][row]), 0, spec.name)
            del i

    def test_absolute_replacements_are_legal_and_not_padding(self):
        mask = legal_absolute_mask()
        for i, spec in enumerate(self.pairs.specs):
            if spec.edited_field != "graph_edge_abs":
                continue
            b = self.pairs.edited.frame(i)
            for pos in spec.positions:
                row = int(pos[0])
                label = int(b["graph_edge_abs"][row])
                relation = int(b["graph_edge_rel"][row])
                self.assertNotEqual(label, 0, f"{spec.name} wrote padding")
                self.assertTrue(mask[relation, label], f"{spec.name} wrote an illegal label")

    def test_temporal_replacements_are_not_padding(self):
        for i, spec in enumerate(self.pairs.specs):
            if spec.edited_field != "graph_edge_temp":
                continue
            a, b = self.frames.frame(spec.source), self.pairs.edited.frame(i)
            row = int(spec.positions[0][0])
            self.assertNotEqual(int(a["graph_edge_temp"][row]), 0)
            self.assertNotEqual(int(b["graph_edge_temp"][row]), 0)

    def test_assignment_keeps_the_label_histogram(self):
        """Same labels, different pairing -- that is the whole point of the group."""
        for i, spec in enumerate(self.pairs.specs):
            if spec.group != "assignment":
                continue
            a, b = self.frames.frame(spec.source), self.pairs.edited.frame(i)
            self.assertEqual(
                sorted(a["graph_edge_abs"].tolist()), sorted(b["graph_edge_abs"].tolist())
            )
            self.assertFalse(np.array_equal(a["graph_edge_abs"], b["graph_edge_abs"]))
            first, second = (int(pos[0]) for pos in spec.positions)
            self.assertEqual(int(a["graph_edge_rel"][first]), int(a["graph_edge_rel"][second]))
            self.assertNotEqual(
                (int(a["graph_edge_src"][first]), int(a["graph_edge_dst"][first])),
                (int(a["graph_edge_src"][second]), int(a["graph_edge_dst"][second])),
            )

    def test_geometry_moves_one_axis_by_the_configured_amount(self):
        for i, spec in enumerate(self.pairs.specs):
            if spec.group != "geometry":
                continue
            a, b = self.frames.frame(spec.source), self.pairs.edited.frame(i)
            delta = np.abs(b["graph_node_centroid"] - a["graph_node_centroid"])
            self.assertEqual(int((delta > 0).sum()), 1)
            self.assertGreaterEqual(float(delta.max()), 0.01 - 1e-6)
            self.assertLessEqual(float(delta.max()), 0.05 + 1e-6)

    def test_controls_are_identical(self):
        for i, spec in enumerate(self.pairs.specs):
            if spec.group != CONTROL_GROUP:
                continue
            a, b = self.frames.frame(spec.source), self.pairs.edited.frame(i)
            for key in GRAPH_KEYS:
                self.assertTrue(np.array_equal(a[key], b[key]))

    def test_construction_is_deterministic(self):
        again = build_pairs(self.frames, per_group=6, controls=4, seed=5)
        self.assertEqual(
            [spec.to_json() for spec in self.pairs.specs],
            [spec.to_json() for spec in again.specs],
        )

    def test_round_trip_through_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.pairs.save(tmp)
            loaded = PairSet.load(tmp)
        self.assertEqual(
            [spec.to_json() for spec in loaded.specs],
            [spec.to_json() for spec in self.pairs.specs],
        )
        for key in GRAPH_KEYS:
            self.assertTrue(
                np.array_equal(loaded.edited.fields[key], self.pairs.edited.fields[key])
            )
        loaded.verify_against(self.frames)


class Verification(unittest.TestCase):
    """The verifier has to reject what it is there to catch."""

    def setUp(self):
        self.frames = make_frames(8, seed=2)
        self.pairs = build_pairs(self.frames, per_group=2, controls=1, seed=2)
        self.spec = next(s for s in self.pairs.specs if s.group == "absolute")
        self.index = self.pairs.specs.index(self.spec)

    def _members(self):
        return self.frames.frame(self.spec.source), self.pairs.edited.frame(self.index)

    def test_a_second_change_is_rejected(self):
        a, b = self._members()
        b["graph_node_centroid"][0, 0] += np.float32(0.01)
        with self.assertRaises(PairError):
            verify_pair(a, b, self.spec)

    def test_an_extra_cell_in_the_same_field_is_rejected(self):
        a, b = self._members()
        row = next(r for r in range(b["graph_edge_rel"].size) if int(b["graph_edge_rel"][r]))
        row = row if row != int(self.spec.positions[0][0]) else row + 1
        b["graph_edge_abs"][row] = (int(b["graph_edge_abs"][row]) % 3) + 1
        with self.assertRaises(PairError):
            verify_pair(a, b, self.spec)

    def test_a_missing_change_is_rejected(self):
        a, _b = self._members()
        with self.assertRaises(PairError):
            verify_pair(a, a, self.spec)

    def test_an_illegal_label_is_rejected(self):
        a, b = self._members()
        row = int(self.spec.positions[0][0])
        mask = legal_absolute_mask()
        relation = int(a["graph_edge_rel"][row])
        illegal = int(np.flatnonzero(~mask[relation])[1])       # [0] is padding
        b["graph_edge_abs"][row] = illegal
        spec = PairSpec(
            name=self.spec.name, group=self.spec.group, source=self.spec.source,
            edited_field="graph_edge_abs", positions=self.spec.positions,
            before=self.spec.before, after=(float(illegal),), note=self.spec.note,
        )
        with self.assertRaises(PairError):
            verify_pair(a, b, spec)

    def test_a_padded_row_edit_is_rejected(self):
        a = self.frames.frame(0)
        pad = int(np.flatnonzero(a["graph_edge_rel"] == 0)[0])
        b = {key: np.array(value, copy=True) for key, value in a.items()}
        b["graph_edge_abs"][pad] = 1
        spec = PairSpec(
            name="bogus", group="absolute", source=0, edited_field="graph_edge_abs",
            positions=((pad,),), before=(float(a["graph_edge_abs"][pad]),), after=(1.0,),
            note="edit on a padded row",
        )
        with self.assertRaises(PairError):
            verify_pair(a, b, spec)

    def test_a_control_that_moved_is_rejected(self):
        spec = next(s for s in self.pairs.specs if s.group == CONTROL_GROUP)
        a = self.frames.frame(spec.source)
        b = {key: np.array(value, copy=True) for key, value in a.items()}
        b["graph_edge_temp"][0] = (int(b["graph_edge_temp"][0]) % 5) + 1
        with self.assertRaises(PairError):
            verify_pair(a, b, spec)


class SharedTables(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed")
    def test_legality_mask_matches_the_decoder(self):
        """One definition of "legal label", not two.

        ``SimpleGraphDecoder`` masks its absolute logits with this mask; an edit
        to a label the decoder scores at -1e9 would be measuring a different
        thing than an edit to one it can predict.
        """
        from graph import _relation_masks

        mask = legal_absolute_mask()
        decoder = _relation_masks(mask.shape[0], mask.shape[1]).numpy()
        self.assertTrue(np.array_equal(mask, decoder))


if __name__ == "__main__":
    unittest.main()
