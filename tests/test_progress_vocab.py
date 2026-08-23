"""The progress table against the graph vocabulary, without torch.

`progress.py` names relations and labels by integer id, and those ids belong to
the graph contract rather than to progress. When the directional support/contain
labels were added, every id from `very-near` upward shifted by two and the
hardcoded table went on scoring the wrong labels in silence -- `ProgressScorer`
validates only `0 < label < n_abs`, so nothing raised.

These tests exist because the ones that did catch it needed torch and therefore
only ran on the server. The scorer's contraction is a weighted sum over one-hot
rows, so it can be replicated here exactly.
"""

import unittest
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from scenegraph.adapters.graph_vocab import (
    build_absolute_vocab,
    build_relation_vocab,
)


@dataclass(frozen=True)
class _Stage:
    name: str
    relation: int
    labels: Tuple[int, ...]
    weight: float


def _progress_tables() -> dict:
    """`progress.py`'s module-level tables, evaluated without importing torch."""
    src = open("progress.py", encoding="utf-8").read()
    ns = {
        "build_absolute_vocab": build_absolute_vocab,
        "build_relation_vocab": build_relation_vocab,
        "ProgressStage": _Stage,
        "Tuple": Tuple,
    }
    exec(src[src.index("_REL = build_relation_vocab()"):
             src.index("@dataclass(frozen=True)")], ns)
    exec(src[src.index("PICK_STAGES: tuple"):src.index("def load_stages")], ns)
    return ns


def _potential(stages, n_abs: int, per_relation: Dict[int, int]) -> float:
    """`ProgressScorer.potential` on one-hot rows: `sum_k w_k * p_k`."""
    relations = []
    for stage in stages:
        if stage.relation not in relations:
            relations.append(stage.relation)
    total = sum(stage.weight for stage in stages)
    weights = np.zeros((len(relations), n_abs))
    for stage in stages:
        for label in stage.labels:
            weights[relations.index(stage.relation), label] += stage.weight / total
    probs = np.zeros_like(weights)
    for row, relation in enumerate(relations):
        probs[row, per_relation[relation]] = 1.0
    return float((probs * weights).sum())


class ProgressIdTest(unittest.TestCase):
    def setUp(self):
        self.ns = _progress_tables()

    def test_every_label_id_matches_the_vocabulary(self):
        absolute = build_absolute_vocab()
        for label in absolute.token_to_id:
            const = "ABS_" + label.upper().replace("-", "_")
            self.assertEqual(self.ns[const], absolute.encode(label), msg=const)

    def test_every_relation_id_matches_the_vocabulary(self):
        relation = build_relation_vocab()
        for name in relation.token_to_id:
            const = {
                "planar-distance": "REL_PLANAR_DISTANCE",
                "height-offset": "REL_HEIGHT_OFFSET",
                "grasp-compatibility": "REL_GRASP_COMPAT",
                "contact-compatibility": "REL_CONTACT_COMPAT",
                "support-compatibility": "REL_SUPPORT_COMPAT",
                "contain-compatibility": "REL_CONTAIN_COMPAT",
            }.get(name, "REL_" + name.upper().replace("-", "_"))
            if const not in self.ns:
                continue          # padding has no constant
            self.assertEqual(self.ns[const], relation.encode(name), msg=const)

    def test_n_abs_is_the_vocabulary_width(self):
        self.assertEqual(self.ns["N_ABS"], len(build_absolute_vocab()))

    def test_the_configured_n_abs_agrees(self):
        import yaml
        with open("configs/model/_base_.yaml") as handle:
            graph = yaml.safe_load(handle)["graph"]
        self.assertEqual(graph["n_abs"], len(build_absolute_vocab()))
        self.assertEqual(graph["n_rel"], len(build_relation_vocab()))


class PickLadderTest(unittest.TestCase):
    """The invariant the torch tests assert, checkable locally."""

    def setUp(self):
        self.ns = _progress_tables()
        self.stages = self.ns["PICK_STAGES"]
        self.n_abs = self.ns["N_ABS"]

    def _labels(self, **names):
        return {self.ns[rel]: self.ns[label] for rel, label in names.items()}

    def test_stage_weights_sum_to_one(self):
        self.assertAlmostEqual(
            sum(stage.weight for stage in self.stages), 1.0, places=9)

    def test_the_best_joint_state_scores_exactly_one(self):
        best = self._labels(
            REL_PLANAR_DISTANCE="ABS_VERY_NEAR",
            REL_HEIGHT_OFFSET="ABS_LEVEL",
            REL_CONTACT_COMPAT="ABS_MATCH",
            REL_GRASP_COMPAT="ABS_MATCH",
            REL_CONTACT="ABS_HOLDS",
            REL_GRASP="ABS_HOLDS",
        )
        self.assertAlmostEqual(
            _potential(self.stages, self.n_abs, best), 1.0, places=9)

    def test_the_worst_joint_state_scores_zero(self):
        worst = self._labels(
            REL_PLANAR_DISTANCE="ABS_VERY_FAR",
            REL_HEIGHT_OFFSET="ABS_FAR_BELOW",
            REL_CONTACT_COMPAT="ABS_UNOBSERVED",
            REL_GRASP_COMPAT="ABS_UNOBSERVED",
            REL_CONTACT="ABS_NOT_HOLDS",
            REL_GRASP="ABS_NOT_HOLDS",
        )
        self.assertAlmostEqual(
            _potential(self.stages, self.n_abs, worst), 0.0, places=9)

    def test_shifted_ids_would_be_caught(self):
        """What the stale table actually scored: 0.5 at the saturated state,
        because only contact and grasp kept their ids."""
        stale = {5: 3, 6: 10, 8: 13, 7: 13, 1: 2, 2: 2}
        self.assertAlmostEqual(
            _potential(self.stages, self.n_abs, stale), 0.5, places=9)

    def test_every_rung_is_reachable(self):
        """A label set naming an id outside the vocabulary would score zero
        forever without raising."""
        width = len(build_absolute_vocab())
        for stage in self.stages:
            for label in stage.labels:
                self.assertTrue(0 < label < width, msg=f"{stage.name}:{label}")


if __name__ == "__main__":
    unittest.main()
