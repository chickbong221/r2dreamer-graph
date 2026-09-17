"""The centimetre bins are compared with measured distances, label by label."""

from __future__ import annotations

import unittest

import numpy as np

from ..evaluation.check_bins import grounding
from . import synthetic as syn
from .test_prompts import valid_bins


class Grounding(unittest.TestCase):
    def test_a_label_whose_measurements_fall_outside_its_range_is_flagged(self):
        spec = syn.graph_spec()
        annotation = syn.annotation(spec)
        n = annotation.n_frames
        names = ["ee", "banana:center", "lid:center", "pot:rim_center"]
        points = np.zeros((n, len(names), 3))
        points[:, 0] = [0.0, 0.0, 0.10]          # gripper 5 cm from the banana, 10 cm above it
        points[:, 1] = [0.05, 0.0, 0.0]
        points[:, 2] = [0.50, 0.0, 0.0]
        points[:, 3] = [0.30, 0.0, 0.0]
        geometry = {"point_names": np.array(names), "points": points, "point_known": np.ones((n, len(names)), bool)}
        report = grounding(spec, [annotation], [geometry], valid_bins(spec))
        row = next(r for r in report["rows"] if r["fact"] == "planar-distance(ee, banana)")
        self.assertEqual(row["label"], "very-near")                # what the synthetic annotation says
        self.assertAlmostEqual(row["measured_cm"]["median"], 5.0)
        self.assertEqual(row["inside_fraction"], 0.0)
        self.assertTrue(row["median_outside_declared"])
        self.assertIn(row, report["flagged"])


if __name__ == "__main__":
    unittest.main()
